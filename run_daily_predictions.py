# run_daily_predictions.py
"""
Main script for generating comprehensive daily MLB predictions.

Produces a formatted report covering:
  • NRFI / YRFI first-inning markets
  • Full-game markets (Moneyline, Run Line, Game/Team Totals)
  • First 5 Innings (F5) markets
  • Player prop predictions (HR, Hit, TB, Run, RBI)

Usage:
    python run_daily_predictions.py
"""

import os
import sys
from datetime import datetime
import pytz
import pandas as pd

from live_scraper import get_todays_matchups
from game_markets_predictor import GameMarketsPredictor, format_odds, get_edge_rating

try:
    from feature_engineering import FeatureEngineer, LEAGUE_AVG_BATTER, LEAGUE_AVG_PITCHER
    _FE_AVAILABLE = True
except ImportError:
    _FE_AVAILABLE = False
    LEAGUE_AVG_BATTER = {
        "1B_Rate": 0.145, "2B_Rate": 0.045, "3B_Rate": 0.004, "HR_Rate": 0.030,
        "BB_Rate": 0.085, "K_Rate": 0.225, "R_Conv": 0.310, "RBI_Conv": 0.150,
        "SB_Conv": 0.050, "Barrel_Rate": 0.080, "xwOBA": 0.320,
        "Archetype": "Balanced", "Hand": "R",
    }
    LEAGUE_AVG_PITCHER = {
        "CALC_HR9": 1.25, "K_Rate": 0.22, "BB_Rate": 0.08, "H_Rate": 0.24, "BF_per_Start": 22
    }

# ---------------------------------------------------------------------------
# Load enhanced XGBoost models (market-specific)
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

MARKETS = ["HR", "Hit", "TB", "Run", "RBI"]

def _load_enhanced_models():
    models = {}
    feature_cols = []
    if not _XGB:
        return models, feature_cols
    feat_path = "enhanced_model_features.txt"
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            feature_cols = f.read().strip().split(",")
    for market in MARKETS:
        path = f"enhanced_model_{market.lower()}.json"
        if os.path.exists(path):
            m = xgb.XGBClassifier()
            m.load_model(path)
            models[market] = m
    return models, feature_cols


ENHANCED_MODELS, ENHANCED_COLS = _load_enhanced_models()

# Fallback to legacy single model
_LEGACY_BRAIN = None
_LEGACY_COLS = []
if not ENHANCED_MODELS and _XGB:
    try:
        _LEGACY_BRAIN = xgb.XGBClassifier()
        _LEGACY_BRAIN.load_model("mlb_xgboost_brain.json")
        with open("xgboost_columns.txt") as f:
            _LEGACY_COLS = f.read().strip().split(",")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_model(market: str, feature_vector: dict, raw_prob: float) -> float:
    """Apply the best available model to adjust the raw Monte Carlo probability."""
    if ENHANCED_MODELS.get(market) and ENHANCED_COLS:
        import pandas as pd
        row = {col: feature_vector.get(col, 0) for col in ENHANCED_COLS}
        X = pd.DataFrame([row])
        try:
            return float(ENHANCED_MODELS[market].predict_proba(X)[0, 1])
        except Exception:
            pass
    if _LEGACY_BRAIN and _LEGACY_COLS:
        import pandas as pd
        row = {col: 0 for col in _LEGACY_COLS}
        for k, v in feature_vector.items():
            if k in row:
                row[k] = v
        X = pd.DataFrame([row])
        try:
            return float(_LEGACY_BRAIN.predict_proba(X)[0, 1])
        except Exception:
            pass
    return raw_prob


def _recommendation(nrfi_prob: float) -> str:
    if nrfi_prob >= 0.65:
        return "🔥🔥🔥 STRONG NRFI"
    if nrfi_prob >= 0.58:
        return "🔥🔥 LEAN NRFI"
    if nrfi_prob >= 0.52:
        return "🔥 SLIGHT NRFI"
    if nrfi_prob <= 0.35:
        return "💥💥💥 STRONG YRFI"
    if nrfi_prob <= 0.42:
        return "💥💥 LEAN YRFI"
    if nrfi_prob <= 0.48:
        return "💥 SLIGHT YRFI"
    return "➖ NEUTRAL (No Edge)"


def _format_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


# ---------------------------------------------------------------------------
# Player prop predictions
# ---------------------------------------------------------------------------

def _predict_player_props(matchups: list, gmp: GameMarketsPredictor) -> list:
    """
    Run Monte Carlo simulations for each batter in each matchup.
    Returns list of prop-result dicts.
    """
    # Import the simulation engine from daily_bets
    try:
        from daily_bets import (
            get_prop_matrices, match_player_name,
            simulate_full_game_with_archetypes, generate_pitcher_profile,
            PARK_FACTORS as LEGACY_PARK,
        )
        batters_db, pitchers_db = get_prop_matrices()
    except Exception as exc:
        print(f"[!] Could not load prop matrices: {exc}")
        return []

    SIM_GAMES = 1_000  # Reduced from 10,000 to prevent timeout
    import random, numpy as np
    results = []
    batter_keys = list(batters_db.keys())

    for m in matchups:
        stadium = m.get("home_stadium", "Unknown")
        park_hr, park_avg = LEGACY_PARK.get(stadium, [1.0, 1.0])
        p_name = m.get("opposing_pitcher", "TBD")
        p_hand = m.get("opposing_pitcher_hand", "R")
        p_stats = pitchers_db.get(match_player_name(p_name, list(pitchers_db.keys())), LEAGUE_AVG_PITCHER)
        p_hr9 = p_stats["CALC_HR9"]
        p_archetype, _ = generate_pitcher_profile(p_hr9)
        w = m.get("weather", {"temp": 72, "wind_speed": 0, "wind_dir": "none"})
        w_boost = 1 + ((w["temp"] - 70) * 0.01) + (
            (w["wind_speed"] / 5) * 0.05 if w["wind_dir"] == "out" else 0
        )

        lineup_stats = [
            {**batters_db.get(match_player_name(b["name"], batter_keys), LEAGUE_AVG_BATTER).copy(),
             "Hand": b["hand"], "Name": b["name"]}
            for b in m["lineup"]
        ]

        for order_index, b in enumerate(lineup_stats):
            has_platoon = (b["Hand"] == "S") or (p_hand != b["Hand"])
            tracker = {"HR_1": 0, "H_1": 0, "TB_2": 0, "R_1": 0, "RBI_1": 0}
            for _ in range(SIM_GAMES):
                hr, hits, tb, runs, rbis, *_ = simulate_full_game_with_archetypes(
                    b, p_hr9, p_hand, w_boost, park_hr, park_avg, order_index
                )
                if hr >= 1: tracker["HR_1"] += 1
                if hits >= 1: tracker["H_1"] += 1
                if tb >= 2: tracker["TB_2"] += 1
                if runs >= 1: tracker["R_1"] += 1
                if rbis >= 1: tracker["RBI_1"] += 1

            raw = {
                "HR": tracker["HR_1"] / SIM_GAMES,
                "Hit": tracker["H_1"] / SIM_GAMES,
                "TB": tracker["TB_2"] / SIM_GAMES,
                "Run": tracker["R_1"] / SIM_GAMES,
                "RBI": tracker["RBI_1"] / SIM_GAMES,
            }

            feature_ctx = {
                "Temp": w["temp"], "Wind_Speed": w["wind_speed"],
                "Lineup_Spot": order_index + 1,
                "Batter_xwOBA": b.get("xwOBA", 0.320),
                "Pitcher_HR9": p_hr9,
                "Platoon_Adv": 1 if has_platoon else 0,
                "Batter_Archetype_Balanced": int(b.get("Archetype", "Balanced") == "Balanced"),
                "Batter_Archetype_Contact": int(b.get("Archetype", "Balanced") == "Contact"),
                "Batter_Archetype_Slugger": int(b.get("Archetype", "Balanced") == "Slugger"),
                "Pitcher_Archetype_Balanced": int(p_archetype == "Balanced"),
                "Pitcher_Archetype_Power": int(p_archetype == "Power"),
                "Pitcher_Archetype_Spin": int(p_archetype == "Spin"),
            }

            props = {}
            for market, raw_p in raw.items():
                fv = {**feature_ctx, "Prob": raw_p, f"Market_{market}": 1}
                props[market] = _apply_model(market, fv, raw_p)

            results.append({
                "team": m["team"],
                "stadium": stadium,
                "weather": w,
                "pitcher": p_name,
                "pitcher_hand": p_hand,
                "pitcher_archetype": p_archetype,
                "order": order_index + 1,
                "name": b["Name"],
                "hand": b["Hand"],
                "archetype": b.get("Archetype", "Balanced"),
                "platoon": has_platoon,
                "props": props,
                "game_time": m.get("game_time", "TBD"),
            })

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_game_report(
    away_team: str, home_team: str, stadium: str,
    weather: dict, away_pitcher: str, home_pitcher: str,
    away_pitcher_hand: str, home_pitcher_hand: str,
    nrfi_result: dict, game_result: dict,
    game_time: str = "",
) -> str:
    lines = []
    W = 70

    temp = weather.get("temp", 72)
    wind = weather.get("wind_speed", 0)
    wind_dir = weather.get("wind_dir", "none")

    lines.append("=" * W)
    lines.append(f"🏟️  {away_team.upper()} @ {home_team.upper()}")
    lines.append(f"⏰ {game_time} | 📍 {stadium}")
    lines.append(f"🌡️ {temp}°F | 💨 {wind}mph {wind_dir}")
    lines.append(f"⚾ Pitchers: {away_pitcher} ({away_pitcher_hand}) vs {home_pitcher} ({home_pitcher_hand})")
    lines.append("=" * W)

    # ---- NRFI section ----
    nrfi_p = nrfi_result.get("nrfi_prob", 0.5)
    yrfi_p = nrfi_result.get("yrfi_prob", 0.5)
    rec = _recommendation(nrfi_p)

    lines.append("┌" + "─" * (W - 2) + "┐")
    lines.append("│" + "   🥇 FIRST INNING MARKETS".center(W - 2) + "│")
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append(f"│  NRFI (No Run 1st)  : {_format_pct(nrfi_p):>6}  ({nrfi_result.get('nrfi_odds', 'N/A'):>6})".ljust(W - 1) + "│")
    lines.append(f"│  YRFI (Yes Run 1st) : {_format_pct(yrfi_p):>6}  ({nrfi_result.get('yrfi_odds', 'N/A'):>6})".ljust(W - 1) + "│")
    lines.append(f"│  Recommendation     : {rec}".ljust(W - 1) + "│")
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append(f"│  Away scores 1st: {_format_pct(nrfi_result.get('away_scores_prob', 0)):>6}  |  Home scores 1st: {_format_pct(nrfi_result.get('home_scores_prob', 0)):>6}".ljust(W - 1) + "│")
    lines.append(f"│  Pitcher Scores: Away {nrfi_result.get('away_pitcher_score', 0):.2f} | Home {nrfi_result.get('home_pitcher_score', 0):.2f}".ljust(W - 1) + "│")
    lines.append("└" + "─" * (W - 2) + "┘")
    lines.append("")

    # ---- Full game markets ----
    lines.append("┌" + "─" * (W - 2) + "┐")
    lines.append("│" + "   📊 FULL GAME MARKETS".center(W - 2) + "│")
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append("│  MONEYLINE".ljust(W - 1) + "│")
    lines.append(
        f"│    Away: {_format_pct(game_result['away_ml_prob'])} ({game_result['away_ml_odds']}) | Home: {_format_pct(game_result['home_ml_prob'])} ({game_result['home_ml_odds']})".ljust(W - 1) + "│"
    )
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append("│  RUN LINE (-1.5)".ljust(W - 1) + "│")
    lines.append(
        f"│    Away: {_format_pct(game_result['away_rl_prob'])} ({game_result['away_rl_odds']}) | Home: {_format_pct(game_result['home_rl_prob'])} ({game_result['home_rl_odds']})".ljust(W - 1) + "│"
    )
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append(f"│  GAME TOTAL ({game_result['game_total_line']})".ljust(W - 1) + "│")
    lines.append(
        f"│    Over: {_format_pct(game_result['game_total_over'])} | Under: {_format_pct(game_result['game_total_under'])} | Mean: {game_result['game_total_mean']:.1f} runs".ljust(W - 1) + "│"
    )
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append("│  TEAM TOTALS".ljust(W - 1) + "│")
    lines.append(
        f"│    Away O/U {game_result['away_tt_line']}: Over {_format_pct(game_result['away_tt_over'])} | Under {_format_pct(game_result['away_tt_under'])}".ljust(W - 1) + "│"
    )
    lines.append(
        f"│    Home O/U {game_result['home_tt_line']}: Over {_format_pct(game_result['home_tt_over'])} | Under {_format_pct(game_result['home_tt_under'])}".ljust(W - 1) + "│"
    )
    lines.append("└" + "─" * (W - 2) + "┘")
    lines.append("")

    # ---- F5 markets ----
    lines.append("┌" + "─" * (W - 2) + "┐")
    lines.append("│" + "   ⏱️  FIRST 5 INNINGS (F5)".center(W - 2) + "│")
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append("│  F5 MONEYLINE".ljust(W - 1) + "│")
    lines.append(
        f"│    Away: {_format_pct(game_result['f5_away_prob'])} | Home: {_format_pct(game_result['f5_home_prob'])} | Tie: {_format_pct(game_result['f5_tie_prob'])}".ljust(W - 1) + "│"
    )
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append(f"│  F5 TOTAL ({game_result['f5_total_line']})".ljust(W - 1) + "│")
    lines.append(
        f"│    Over: {_format_pct(game_result['f5_total_over'])} | Under: {_format_pct(game_result['f5_total_under'])}".ljust(W - 1) + "│"
    )
    lines.append("└" + "─" * (W - 2) + "┘")
    lines.append("")

    return "\n".join(lines)


def _format_props_section(props_list: list) -> str:
    """Format player props for one team's matchup."""
    if not props_list:
        return ""
    lines = []
    m0 = props_list[0]
    game_time = m0.get('game_time', 'TBD')
    lines.append(f"{'=' * 70}")
    lines.append(f"⏰ {game_time} | PLAYER PROPS: {m0['team']} vs {m0['pitcher']} ({m0['pitcher_hand']}HP - {m0['pitcher_archetype']} Pitcher)")
    lines.append(f"ENV: {m0['stadium']} | {m0['weather']['temp']}°F | Wind: {m0['weather']['wind_speed']}mph {m0['weather']['wind_dir']}")
    lines.append(f"{'=' * 70}")

    for batter in props_list:
        platoon_tag = "🔥 PLATOON ADV" if batter["platoon"] else "❄️ NO ADV"
        lines.append(f"\n> {batter['order']}. {batter['name'].upper()} ({batter['hand']} | {batter['archetype']} | {platoon_tag})")
        lines.append("  MARKET          | TRUE PROB | FAIR ODDS")
        lines.append("  " + "-" * 48)
        p = batter["props"]
        for market, label in [("HR", "To Hit a HR     "), ("Hit", "To Record 1+ Hit"),
                               ("TB", "To Record 2+ TB "), ("Run", "To Record 1+ Run"),
                               ("RBI", "To Record 1+ RBI")]:
            prob = p.get(market, 0.0)
            lines.append(f"  {label}| {prob * 100:>8.1f}% | {format_odds(prob):>9}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_daily_predictions():
    eastern = pytz.timezone("US/Eastern")
    today_str = datetime.now(eastern).strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"   MLB ENHANCED PREDICTIONS - {today_str}")
    print("=" * 70)

    # Init predictors
    gmp = GameMarketsPredictor()

    print("\n[1/3] Fetching today's matchups…")
    matchups = get_todays_matchups()
    if not matchups:
        print("[!] No matchups found for today.")
        return

    # Pair away/home teams
    paired = {}
    for m in matchups:
        stadium = m["home_stadium"]
        if stadium not in paired:
            paired[stadium] = {}
        # The team that is "at home" in the matchup list is the one whose
        # home_stadium matches their own team name — but the scraper gives us
        # one entry per batting team.  We need both sides.
        # Heuristic: determine home vs away by checking if team name appears in stadium.
        paired[stadium][m["team"]] = m

    print("\n[2/3] Running game simulations…")
    report_parts = []
    header = (
        "=" * 70 + "\n"
        f"MLB ENHANCED PREDICTIONS - {today_str}\n"
        + "=" * 70 + "\n"
    )
    report_parts.append(header)

    props_results = _predict_player_props(matchups, gmp)

    # Build a dict: stadium → list of team matchup dicts
    games_by_stadium: dict = {}
    for m in matchups:
        stadium = m["home_stadium"]
        games_by_stadium.setdefault(stadium, []).append(m)

    for stadium, teams in games_by_stadium.items():
        if len(teams) < 2:
            # We only have one side — do our best
            m = teams[0]
            away_team = m["team"]
            home_team = "Unknown"
            away_pitcher = m["opposing_pitcher"]
            home_pitcher = "TBD"
            away_ph = m.get("opposing_pitcher_hand", "R")
            home_ph = "R"
            away_lineup = m["lineup"]
            home_lineup = []
            weather = m.get("weather", {"temp": 72, "wind_speed": 0, "wind_dir": "none"})
        else:
            m0, m1 = teams[0], teams[1]
            # m0 bats against m1's pitcher → m1 is the home team's pitcher side
            away_team = m0["team"]
            home_team = m1["team"]
            away_pitcher = m0["opposing_pitcher"]
            home_pitcher = m1["opposing_pitcher"]
            away_ph = m0.get("opposing_pitcher_hand", "R")
            home_ph = m1.get("opposing_pitcher_hand", "R")
            away_lineup = m0["lineup"]
            home_lineup = m1["lineup"]
            weather = m0.get("weather", {"temp": 72, "wind_speed": 0, "wind_dir": "none"})

        # NRFI prediction
        nrfi_result = gmp.predict_nrfi_probability(
            away_lineup, home_lineup,
            away_pitcher, home_pitcher,
            stadium=stadium,
            away_pitcher_hand=away_ph,
            home_pitcher_hand=home_ph,
            weather=weather,
            n_simulations=1_500,  # Reduced from 5,000 to prevent timeout
        )

        # Full game prediction
        game_result = gmp.predict_full_game(
            away_lineup, home_lineup,
            away_pitcher, home_pitcher,
            stadium=stadium,
            away_pitcher_hand=away_ph,
            home_pitcher_hand=home_ph,
            weather=weather,
            n_simulations=1_500,  # Reduced from 5,000 to prevent timeout
        )

        # Get game_time from one of the matchups
        game_time = teams[0].get('game_time', 'TBD')

        game_section = _format_game_report(
            away_team, home_team, stadium, weather,
            away_pitcher, home_pitcher, away_ph, home_ph,
            nrfi_result, game_result,
            game_time=game_time,
        )
        report_parts.append(game_section)

        # Player props for this game
        for m in teams:
            team_props = [p for p in props_results if p["team"] == m["team"]]
            if team_props:
                report_parts.append(_format_props_section(team_props))
                report_parts.append("")

    print("\n[3/3] Generating report…")
    full_report = "\n".join(report_parts)
    print(full_report)

    out_path = "enhanced_predictions_report.txt"
    with open(out_path, "w") as f:
        f.write(full_report)
    print(f"\n[SUCCESS] Report saved to {out_path}")


if __name__ == "__main__":
    run_daily_predictions()

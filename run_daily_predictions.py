# run_daily_predictions.py
"""
Main script for generating comprehensive daily MLB predictions.
Saves results to enhanced_predictions_report.txt and mlb_predictions.db
"""

import os
import sys
import sqlite3
from datetime import datetime
import pytz
import pandas as pd

from live_scraper import get_todays_matchups
from game_markets_predictor import GameMarketsPredictor, format_odds, get_edge_rating

# ---------------------------------------------------------------------------
# League Average Fallbacks
# ---------------------------------------------------------------------------
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
# Load enhanced XGBoost models
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


# ---------------------------------------------------------------------------
# Database Logic (With Schema Fix)
# ---------------------------------------------------------------------------

def _write_to_sqlite(props_results, game_ledger_data):
    """Saves predictions into the SQLite database with duplicate protection."""
    db_path = 'mlb_predictions.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Ensure Table Schemas
    cursor.execute('''CREATE TABLE IF NOT EXISTS game_predictions (
        id INTEGER PRIMARY KEY, date TEXT, away_team TEXT, home_team TEXT, 
        stadium TEXT, market TEXT, probability REAL, fair_odds TEXT, game_total_line REAL
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS player_predictions (
        id INTEGER PRIMARY KEY, date TEXT, player_name TEXT, team TEXT, 
        opponent TEXT, market TEXT, probability REAL, lineup_spot INTEGER
    )''')

    # 2. Schema Verification: Ensure 'date' column exists to prevent OperationalError
    cursor.execute("PRAGMA table_info(game_predictions)")
    cols = [i[1] for i in cursor.fetchall()]
    if 'date' not in cols:
        print("[!] DB Schema Outdated. Dropping and Recreating 'game_predictions'...")
        cursor.execute("DROP TABLE game_predictions")
        cursor.execute('''CREATE TABLE game_predictions (
            id INTEGER PRIMARY KEY, date TEXT, away_team TEXT, home_team TEXT, 
            stadium TEXT, market TEXT, probability REAL, fair_odds TEXT, game_total_line REAL
        )''')

    run_date = datetime.now().strftime('%Y-%m-%d')

    # 3. Game Data Insert
    for entry in game_ledger_data:
        gr, nr = entry['game_result'], entry['nrfi_result']
        gt_line = gr.get('game_total_line', 8.5)

        cursor.execute("DELETE FROM game_predictions WHERE date=? AND away_team=? AND home_team=?",
                       (run_date, entry['away_team'], entry['home_team']))

        markets = [
            ('ML_Away', gr.get('away_ml_prob'), gr.get('away_ml_odds')),
            ('ML_Home', gr.get('home_ml_prob'), gr.get('home_ml_odds')),
            ('NRFI', nr.get('nrfi_prob'), nr.get('nrfi_odds')),
            ('Total_Over', gr.get('game_total_over'), "N/A"),
            ('Total_Under', gr.get('game_total_under'), "N/A"),
            ('Expected_Total', gr.get('game_total_mean'), "N/A")
        ]

        for m_name, prob, odds in markets:
            cursor.execute('''INSERT INTO game_predictions 
                (date, away_team, home_team, stadium, market, probability, fair_odds, game_total_line)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                           (run_date, entry['away_team'], entry['home_team'], entry['stadium'], m_name, float(prob),
                            str(odds), float(gt_line)))

    # 4. Player Prop Insert
    for prop in props_results:
        cursor.execute("DELETE FROM player_predictions WHERE date=? AND player_name=?", (run_date, prop['name']))
        for m_name, prob in prop['props'].items():
            cursor.execute('''INSERT INTO player_predictions 
                (date, player_name, team, opponent, market, probability, lineup_spot)
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (run_date, prop['name'], prop['team'], prop['pitcher'], m_name, float(prob), prop['order']))

    conn.commit()
    conn.close()
    print(f" [SUCCESS] Database updated for {run_date}")


# ---------------------------------------------------------------------------
# Formatting & Logic Helpers
# ---------------------------------------------------------------------------

def _recommendation(nrfi_prob: float) -> str:
    if nrfi_prob >= 0.65: return "🔥🔥🔥 STRONG NRFI"
    if nrfi_prob >= 0.58: return "🔥🔥 LEAN NRFI"
    if nrfi_prob >= 0.52: return "🔥 SLIGHT NRFI"
    if nrfi_prob <= 0.35: return "💥💥💥 STRONG YRFI"
    if nrfi_prob <= 0.42: return "💥💥 LEAN YRFI"
    if nrfi_prob <= 0.48: return "💥 SLIGHT YRFI"
    return "➖ NEUTRAL"


def _format_game_report(away_t, home_t, stadium, weather, away_p, home_p, away_ph, home_ph, nrfi_r, game_r,
                        game_time) -> str:
    W = 70
    lines = [
        "=" * W, f"🏟️  {away_t.upper()} @ {home_t.upper()}",
        f"⏰ {game_time} | 📍 {stadium}",
        f"🌡️ {weather['temp']}°F | 💨 {weather['wind_speed']}mph {weather['wind_dir']}",
        f"⚾ Pitchers: {away_p} ({away_ph}) vs {home_p} ({home_ph})", "=" * W
    ]

    # Game Markets Section
    lines.append(f"┌{'─' * (W - 2)}┐")
    lines.append(f"│{'📊 GAME MARKETS (Full Game)'.center(W - 2)}│")
    lines.append(f"├{'─' * (W - 2)}┤")
    lines.append(
        f"│  ML: {away_t} {game_r['away_ml_prob'] * 100:.1f}% | {home_t} {game_r['home_ml_prob'] * 100:.1f}%".ljust(
            W - 1) + "│")
    gt = game_r.get('game_total_line', 8.5)
    lines.append(
        f"│  O/U {gt}: Over {game_r['game_total_over'] * 100:.1f}% | Under {game_r['game_total_under'] * 100:.1f}%".ljust(
            W - 1) + "│")
    lines.append(f"│  AI Projected Score: {game_r.get('game_total_mean', 0):.2f} runs".ljust(W - 1) + "│")
    lines.append(f"└{'─' * (W - 2)}┘")

    # NRFI Section
    lines.append(f"┌{'─' * (W - 2)}┐")
    lines.append(f"│{'🥇 FIRST INNING MARKETS'.center(W - 2)}│")
    lines.append(f"├{'─' * (W - 2)}┤")
    lines.append(
        f"│  NRFI: {nrfi_r.get('nrfi_prob', 0) * 100:.1f}% ({nrfi_r.get('nrfi_odds', 'N/A')})".ljust(W - 1) + "│")
    lines.append(f"│  Recommendation: {_recommendation(nrfi_r.get('nrfi_prob', 0.5))}".ljust(W - 1) + "│")
    lines.append(f"└{'─' * (W - 2)}┘\n")
    return "\n".join(lines)


def _format_props_section(props_list: list) -> str:
    if not props_list: return ""
    lines = ["-" * 40, f"PLAYER PROPS: {props_list[0]['team']}", "-" * 40]
    for b in props_list:
        lines.append(f"> {b['order']}. {b['name']} ({b['hand']})")
        lines.append(f"  Hit | {b['props']['Hit'] * 100:.1f}% | {format_odds(b['props']['Hit'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation & Execution
# ---------------------------------------------------------------------------

def _apply_model(market: str, feature_vector: dict, raw_prob: float) -> float:
    if ENHANCED_MODELS.get(market) and ENHANCED_COLS:
        row = {col: feature_vector.get(col, 0) for col in ENHANCED_COLS}
        try:
            return float(ENHANCED_MODELS[market].predict_proba(pd.DataFrame([row]))[0, 1])
        except:
            pass
    return raw_prob


def _predict_player_props(matchups: list, gmp: GameMarketsPredictor) -> list:
    try:
        from daily_bets import get_prop_matrices, match_player_name, simulate_full_game_with_archetypes, \
            PARK_FACTORS as PF
        batters_db, pitchers_db = get_prop_matrices()
    except:
        return []

    SIM_GAMES = 1000
    results = []
    key_map = {"HR": "HR_1", "Hit": "H_1", "TB": "TB_2", "Run": "R_1", "RBI": "RBI_1"}

    for m in matchups:
        stadium = m.get("home_stadium", "Unknown")
        park_hr, park_avg = PF.get(stadium, [1.0, 1.0])
        p_name, p_hand = m.get("opposing_pitcher", "TBD"), m.get("opposing_pitcher_hand", "R")
        p_stats = pitchers_db.get(match_player_name(p_name, list(pitchers_db.keys())), LEAGUE_AVG_PITCHER)
        p_hr9 = p_stats["CALC_HR9"]

        w = m.get("weather", {"temp": 72, "wind_speed": 0, "wind_dir": "none"})
        w_boost = 1 + ((w["temp"] - 70) * 0.01) + ((w["wind_speed"] / 5) * 0.05 if w["wind_dir"] == "out" else 0)

        for idx, b_info in enumerate(m["lineup"]):
            b_key = match_player_name(b_info['name'], list(batters_db.keys()))
            b = batters_db.get(b_key, LEAGUE_AVG_BATTER).copy()
            b["Hand"], b["Name"] = b_info["hand"], b_info["name"]

            tracker = {k: 0 for k in key_map.values()}
            for _ in range(SIM_GAMES):
                hr, hits, tb, runs, rbis, *_ = simulate_full_game_with_archetypes(b, p_hr9, p_hand, w_boost, park_hr,
                                                                                  park_avg, idx)
                if hr >= 1: tracker["HR_1"] += 1
                if hits >= 1: tracker["H_1"] += 1
                if tb >= 2: tracker["TB_2"] += 1
                if runs >= 1: tracker["R_1"] += 1
                if rbis >= 1: tracker["RBI_1"] += 1

            raw = {k: v / SIM_GAMES for k, v in tracker.items()}
            f_ctx = {"Temp": w["temp"], "Wind_Speed": w["wind_speed"], "Lineup_Spot": idx + 1,
                     "Platoon_Adv": 1 if (b["Hand"] != p_hand) else 0}
            props = {mkt: _apply_model(mkt, {**f_ctx, "Prob": raw[key_map[mkt]]}, raw[key_map[mkt]]) for mkt in MARKETS}
            results.append({"team": m["team"], "name": b["Name"], "order": idx + 1, "hand": b["Hand"], "props": props,
                            "pitcher": p_name})
    return results


def run_daily_predictions():
    eastern = pytz.timezone("US/Eastern")
    today_str = datetime.now(eastern).strftime("%Y-%m-%d")
    print(f"\n[!] Starting Enhanced Predictions for {today_str}...")

    gmp = GameMarketsPredictor()
    matchups = get_todays_matchups()
    if not matchups: return

    props_results = _predict_player_props(matchups, gmp)
    stadium_groups = {}
    for m in matchups: stadium_groups.setdefault(m["home_stadium"], []).append(m)

    report_parts = ["=" * 70 + f"\nMLB ENHANCED PREDICTIONS - {today_str}\n" + "=" * 70]
    _game_ledger_data = []

    for stadium, teams in stadium_groups.items():
        if len(teams) < 2: continue
        m0, m1 = teams[0], teams[1]
        w = m0.get("weather", {"temp": 72, "wind_speed": 0, "wind_dir": "none"})

        nrfi_res = gmp.predict_nrfi_probability(m0["lineup"], m1["lineup"], m0["opposing_pitcher"],
                                                m1["opposing_pitcher"], stadium, m0["opposing_pitcher_hand"],
                                                m1["opposing_pitcher_hand"], w, 1000)
        game_res = gmp.predict_full_game(m0["lineup"], m1["lineup"], m0["opposing_pitcher"], m1["opposing_pitcher"],
                                         stadium, m0["opposing_pitcher_hand"], m1["opposing_pitcher_hand"], w, 1000)

        _game_ledger_data.append({"stadium": stadium, "weather": w, "away_team": m0["team"], "home_team": m1["team"],
                                  "nrfi_result": nrfi_res, "game_result": game_res})
        report_parts.append(
            _format_game_report(m0["team"], m1["team"], stadium, w, m0["opposing_pitcher"], m1["opposing_pitcher"],
                                m0["opposing_pitcher_hand"], m1["opposing_pitcher_hand"], nrfi_res, game_res,
                                m0["game_time"]))

        for m in teams:
            t_props = [p for p in props_results if p["team"] == m["team"]]
            report_parts.append(_format_props_section(t_props))

    with open("enhanced_predictions_report.txt", "w") as f:
        f.write("\n".join(report_parts))

    _write_to_sqlite(props_results, _game_ledger_data)
    print("\n[SUCCESS] Pipeline Complete.")


if __name__ == "__main__":
    run_daily_predictions()
# results_tracker.py
"""
System 2 Judge: grades yesterday's predictions from prediction_ledger.csv against
real MLB outcomes. Computes per-market Brier scores, calibration analysis,
error pattern detection, writes training_feedback.json, and sends 2 Discord messages.
"""
import json
import os

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK", "")

# ---------------------------------------------------------------------------
# Per-market Brier benchmarks (lower = better)
# ---------------------------------------------------------------------------
MARKET_BENCHMARKS: dict = {
    "HR": 0.20,
    "Hit": 0.15,
    "TB": 0.18,
    "Run": 0.18,
    "RBI": 0.18,
    "NRFI": 0.22,
    "ML_Away": 0.24,
    "ML_Home": 0.24,
}
_DEFAULT_BENCHMARK = 0.24  # F5, game totals, team totals


def _get_benchmark(market: str) -> float:
    if market in MARKET_BENCHMARKS:
        return MARKET_BENCHMARKS[market]
    if market.startswith(("ML_", "F5_ML")):
        return 0.24
    return _DEFAULT_BENCHMARK


def _market_group(market: str) -> str:
    """Collapse variable markets (Over_8.5, etc.) to a canonical group name."""
    if market in ("HR", "Hit", "TB", "Run", "RBI", "NRFI", "ML_Away", "ML_Home"):
        return market
    if market.startswith("F5_ML"):
        return "F5_ML"
    if market.startswith("F5_Over"):
        return "F5_Total"
    if market.startswith("Over_"):
        return "Game_Total"
    if market.startswith("TeamTotal_Away"):
        return "TeamTotal_Away"
    if market.startswith("TeamTotal_Home"):
        return "TeamTotal_Home"
    return market


# ---------------------------------------------------------------------------
# MLB API helpers
# ---------------------------------------------------------------------------

def _fetch_linescore(game_pk: int) -> dict:
    """Fetch the linescore (inning-by-inning runs) for a single game."""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
    try:
        return requests.get(url, timeout=5).json()
    except Exception:
        return {}


def fetch_game_outcomes(yesterday_str: str) -> dict:
    """
    Returns a dict of game-level results keyed by game_pk.
    Each entry has: away_runs, home_runs, away_runs_1st, home_runs_1st,
    away_runs_f5, home_runs_f5, away_team, home_team.
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={yesterday_str}"
    print(f"Fetching official MLB game outcomes for {yesterday_str}...")
    try:
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"[!] API Error: {e}")
        return {}

    outcomes = {}
    if "dates" not in data or not data["dates"]:
        return outcomes

    for game in data["dates"][0].get("games", []):
        if game["status"]["statusCode"] not in ["F", "O"]:
            continue
        game_pk = game["gamePk"]
        away_team = game["teams"]["away"]["team"]["name"]
        home_team = game["teams"]["home"]["team"]["name"]
        linescore = _fetch_linescore(game_pk)
        innings = linescore.get("innings", [])
        away_total = linescore.get("teams", {}).get("away", {}).get("runs", 0) or 0
        home_total = linescore.get("teams", {}).get("home", {}).get("runs", 0) or 0
        away_1st = home_1st = 0
        if innings:
            away_1st = innings[0].get("away", {}).get("runs", 0) or 0
            home_1st = innings[0].get("home", {}).get("runs", 0) or 0
        away_f5 = sum(inn.get("away", {}).get("runs", 0) or 0 for inn in innings[:5])
        home_f5 = sum(inn.get("home", {}).get("runs", 0) or 0 for inn in innings[:5])
        outcomes[game_pk] = {
            "away_team": away_team,
            "home_team": home_team,
            "away_runs": int(away_total),
            "home_runs": int(home_total),
            "away_runs_1st": int(away_1st),
            "home_runs_1st": int(home_1st),
            "away_runs_f5": int(away_f5),
            "home_runs_f5": int(home_f5),
        }
    return outcomes


def fetch_yesterdays_boxscores(yesterday_str: str) -> dict:
    """Pings the MLB API for yesterday's games and loops individual boxscores."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={yesterday_str}"
    print(f"Fetching official MLB box scores for {yesterday_str}...")
    try:
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"[!] API Error: {e}")
        return {}

    actuals: dict = {}
    if "dates" not in data or not data["dates"]:
        return actuals

    for game in data["dates"][0].get("games", []):
        if game["status"]["statusCode"] not in ["F", "O"]:
            continue
        game_pk = game["gamePk"]
        box_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        try:
            box_data = requests.get(box_url, timeout=5).json()
        except Exception:
            continue
        boxscore = box_data.get("teams", {})
        for team_side in ["away", "home"]:
            players = boxscore.get(team_side, {}).get("players", {})
            for pid, pdata in players.items():
                name = pdata["person"]["fullName"].replace("*", "").strip()
                stats = pdata.get("stats", {}).get("batting", {})
                if stats:
                    if name not in actuals:
                        actuals[name] = {"HR": 0, "Hit": 0, "TB": 0, "Run": 0, "RBI": 0}
                    actuals[name]["HR"] += stats.get("homeRuns", 0)
                    actuals[name]["Hit"] += stats.get("hits", 0)
                    actuals[name]["TB"] += stats.get("totalBases", 0)
                    actuals[name]["Run"] += stats.get("runs", 0)
                    actuals[name]["RBI"] += stats.get("rbi", 0)
    return actuals


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _match_game(away_t: str, home_t: str, game_outcomes: dict) -> "dict | None":
    """Fuzzy-match a (away, home) team pair against the API outcomes dict."""
    for go in game_outcomes.values():
        if go["away_team"] == away_t and go["home_team"] == home_t:
            return go
        if away_t and home_t:
            if (away_t in go["away_team"] or go["away_team"] in away_t) and \
               (home_t in go["home_team"] or go["home_team"] in home_t):
                return go
    return None


def _grade_actual(market: str, prob: float, matched: "dict | None",
                  p_stats: "dict | None") -> "tuple[float, int] | None":
    """Return (brier_score, actual_outcome) for one prediction row, or None if unresolvable."""
    actual: int | None = None

    if p_stats is not None:
        # Player prop
        if market == "HR":
            actual = int(p_stats.get("HR", 0) >= 1)
        elif market == "Hit":
            actual = int(p_stats.get("Hit", 0) >= 1)
        elif market == "TB":
            actual = int(p_stats.get("TB", 0) >= 2)
        elif market == "Run":
            actual = int(p_stats.get("Run", 0) >= 1)
        elif market == "RBI":
            actual = int(p_stats.get("RBI", 0) >= 1)
    elif matched is not None:
        # Game-level market
        if market == "NRFI":
            actual = int(matched["away_runs_1st"] == 0 and matched["home_runs_1st"] == 0)
        elif market == "ML_Away":
            actual = int(matched["away_runs"] > matched["home_runs"])
        elif market == "ML_Home":
            actual = int(matched["home_runs"] > matched["away_runs"])
        elif market.startswith("Over_"):
            try:
                line = float(market[len("Over_"):])
                actual = int(matched["away_runs"] + matched["home_runs"] > line)
            except ValueError:
                pass
        elif market == "F5_ML_Away":
            actual = int(matched.get("away_runs_f5", 0) > matched.get("home_runs_f5", 0))
        elif market == "F5_ML_Home":
            actual = int(matched.get("home_runs_f5", 0) > matched.get("away_runs_f5", 0))
        elif market.startswith("F5_Over_"):
            try:
                line = float(market[len("F5_Over_"):])
                actual = int(matched.get("away_runs_f5", 0) + matched.get("home_runs_f5", 0) > line)
            except ValueError:
                pass
        elif market.startswith("TeamTotal_Away_Over_"):
            try:
                line = float(market[len("TeamTotal_Away_Over_"):])
                actual = int(matched["away_runs"] > line)
            except ValueError:
                pass
        elif market.startswith("TeamTotal_Home_Over_"):
            try:
                line = float(market[len("TeamTotal_Home_Over_"):])
                actual = int(matched["home_runs"] > line)
            except ValueError:
                pass

    if actual is None:
        return None
    return (prob - actual) ** 2, actual


def grade_all_markets(df: pd.DataFrame, actuals: dict, game_outcomes: dict) -> pd.DataFrame:
    """Grade all prediction rows and return an enriched DataFrame with Actual_Outcome and Brier."""
    graded_rows = []
    for _, row in df.iterrows():
        player = str(row.get("Player", "")).replace("*", "").strip()
        market = str(row.get("Market", ""))
        prob = float(row.get("Prob", 0.5))

        if player == "GAME_TOTAL":
            away_t = str(row.get("Away_Team", ""))
            home_t = str(row.get("Home_Team", ""))
            matched = _match_game(away_t, home_t, game_outcomes)
            if matched is None:
                continue
            result = _grade_actual(market, prob, matched, None)
        else:
            p_stats = actuals.get(player)
            if p_stats is None:
                continue
            result = _grade_actual(market, prob, None, p_stats)

        if result is None:
            continue
        brier, actual_outcome = result
        graded_row = {k: v for k, v in row.items()}
        graded_row["Actual_Outcome"] = actual_outcome
        graded_row["Brier"] = brier
        graded_row["Market_Group"] = _market_group(market)
        graded_rows.append(graded_row)

    return pd.DataFrame(graded_rows) if graded_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def compute_calibration(graded_df: pd.DataFrame) -> dict:
    """Compute ECE, MCE, and per-bin calibration data."""
    if graded_df.empty:
        return {"ece": 0.0, "mce": 0.0, "worst_bin": "N/A", "bins": []}

    preds = graded_df["Prob"].values.astype(float)
    acts = graded_df["Actual_Outcome"].values.astype(float)
    bins_data = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        mask = (preds >= lo) & (preds < hi) if hi < 1.0 else (preds >= lo) & (preds <= hi)
        if mask.sum() == 0:
            continue
        mean_pred = float(preds[mask].mean())
        actual_rate = float(acts[mask].mean())
        count = int(mask.sum())
        bins_data.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "mean_pred": round(mean_pred, 4),
            "actual_rate": round(actual_rate, 4),
            "count": count,
        })

    if not bins_data:
        return {"ece": 0.0, "mce": 0.0, "worst_bin": "N/A", "bins": []}

    total = sum(b["count"] for b in bins_data)
    ece = sum(b["count"] * abs(b["mean_pred"] - b["actual_rate"]) for b in bins_data) / total
    worst = max(bins_data, key=lambda b: abs(b["mean_pred"] - b["actual_rate"]))
    mce = abs(worst["mean_pred"] - worst["actual_rate"])

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "worst_bin": worst["bin"],
        "bins": bins_data,
    }


# ---------------------------------------------------------------------------
# Error pattern detection
# ---------------------------------------------------------------------------

def compute_error_patterns(graded_df: pd.DataFrame) -> list:
    """Identify slices where Brier > 1.5× market average with n >= 5."""
    if graded_df.empty:
        return []

    # Compute per-market-group averages first
    market_avg: dict = {}
    for mg, mdf in graded_df.groupby("Market_Group"):
        market_avg[mg] = float(mdf["Brier"].mean())

    weak_slices = []

    def _check(dimension: str, value, subset: pd.DataFrame, market_group: str) -> None:
        if len(subset) < 5:
            return
        brier = float(subset["Brier"].mean())
        avg = market_avg.get(market_group, 0)
        if avg > 0 and brier > 1.5 * avg:
            weak_slices.append({
                "dimension": dimension,
                "value": str(value),
                "market": market_group,
                "brier": round(brier, 4),
                "n": len(subset),
            })

    # By Batter_Archetype × Market_Group
    for col in ("Batter_Archetype", "Pitcher_Archetype"):
        if col not in graded_df.columns:
            continue
        col_df = graded_df[graded_df[col].notna() & (graded_df[col] != "")]
        for (val, mg), sub in col_df.groupby([col, "Market_Group"]):
            _check(col, val, sub, mg)

    # By Platoon_Adv × Market_Group
    if "Platoon_Adv" in graded_df.columns:
        platoon_df = graded_df[pd.to_numeric(graded_df["Platoon_Adv"], errors="coerce").notna()].copy()
        platoon_df["Platoon_Adv"] = pd.to_numeric(platoon_df["Platoon_Adv"])
        for (val, mg), sub in platoon_df.groupby(["Platoon_Adv", "Market_Group"]):
            _check("Platoon_Adv", val, sub, mg)

    # By Lineup position group × Market_Group
    if "Lineup_Spot" in graded_df.columns:
        ls_df = graded_df[pd.to_numeric(graded_df["Lineup_Spot"], errors="coerce").notna()].copy()
        ls_df["Lineup_Spot"] = pd.to_numeric(ls_df["Lineup_Spot"])
        for group_label, lo, hi in [("1-3", 1, 3), ("4-6", 4, 6), ("7-9", 7, 9)]:
            mask = (ls_df["Lineup_Spot"] >= lo) & (ls_df["Lineup_Spot"] <= hi)
            for mg, sub in ls_df[mask].groupby("Market_Group"):
                _check("Lineup_Position", group_label, sub, mg)

    # By Temperature group × Market_Group
    if "Temp" in graded_df.columns:
        t_df = graded_df[pd.to_numeric(graded_df["Temp"], errors="coerce").notna()].copy()
        t_df["Temp"] = pd.to_numeric(t_df["Temp"])
        temp_groups = [
            ("cold<65", t_df["Temp"] < 65),
            ("mild65-80", (t_df["Temp"] >= 65) & (t_df["Temp"] <= 80)),
            ("hot>80", t_df["Temp"] > 80),
        ]
        for group_label, cond in temp_groups:
            for mg, sub in t_df[cond].groupby("Market_Group"):
                _check("Temperature", group_label, sub, mg)

    # By Stadium × Market_Group
    if "Stadium" in graded_df.columns:
        for (stad, mg), sub in graded_df.groupby(["Stadium", "Market_Group"]):
            _check("Stadium", stad, sub, mg)

    return weak_slices


# ---------------------------------------------------------------------------
# Discord helpers
# ---------------------------------------------------------------------------

def _send_discord_chunks(message: str, webhook: str) -> None:
    """Split a long message into ≤1900-char chunks and post each to Discord."""
    if not webhook:
        print("[!] No Discord webhook configured — skipping.")
        return
    limit = 1900
    chunks = []
    while len(message) > limit:
        idx = message.rfind("\n", 0, limit)
        if idx == -1:
            idx = limit
        chunks.append(message[:idx])
        message = message[idx:]
    chunks.append(message)
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            resp = requests.post(webhook, json={"content": chunk}, timeout=15)
            if resp.status_code not in (200, 204):
                print(f"[!] Discord returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[!] Discord send error: {e}")


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _build_report_card(date_str: str, graded_df: pd.DataFrame, calibration: dict,
                       weak_slices: list, prev_brier: float | None) -> str:
    """Build the Message 2 Performance Report Card."""
    total = len(graded_df)
    if total == 0:
        return f"📊 **MLB System 2 Report Card ({date_str})** 📊\nNo graded predictions available."

    overall_brier = float(graded_df["Brier"].mean())
    wins = int(((graded_df["Prob"] >= 0.5) == (graded_df["Actual_Outcome"] == 1)).sum())
    accuracy = wins / total * 100

    props_count = int((graded_df["Player"] != "GAME_TOTAL").sum())
    games_count = int(
        graded_df[graded_df["Player"] == "GAME_TOTAL"]
        .groupby(["Away_Team", "Home_Team", "Date"]).ngroups
    ) if "Away_Team" in graded_df.columns else 0

    lines = [
        f"📊 **MLB System 2 Report Card ({date_str})** 📊",
        "",
        "🎯 **OVERALL**",
        f"  Brier Score: {overall_brier:.4f} | Binary Accuracy: {accuracy:.1f}% | ECE: {calibration['ece']:.3f}",
        f"  Props Graded: {props_count} | Games Graded: {games_count}",
        "",
        "📈 **MARKET BREAKDOWN**",
        f"  {'Market':<10} | {'Brier':>6} | {'Bench':>6} | {'Status':<10} | {'n':>5}",
        "  " + "-" * 50,
    ]

    # Compute per-market-group brier
    for mg, mdf in sorted(graded_df.groupby("Market_Group")):
        b = float(mdf["Brier"].mean())
        bench = _get_benchmark(mg)
        status = "✅ OK" if b <= bench else "❌ UNDER"
        lines.append(f"  {mg:<10} | {b:>6.3f} | {bench:>6.3f} | {status:<10} | {len(mdf):>5}")

    lines += [
        "",
        "📐 **CALIBRATION**",
        f"  ECE: {calibration['ece']:.3f} {'✅' if calibration['ece'] < 0.05 else '⚠️'} (target: < 0.05)",
        f"  MCE: {calibration['mce']:.3f} (worst bin: {calibration['worst_bin']})",
        f"  {'Bin':<10} | {'Pred':>6} | {'Actual':>7} | {'Count':>6}",
        "  " + "-" * 40,
    ]
    for b in calibration.get("bins", []):
        lines.append(
            f"  {b['bin']:<10} | {b['mean_pred']:>6.3f} | {b['actual_rate']:>7.3f} | {b['count']:>6}"
        )

    if weak_slices:
        lines += ["", "⚠️ **WEAK SPOTS** (Brier > 1.5× market avg, n ≥ 5)"]
        for ws in weak_slices[:10]:  # Cap at 10 to keep message manageable
            lines.append(
                f"  {ws['dimension']}={ws['value']} + {ws['market']}: "
                f"Brier {ws['brier']:.3f} (n={ws['n']})"
            )

    if prev_brier is not None and prev_brier > 0:
        delta = prev_brier - overall_brier
        pct = abs(delta) / prev_brier * 100
        direction = "improved" if delta > 0 else "worsened"
        lines += ["", f"📊 **vs Previous**: Avg Brier {direction} by {abs(delta):.4f} ({pct:.1f}%) since last report"]

    return "\n".join(lines)


def _build_backtest_grades(date_str: str, graded_df: pd.DataFrame) -> str:
    """Build the Message 3 Backtest Grades summary."""
    if graded_df.empty:
        return f"📋 **MLB Backtest Grades ({date_str})** 📋\nNo graded predictions available."

    lines = [f"📋 **MLB Backtest Grades ({date_str})** 📋", ""]

    prop_markets = ["HR", "Hit", "TB", "Run", "RBI"]
    game_markets = ["NRFI", "ML_Away", "ML_Home", "Game_Total",
                    "F5_ML", "F5_Total", "TeamTotal_Away", "TeamTotal_Home"]

    prop_df = graded_df[graded_df["Market_Group"].isin(prop_markets)]
    game_df = graded_df[graded_df["Player"] == "GAME_TOTAL"]

    if not prop_df.empty:
        lines.append("⚾ **PLAYER PROPS**")
        for market in prop_markets:
            mdf = prop_df[prop_df["Market_Group"] == market]
            if mdf.empty:
                continue
            b = float(mdf["Brier"].mean())
            bench = _get_benchmark(market)
            wins_m = int(((mdf["Prob"] >= 0.5) == (mdf["Actual_Outcome"] == 1)).sum())
            acc = wins_m / len(mdf) * 100
            flag = "✅" if b <= bench else "❌"
            lines.append(f"  {flag} {market:<4}: Brier={b:.4f} (bench={bench}) | Acc={acc:.1f}% | n={len(mdf)}")

    if not game_df.empty:
        lines.append("")
        lines.append("🏟️ **GAME MARKETS**")
        for mg in game_markets:
            mdf = game_df[game_df["Market_Group"] == mg]
            if mdf.empty:
                continue
            b = float(mdf["Brier"].mean())
            bench = _get_benchmark(mg)
            flag = "✅" if b <= bench else "❌"
            lines.append(f"  {flag} {mg:<14}: Brier={b:.4f} (bench={bench}) | n={len(mdf)}")

    overall_brier = float(graded_df["Brier"].mean())
    lines += ["", f"Overall Brier: {overall_brier:.4f} across {len(graded_df)} predictions"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main grading function
# ---------------------------------------------------------------------------

def grade_ledger() -> None:
    if not os.path.exists("prediction_ledger.csv"):
        print("[!] No prediction_ledger.csv found. Run run_daily_predictions.py first.")
        return

    df = pd.read_csv("prediction_ledger.csv")

    eastern = pytz.timezone("US/Eastern")
    yesterday_str = (datetime.now(eastern) - timedelta(days=1)).strftime("%Y-%m-%d")

    yesterday_bets = df[df["Date"] == yesterday_str]
    if yesterday_bets.empty:
        print(f"[!] No predictions found in ledger for {yesterday_str}.")
        return

    # Fetch real outcomes
    actuals = fetch_yesterdays_boxscores(yesterday_str)
    game_outcomes = fetch_game_outcomes(yesterday_str)

    if not actuals and not game_outcomes:
        print("[!] Could not fetch any outcomes. No games played yesterday?")
        return

    # Grade all markets
    graded_df = grade_all_markets(yesterday_bets, actuals, game_outcomes)

    if graded_df.empty:
        print("[!] No predictions could be graded.")
        return

    print(f"[+] Graded {len(graded_df)} predictions for {yesterday_str}")

    # Calibration
    calibration = compute_calibration(graded_df)

    # Error patterns
    weak_slices = compute_error_patterns(graded_df)

    # Per-market-group Brier scores
    market_briers: dict = {}
    for mg, mdf in graded_df.groupby("Market_Group"):
        b = float(mdf["Brier"].mean())
        bench = _get_benchmark(mg)
        market_briers[mg] = {
            "brier": round(b, 4),
            "benchmark": bench,
            "status": "OK" if b <= bench else "UNDERPERFORMING",
            "sample_size": len(mdf),
        }

    overall_brier = float(graded_df["Brier"].mean())

    # Load previous brier for trend comparison
    prev_brier: float | None = None
    if os.path.exists("training_feedback.json"):
        try:
            with open("training_feedback.json") as f:
                prev_data = json.load(f)
            if prev_data.get("date") != yesterday_str:
                prev_brier = float(prev_data.get("overall_brier", 0)) or None
        except Exception:
            pass

    # Write training_feedback.json
    feedback = {
        "date": yesterday_str,
        "overall_brier": round(overall_brier, 4),
        "calibration_ece": calibration["ece"],
        "markets": market_briers,
        "weak_slices": weak_slices,
        "calibration_bins": calibration["bins"],
    }
    with open("training_feedback.json", "w") as f:
        json.dump(feedback, f, indent=2)
    print("[+] Wrote training_feedback.json")

    # Append graded rows to historical_training_data.csv
    training_cols = ["Date", "Player", "Market", "Prob", "Actual_Outcome",
                     "Batter_Archetype", "Pitcher_Archetype", "Platoon_Adv",
                     "Lineup_Spot", "Temp", "Stadium", "Away_Team", "Home_Team"]
    save_cols = [c for c in training_cols if c in graded_df.columns]
    new_training = graded_df[save_cols].copy()
    training_file = "historical_training_data.csv"
    if os.path.exists(training_file):
        existing = pd.read_csv(training_file)
        if yesterday_str in existing.get("Date", pd.Series(dtype=str)).values:
            print(f"[INFO] Training data for {yesterday_str} already exists. Skipping append.")
        else:
            updated = pd.concat([existing, new_training], ignore_index=True)
            updated.to_csv(training_file, index=False)
            print(f"[+] Appended {len(new_training)} graded rows to {training_file}")
    else:
        new_training.to_csv(training_file, index=False)
        print(f"[+] Created {training_file} with {len(new_training)} graded rows")

    # ---- Build and send Discord messages ----
    report_card = _build_report_card(yesterday_str, graded_df, calibration,
                                     weak_slices, prev_brier)
    backtest_grades = _build_backtest_grades(yesterday_str, graded_df)

    print("\n--- Report Card ---")
    print(report_card)
    print("\n--- Backtest Grades ---")
    print(backtest_grades)

    if DISCORD_WEBHOOK_URL:
        _send_discord_chunks(report_card, DISCORD_WEBHOOK_URL)
        _send_discord_chunks(backtest_grades, DISCORD_WEBHOOK_URL)
    else:
        print("[!] DISCORD_WEBHOOK not set — skipping Discord send.")


if __name__ == "__main__":
    grade_ledger()


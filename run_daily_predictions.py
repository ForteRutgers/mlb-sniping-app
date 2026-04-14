# run_daily_predictions.py
"""
Main script for generating comprehensive daily MLB predictions.
Saves results to enhanced_predictions_report.txt, mlb_predictions.db, and Discord.
"""

import os
import sys
import sqlite3
import subprocess
import requests
import json
from datetime import datetime
import pytz
import pandas as pd

from live_scraper import get_todays_matchups
from game_markets_predictor import GameMarketsPredictor, format_odds, get_edge_rating


# ---------------------------------------------------------------------------
# 1. DISCORD & HR ENGINE HELPERS
# ---------------------------------------------------------------------------
def send_to_discord(message_text):
    """Sends the prediction report to Discord via Webhook."""
    # !!! PASTE YOUR ACTUAL WEBHOOK URL HERE !!!
    webhook_url = "https://discord.com/api/webhooks/1489980544954400828/qxIgs-7qAOm2suqWQ3yXm9BG3JbKLvLZKDId7IMfNlFS4l27OhnEUQxCB140lVNZLgZd"

    if "YOUR_WEBHOOK_URL" in webhook_url or not webhook_url.startswith("https"):
        print("[!] Warning: Discord Webhook URL is not set. Skipping notification.")
        return

    # Discord 2000 character limit safety
    if len(message_text) > 2000:
        message_text = message_text[:1990] + "..."

    try:
        requests.post(webhook_url, data=json.dumps({"content": message_text}),
                      headers={"Content-Type": "application/json"})
    except Exception as e:
        print(f"[!] Error sending to Discord: {e}")


def _get_hr_discord_snippet():
    """Triggers the HR engine and reads the results."""
    snippet_path = 'hr_prop_engine/discord_snippet.txt'
    try:
        subprocess.run([sys.executable, "hr_prop_engine/predict_daily_hrs.py"], check=True)
    except Exception as e:
        return f"\n⚠️ HR Prop Engine Notice: Could not generate picks today. ({e})"

    if os.path.exists(snippet_path):
        with open(snippet_path, 'r') as f:
            return f.read()
    return "\n• No high-value HR props identified today."


# ---------------------------------------------------------------------------
# 2. ML & FEATURE ENGINEERING FALLBACKS
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
# 3. DATABASE LOGIC
# ---------------------------------------------------------------------------
def _write_to_sqlite(props_results, game_ledger_data):
    """Saves predictions into the SQLite database with duplicate protection and schema repair."""
    db_path = 'mlb_predictions.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- TABLE 1: GAME PREDICTIONS ---
    cursor.execute('''CREATE TABLE IF NOT EXISTS game_predictions
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY,
                          date
                          TEXT,
                          away_team
                          TEXT,
                          home_team
                          TEXT,
                          stadium
                          TEXT,
                          market
                          TEXT,
                          probability
                          REAL,
                          fair_odds
                          TEXT,
                          game_total_line
                          REAL
                      )''')

    cursor.execute("PRAGMA table_info(game_predictions)")
    game_cols = [i[1] for i in cursor.fetchall()]
    if 'date' not in game_cols:
        print("[!] Game Table Schema Outdated (missing 'date'). Recreating...")
        cursor.execute("DROP TABLE game_predictions")
        cursor.execute('''CREATE TABLE game_predictions
                          (
                              id              INTEGER PRIMARY KEY,
                              date            TEXT,
                              away_team       TEXT,
                              home_team       TEXT,
                              stadium         TEXT,
                              market          TEXT,
                              probability     REAL,
                              fair_odds       TEXT,
                              game_total_line REAL
                          )''')

    # --- TABLE 2: PLAYER PREDICTIONS ---
    cursor.execute('''CREATE TABLE IF NOT EXISTS player_predictions
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY,
                          date
                          TEXT,
                          player_name
                          TEXT,
                          team
                          TEXT,
                          opponent
                          TEXT,
                          market
                          TEXT,
                          probability
                          REAL,
                          lineup_spot
                          INTEGER
                      )''')

    cursor.execute("PRAGMA table_info(player_predictions)")
    player_cols = [i[1] for i in cursor.fetchall()]
    if 'date' not in player_cols:
        print("[!] Player Table Schema Outdated (missing 'date'). Recreating...")
        cursor.execute("DROP TABLE player_predictions")
        cursor.execute('''CREATE TABLE player_predictions
                          (
                              id          INTEGER PRIMARY KEY,
                              date        TEXT,
                              player_name TEXT,
                              team        TEXT,
                              opponent    TEXT,
                              market      TEXT,
                              probability REAL,
                              lineup_spot INTEGER
                          )''')

    eastern = pytz.timezone("US/Eastern")
    run_date = datetime.now(eastern).strftime('%Y-%m-%d')

    for entry in game_ledger_data:
        gr, nr = entry.get('game_result', {}), entry.get('nrfi_result', {})
        gt_line = gr.get('game_total_line', 8.5)

        cursor.execute("DELETE FROM game_predictions WHERE date=? AND away_team=? AND home_team=?",
                       (run_date, entry['away_team'], entry['home_team']))

        markets = [
            ('ML_Away', gr.get('away_ml_prob', 0), gr.get('away_ml_odds', 'N/A')),
            ('ML_Home', gr.get('home_ml_prob', 0), gr.get('home_ml_odds', 'N/A')),
            ('NRFI', nr.get('nrfi_prob', 0), nr.get('nrfi_odds', 'N/A')),
            ('Total_Over', gr.get('game_total_over', 0), "N/A"),
            ('Total_Under', gr.get('game_total_under', 0), "N/A"),
            ('Expected_Total', gr.get('game_total_mean', 0), "N/A")
        ]

        for m_name, prob, odds in markets:
            cursor.execute('''INSERT INTO game_predictions
                              (date, away_team, home_team, stadium, market, probability, fair_odds, game_total_line)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                           (run_date, entry['away_team'], entry['home_team'], entry['stadium'], m_name, float(prob),
                            str(odds), float(gt_line)))

    for prop in props_results:
        cursor.execute("DELETE FROM player_predictions WHERE date=? AND player_name=?", (run_date, prop['name']))
        for m_name, prob in prop.get('props', {}).items():
            cursor.execute('''INSERT INTO player_predictions
                                  (date, player_name, team, opponent, market, probability, lineup_spot)
                              VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (run_date, prop['name'], prop['team'], prop['pitcher'], m_name, float(prob), prop['order']))

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Database updated for {run_date}")


# ---------------------------------------------------------------------------
# 4. FORMATTING HELPERS
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
    away_ml = game_r.get('away_ml_prob', 0.5) * 100
    home_ml = game_r.get('home_ml_prob', 0.5) * 100
    over_p = game_r.get('game_total_over', 0.5) * 100
    under_p = game_r.get('game_total_under', 0.5) * 100
    total_line = game_r.get('game_total_line', 8.5)
    nrfi_p = nrfi_r.get('nrfi_prob', 0.5) * 100
    yrfi_p = (1 - (nrfi_p / 100)) * 100

    report = (
        f"**{away_t} @ {home_t}** ({game_time})\n"
        f"```yaml\n"
        f"ML:    {away_t} {away_ml:.0f}% vs {home_t} {home_ml:.0f}%\n"
        f"O/U {total_line}: Over {over_p:.0f}%  vs Under {under_p:.0f}%\n"
        f"1st:   NRFI {nrfi_p:.0f}%  vs YRFI {yrfi_p:.0f}%\n"
        f"Edge:  {_recommendation(nrfi_p / 100)}\n"
        f"```\n"
    )
    return report


def _format_props_section(props_list: list) -> str:
    if not props_list: return ""
    lines = ["**🔥 TOP PLAYER PROPS 🔥**"]
    for prop in props_list[:5]:  # Display top 5 to save space
        lines.append(
            f"• **{prop.get('name', 'Unknown')}**: {list(prop.get('props', {}).keys())[0]} ({list(prop.get('props', {}).values())[0] * 100:.1f}%)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 5. MAIN EXECUTION LOOP
# ---------------------------------------------------------------------------
def run_daily_predictions():
    eastern = pytz.timezone("US/Eastern")
    today_str = datetime.now(eastern).strftime("%Y-%m-%d")

    print("=" * 70)
    print(f" MLB ENHANCED PREDICTIONS - {today_str} ")
    print("=" * 70)

    gmp = GameMarketsPredictor()
    print("\n[1/3] Fetching today's matchups...")
    matchups = get_todays_matchups()

    report_parts = [
        f"========================================\nMLB ENHANCED PREDICTIONS - {today_str}\n========================================"]
    game_ledger = []
    all_props = []

    if not matchups:
        report_parts.append("\n[!] No games scheduled or found for today.")
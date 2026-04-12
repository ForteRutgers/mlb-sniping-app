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
# Database Logic (With FULL Schema Fix for both tables)
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

    # Schema Check for Game Table
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

    # Schema Check for Player Table
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

    # Use current Eastern Time for consistent daily tracking
    eastern = pytz.timezone("US/Eastern")
    run_date = datetime.now(eastern).strftime('%Y-%m-%d')

    # --- INSERT GAME DATA ---
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

    # --- INSERT PLAYER DATA ---
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
    # Condensed "Both Sides" formatting for Discord
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

    lines = []
    for prop in props_list:
        lines.append(f"{prop.get('name', 'Unknown')} - {prop.get('market', 'Prop')}")

    return "\n".join(lines)
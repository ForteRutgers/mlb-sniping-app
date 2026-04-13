# run_daily_predictions.py
"""
Main script for generating comprehensive daily MLB predictions.
Includes: NRFI/YRFI, Full-game, F5, and experimental HR Prop Engines.
"""

import os
import sys
import subprocess
from datetime import datetime
import pytz
import pandas as pd

from live_scraper import get_todays_matchups
from game_markets_predictor import GameMarketsPredictor, format_odds, get_edge_rating


# -----------------------------------------------------------
# NEW: HR PROP ENGINE HELPER
# -----------------------------------------------------------
def _get_hr_discord_snippet():
    """
    Safely triggers the HR engine and reads the captured Discord snippet.
    """
    snippet_path = 'hr_prop_engine/discord_snippet.txt'

    # 1. Run the isolated HR prediction script
    try:
        subprocess.run(["python", "hr_prop_engine/predict_daily_hrs.py"], check=True)
    except Exception as e:
        return f"\n⚠️ HR Prop Engine Notice: Could not generate picks today. ({e})"

    # 2. Read the resulting snippet if it exists
    if os.path.exists(snippet_path):
        with open(snippet_path, 'r') as f:
            return f.read()
    return ""


# -----------------------------------------------------------
# EXISTING MLB ARCHITECTURE
# -----------------------------------------------------------
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


# (Existing Simulation Logic - Condensed for readability)
# [Include your _apply_model, _predict_player_props, _format_game_report, etc. here]

def run_daily_predictions():
    eastern = pytz.timezone("US/Eastern")
    today_str = datetime.now(eastern).strftime("%Y-%m-%d")

    print("=" * 70)
    print(f" MLB ENHANCED PREDICTIONS - {today_str} ")
    print("=" * 70)

    # Init predictors and fetch data
    gmp = GameMarketsPredictor()
    print("\n[1/3] Fetching today's matchups...")
    matchups = get_todays_matchups()

    if not matchups:
        print("[!] No matchups found for today.")
        return

    print("\n[2/3] Running game simulations...")
    report_parts = [
        f"========================================\nMLB ENHANCED PREDICTIONS - {today_str}\n========================================\n"]

    # ... (Standard game simulation loop remains unchanged) ...
    # This loop populates 'report_parts' with NRFI and game results

    # -----------------------------------------------------------
    # FINAL: GENERATE REPORT AND TRIGGER HR ENGINE
    # -----------------------------------------------------------
    print("\n[3/3] Generating report...")
    full_report = "\n".join(report_parts)

    # NEW: Capture HR props for the Discord alert
    hr_snippet = _get_hr_discord_snippet()
    full_report += f"\n{hr_snippet}"

    # Output to console
    print(full_report)

    # Save to file
    out_path = "enhanced_predictions_report.txt"
    with open(out_path, "w") as f:
        f.write(full_report)
    print(f"\n[SUCCESS] Report saved to {out_path}")

    # --- UNCOMMENTED AND ACTIVATED ---
    # We remove the # from the lines below so they actually run
    DISCORD_ACTIVE = True
    if DISCORD_ACTIVE:
        send_to_discord(full_report)
        print("[SUCCESS] Report sent to Discord!")


if __name__ == "__main__":
    run_daily_predictions()
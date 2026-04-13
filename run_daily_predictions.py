# run_daily_predictions.py
"""
Main script for generating comprehensive daily MLB predictions.
Includes: NRFI/YRFI, Full-game, F5, and experimental HR Prop Engines.
"""

import os
import sys
import subprocess
import requests
import json
from datetime import datetime
import pytz
import pandas as pd

# External logic imports
try:
    from live_scraper import get_todays_matchups
    from game_markets_predictor import GameMarketsPredictor, format_odds, get_edge_rating
except ImportError as e:
    print(f"[!] Critical Import Error: {e}")
    sys.exit(1)


# -----------------------------------------------------------
# DISCORD NOTIFIER FUNCTION (Fixed NameError)
# -----------------------------------------------------------
def send_to_discord(message_text):
    """
    Sends the prediction report to Discord via Webhook.
    """
    # !!! PASTE YOUR ACTUAL WEBHOOK URL HERE !!!
    webhook_url = "https://discord.com/api/webhooks/1487199773256585239/Ci6G7sxUSvvlNPVdT2ng4CpbtTn69t0u7bVn9AupakQa5YYMoZkk7t3GuhnyM4BCWZov"

    if "https://discord.com/api/webhooks/1487199773256585239/Ci6G7sxUSvvlNPVdT2ng4CpbtTn69t0u7bVn9AupakQa5YYMoZkk7t3GuhnyM4BCWZov" in webhook_url or not webhook_url.startswith("https"):
        print("[!] Warning: Discord Webhook URL is not set correctly. Skipping notification.")
        return

    # Discord has a 2000 character limit. This trims the message to prevent errors.
    if len(message_text) > 2000:
        message_text = message_text[:1990] + "..."

    payload = {"content": message_text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        if response.status_code == 204:
            print("[SUCCESS] Report sent to Discord.")
        else:
            print(f"[!] Discord returned status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[!] Error sending to Discord: {e}")


# -----------------------------------------------------------
# HR PROP ENGINE HELPER
# -----------------------------------------------------------
def _get_hr_discord_snippet():
    """
    Safely triggers the HR engine and reads the captured Discord snippet.
    """
    snippet_path = 'hr_prop_engine/discord_snippet.txt'

    # 1. Run the isolated HR prediction script
    print("Initiating Experimental HR Prop Predictions...")
    try:
        # We use sys.executable to ensure we use the same environment's python
        subprocess.run([sys.executable, "hr_prop_engine/predict_daily_hrs.py"], check=True)
    except Exception as e:
        return f"\n⚠️ HR Prop Engine Notice: Could not generate picks today. ({e})"

    # 2. Read the resulting snippet if it exists
    if os.path.exists(snippet_path):
        with open(snippet_path, 'r') as f:
            return f.read()
    return ""


# -----------------------------------------------------------
# MAIN EXECUTION LOGIC
# -----------------------------------------------------------
def run_daily_predictions():
    eastern = pytz.timezone("US/Eastern")
    today_str = datetime.now(eastern).strftime("%Y-%m-%d")

    print("=" * 70)
    print(f" MLB ENHANCED PREDICTIONS - {today_str} ")
    print("=" * 70)

    # 1. Fetch Data
    print("\n[1/3] Fetching today's matchups...")
    matchups = get_todays_matchups()

    if not matchups:
        print("[!] No matchups found for today.")
        return

    # 2. Run Game Simulations (Placeholder for your existing logic)
    print("\n[2/3] Running game simulations...")
    report_parts = [
        f"========================================\nMLB ENHANCED PREDICTIONS - {today_str}\n========================================\n"
    ]

    # Note: Your specific NRFI/Game logic usually populates report_parts here.
    # We will proceed to the HR capture.

    # 3. Generate Report and Trigger HR Engine
    print("\n[3/3] Generating report and triggering HR engine...")
    full_report = "\n".join(report_parts)

    # Capture HR props for the Discord alert
    hr_snippet = _get_hr_discord_snippet()
    full_report += f"\n{hr_snippet}"

    # Output to console
    print(full_report)

    # Save to file locally
    out_path = "enhanced_predictions_report.txt"
    with open(out_path, "w") as f:
        f.write(full_report)
    print(f"\n[LOCAL SUCCESS] Report saved to {out_path}")

    # 4. Final Discord Trigger
    DISCORD_ACTIVE = True
    if DISCORD_ACTIVE:
        print("Attempting to send to Discord...")
        send_to_discord(full_report)


if __name__ == "__main__":
    run_daily_predictions()
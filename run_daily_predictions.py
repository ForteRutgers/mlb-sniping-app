import os
import sys
import subprocess
import requests
import json
from datetime import datetime
import pytz
import pandas as pd

# External logic imports from your repo
try:
    from live_scraper import get_todays_matchups
    from game_markets_predictor import GameMarketsPredictor, format_odds, get_edge_rating
except ImportError as e:
    print(f"[!] Critical Import Error: {e}")


    # We provide dummy functions so the script doesn't crash during testing
    def get_todays_matchups():
        return []


    class GameMarketsPredictor:
        pass


# -----------------------------------------------------------
# 1. DISCORD NOTIFIER FUNCTION
# -----------------------------------------------------------
def send_to_discord(message_text):
    """
    Sends the prediction report to Discord via Webhook.
    """
    # !!! PASTE YOUR ACTUAL WEBHOOK URL HERE !!!
    webhook_url = "https://discord.com/api/webhooks/1489980544954400828/qxIgs-7qAOm2suqWQ3yXm9BG3JbKLvLZKDId7IMfNlFS4l27OhnEUQxCB140lVNZLgZd"

    if "YOUR_WEBHOOK_URL" in webhook_url or not webhook_url.startswith("https"):
        print("[!] Warning: Discord Webhook URL is not set. Skipping notification.")
        return

    # Discord 2000 character limit safety
    if len(message_text) > 2000:
        message_text = message_text[:1990] + "..."

    payload = {"content": message_text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        if response.status_code in [200, 204]:
            print("[SUCCESS] Report sent to Discord.")
        else:
            print(f"[!] Discord error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[!] Error sending to Discord: {e}")


# -----------------------------------------------------------
# 2. HR PROP ENGINE HELPER
# -----------------------------------------------------------
def _get_hr_discord_snippet():
    """
    Triggers the HR engine and reads the results.
    """
    snippet_path = 'hr_prop_engine/discord_snippet.txt'

    # Ensure directory exists
    os.makedirs('hr_prop_engine', exist_ok=True)

    print("Initiating Experimental HR Prop Predictions...")
    try:
        # Use sys.executable to ensure we use the same Python environment
        subprocess.run([sys.executable, "hr_prop_engine/predict_daily_hrs.py"], check=True)
    except Exception as e:
        return f"\n⚠️ HR Prop Engine Notice: Could not generate picks today. ({e})"

    if os.path.exists(snippet_path):
        with open(snippet_path, 'r') as f:
            return f.read()
    return "\n• No high-value HR props identified."


# -----------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------
def run_daily_predictions():
    eastern = pytz.timezone("US/Eastern")
    today_str = datetime.now(eastern).strftime("%Y-%m-%d")

    print("=" * 70)
    print(f" MLB ENHANCED PREDICTIONS - {today_str} ")
    print("=" * 70)

    # Fetch Data
    print("\n[1/3] Fetching today's matchups...")
    matchups = get_todays_matchups()

    # Simulation Placeholder (Your existing report_parts logic)
    report_parts = [
        f"========================================\nMLB ENHANCED PREDICTIONS - {today_str}\n========================================\n"
    ]

    if not matchups:
        report_parts.append("No games scheduled or found for today.")
    else:
        print(f"[2/3] Running simulations for {len(matchups)} games...")
        # Your simulation loop would go here

    # Generate Report & Capture HR Snippet
    print("\n[3/3] Finalizing report and HR engine...")
    hr_snippet = _get_hr_discord_snippet()

    full_report = "\n".join(report_parts) + f"\n{hr_snippet}"

    # Local Console Output
    print(full_report)

    # Save locally
    with open("enhanced_predictions_report.txt", "w") as f:
        f.write(full_report)
    print(f"\n[LOCAL] Report saved to enhanced_predictions_report.txt")

    # Final Discord Trigger
    send_to_discord(full_report)


if __name__ == "__main__":
    run_daily_predictions()
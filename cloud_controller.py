# cloud_controller.py
import requests
from datetime import datetime
import pytz
import subprocess
import sys
import os
import time

# =====================================================================
# 🚨 PASTE YOUR DISCORD WEBHOOK URL BELOW (Inside the quotes) 🚨
# =====================================================================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1487199773256585239/Ci6G7sxUSvvlNPVdT2ng4CpbtTn69t0u7bVn9AupakQa5YYMoZkk7t3GuhnyM4BCWZov"


def send_discord_alert():
    """Sends a success message and the betting dashboard to your Discord."""
    if DISCORD_WEBHOOK_URL == "https://discord.com/api/webhooks/1487199773256585239/Ci6G7sxUSvvlNPVdT2ng4CpbtTn69t0u7bVn9AupakQa5YYMoZkk7t3GuhnyM4BCWZov":
        print("    [!] Discord Webhook URL not set. Skipping text alert.")
        return

    print("    [+] Sending dashboard to Discord...")

    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern).strftime('%I:%M %p')

    message = {
        "content": f"🚨 **MLB Sniping Engine Complete! ({current_time} EST)** 🚨\nOfficial 9-man lineups locked and simulated. Here is your updated dashboard:"
    }

    file_path = "betting_dashboard_report.txt"

    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                response = requests.post(
                    DISCORD_WEBHOOK_URL,
                    data=message,
                    files={"file": (file_path, f, "text/plain")}
                )
            if response.status_code in [200, 204]:
                print("    [SUCCESS] Discord alert delivered to your phone!")
            else:
                print(f"    [!] Failed to send Discord alert: {response.status_code}")
        else:
            print("    [!] Dashboard file not found. Sending text-only alert.")
            requests.post(DISCORD_WEBHOOK_URL, json=message)
    except Exception as e:
        print(f"    [!] Error sending to Discord: {e}")


def check_lineup_confirmed(game_pk):
    """Pings the live MLB boxscore to see if the manager has submitted the official batting order."""
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        data = requests.get(url).json()
        boxscore = data.get('liveData', {}).get('boxscore', {}).get('teams', {})
        away_lineup = boxscore.get('away', {}).get('battingOrder', [])
        home_lineup = boxscore.get('home', {}).get('battingOrder', [])

        if len(away_lineup) >= 9 and len(home_lineup) >= 9:
            return True
        return False
    except Exception as e:
        print(f"    [!] Error checking lineup for game {game_pk}: {e}")
        return False


# =====================================================================
# SELF-IMPROVEMENT PIPELINE — Runs before predictions each day
# =====================================================================

_PIPELINE_STEP_TIMEOUT = 600  # 10 minutes per learning step
_PREDICTION_LEDGER = "prediction_ledger.csv"
_HISTORICAL_DATA = "historical_training_data.csv"


def _run_step(script_name, description):
    """
    Safely runs a single step of the self-improvement pipeline.
    Returns True if the step succeeded, False if it failed.
    Failures are logged but NEVER stop the rest of the pipeline.
    """
    print(f"\n    [LEARNING] {description}...")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=_PIPELINE_STEP_TIMEOUT
        )
        if result.returncode == 0:
            print(f"    [OK] {description} — completed successfully.")
            return True
        else:
            print(f"    [WARN] {description} — exited with code {result.returncode}.")
            if result.stderr:
                # Print only the last 3 lines of the error to keep logs clean
                error_lines = result.stderr.strip().split('\n')[-3:]
                for line in error_lines:
                    print(f"           {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    [WARN] {description} — timed out after {_PIPELINE_STEP_TIMEOUT // 60} minutes. Skipping.")
        return False
    except Exception as e:
        print(f"    [WARN] {description} — unexpected error: {e}. Skipping.")
        return False


def run_self_improvement_pipeline():
    """
    Executes the full self-improvement loop BEFORE daily predictions.

    Order of operations:
      1. Grade yesterday's predictions against real MLB outcomes
      2. Retrain the AI Corrector on all accumulated graded history

    Every step is wrapped in try/except so that a failure in any single
    step (e.g., no games yesterday, API down) will NOT prevent the
    daily predictions from running.
    """
    print("\n" + "=" * 70)
    print("  🧠 SELF-IMPROVEMENT PIPELINE — Learning from yesterday's results")
    print("=" * 70)

    steps_attempted = 0
    steps_succeeded = 0

    # --- Step 1: Grade yesterday's predictions ---
    # This compares what we predicted to what actually happened,
    # writes training_feedback.json, and appends to historical_training_data.csv
    if os.path.exists("results_tracker.py") and os.path.exists(_PREDICTION_LEDGER):
        steps_attempted += 1
        if _run_step("results_tracker.py", "Step 1/2: Grading yesterday's predictions"):
            steps_succeeded += 1
    else:
        print("    [SKIP] results_tracker.py or prediction_ledger.csv not found — skipping grading step.")

    # --- Step 2: Retrain the AI Corrector on accumulated graded history ---
    # This reads historical_training_data.csv (which grows every day after grading)
    # and retrains the XGBoost corrector that adjusts raw Monte Carlo probabilities
    # It will regenerate mlb_xgboost_brain.json from the graded data
    if os.path.exists("ai_corrector.py") and os.path.exists(_HISTORICAL_DATA):
        steps_attempted += 1
        if _run_step("ai_corrector.py", "Step 2/2: Retraining AI Corrector on graded history"):
            steps_succeeded += 1
    else:
        print("    [SKIP] ai_corrector.py or historical_training_data.csv not found — skipping corrector retrain.")

    # --- Summary ---
    print("\n" + "-" * 70)
    print(f"  🧠 SELF-IMPROVEMENT COMPLETE: {steps_succeeded}/{steps_attempted} steps succeeded.")
    if steps_attempted == 0:
        print("     (No learning scripts or data found. Models will run with current weights.)")
    print("-" * 70 + "\n")


# =====================================================================
# MAIN ENGINE — Now calls the learning pipeline before predicting
# =====================================================================

def run_engine():
    print(f"\n[TRIGGER] Official 9-Man Lineups Confirmed! Launching MLB Sniping Engine...")

    # === NEW: Learn from yesterday BEFORE making today's predictions ===
    run_self_improvement_pipeline()

    # === Generate today's predictions with the (now smarter) models ===
    print("[ENGINE] Running daily predictions with updated models...")
    subprocess.run([sys.executable, "daily_bets.py", "--auto"])
    print(f"\n[SUCCESS] Engine finished. Dashboard generated with locked lineups.")

    # Trigger the Discord notification right after the math finishes
    send_discord_alert()


def check_schedule():
    now_utc = datetime.now(pytz.utc)
    today = now_utc.strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"

    print(f"=== MLB CLOUD CONTROLLER (Verification Mode) ===")
    print(f"Current Server Time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        data = requests.get(url).json()
        games = data['dates'][0]['games']
    except (KeyError, IndexError):
        print("No games scheduled for today or API unavailable.")
        return

    trigger_run = False
    for game in games:
        if game['status']['statusCode'] in ['F', 'O', 'C', 'I']:
            continue

        game_pk = game['gamePk']
        game_time_str = game['gameDate'].replace("Z", "+00:00")
        game_time_utc = datetime.fromisoformat(game_time_str)

        time_diff = (game_time_utc - now_utc).total_seconds() / 60.0

        if 0 <= time_diff <= 45:
            matchup = f"{game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']}"
            print(f" -> MATCH IN WINDOW: {matchup} (Starts in {int(time_diff)} mins)")

            if check_lineup_confirmed(game_pk):
                print(f"    [+] Official lineups detected and locked!")
                trigger_run = True
            else:
                print(f"    [-] Lineups not yet submitted by managers. Waiting for next cycle...")

    if trigger_run:
        run_engine()
    else:
        print(" -> No actionable games with confirmed lineups right now. Sleeping safely...")


if __name__ == "__main__":
    print("[CLOUD CONTROLLER] Starting polling loop...")
    try:
        while True:
            check_schedule()
            print("[CLOUD CONTROLLER] Sleeping for 5 minutes...")
            time.sleep(300)  # Check every 5 minutes
    except KeyboardInterrupt:
        print("[CLOUD CONTROLLER] Shutdown signal received. Exiting.")
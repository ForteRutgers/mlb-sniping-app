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


def run_engine():
    print(f"\n[TRIGGER] Official 9-Man Lineups Confirmed! Launching MLB Monte Carlo Engine...")
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
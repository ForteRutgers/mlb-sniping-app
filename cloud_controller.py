# cloud_controller.py
import requests
from datetime import datetime
import pytz
import subprocess
import sys

def run_engine():
    print("\n[TRIGGER] 30-Minute Window Hit! Launching MLB Monte Carlo Engine...")
    subprocess.run([sys.executable, "daily_bets.py", "--auto"])
    print("\n[SUCCESS] Engine finished. Dashboard generated.")

def check_schedule():
    now_utc = datetime.now(pytz.utc)
    today = now_utc.strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"

    print(f"=== MLB CLOUD CONTROLLER ===")
    print(f"Current Server Time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        data = requests.get(url).json()
        games = data['dates'][0]['games']
    except (KeyError, IndexError):
        print("No games scheduled for today or API unavailable.")
        return

    run_needed = False
    for game in games:
        # Skip games that are already Final (F), Over (O), Postponed (P), or Cancelled (C)
        if game['status']['statusCode'] in ['F', 'O', 'P', 'C']:
            continue

        game_time_str = game['gameDate'].replace("Z", "+00:00")
        game_time_utc = datetime.fromisoformat(game_time_str)

        # Calculate minutes until first pitch
        time_diff = (game_time_utc - now_utc).total_seconds() / 60.0

        # If a game starts in exactly 20 to 40 minutes, trigger the run!
        # (We use a 20-min window because the GitHub Action runs every 15 minutes)
        if 20 <= time_diff <= 40:
            matchup = f"{game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']}"
            print(f" -> MATCH DETECTED: {matchup} (Starts in {int(time_diff)} mins)")
            run_needed = True

    if run_needed:
        run_engine()
    else:
        print(" -> No games starting within the next 30 minutes. Sleeping safely...")

if __name__ == "__main__":
    check_schedule()
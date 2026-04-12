# backtest_totals.py
import sqlite3
import requests
from datetime import datetime


def fetch_actual_scores(date_str):
    """Fetches official final scores from the MLB API for a given date."""
    print(f" -> Fetching official MLB scores for {date_str}...")
    # The MLB API uses YYYY-MM-DD format
    api_date = date_str.split(" ")[0]
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={api_date}"

    response = requests.get(url)
    data = response.json()

    results = {}
    if not data.get('dates'):
        return results

    for game in data['dates'][0].get('games', []):
        # Only process completed games
        if game['status']['statusCode'] == 'F':
            away = game['teams']['away']['team']['name']
            home = game['teams']['home']['team']['name']
            total_runs = game['teams']['away']['score'] + game['teams']['home']['score']

            # Create a simple matching key
            match_key = f"{away} @ {home}".lower()
            results[match_key] = total_runs

    return results


def run_totals_backtest():
    print("========================================")
    print("   MLB TOTALS (O/U) BACKTEST ENGINE     ")
    print("========================================")

    conn = sqlite3.connect('mlb_predictions.db')
    cursor = conn.cursor()

    # Get all the "Total" predictions from the database
    cursor.execute('''
                   SELECT prediction_date, away_team, home_team, market, probability
                   FROM game_predictions
                   WHERE market LIKE 'Total_%'
                   ''')
    predictions = cursor.fetchall()

    if not predictions:
        print("[!] No 'Total' predictions found in the database yet.")
        print("    Make sure you ran run_daily_predictions.py with the updated code.")
        return

    # Group predictions by date to avoid spamming the MLB API
    dates_to_check = set([row[0].split(" ")[0] for row in predictions])
    actual_scores = {}
    for d in dates_to_check:
        actual_scores.update(fetch_actual_scores(d))

    correct_picks = 0
    total_graded = 0

    print("\n--- GRADING RESULTS ---")
    for row in predictions:
        pred_date, away, home, market, prob = row

        # Only grade predictions where the AI had a lean (>50% probability)
        if prob < 0.50:
            continue

        match_key = f"{away} @ {home}".lower()

        # We need to do a flexible match because API team names might slightly differ from your scraper
        actual_runs = None
        for api_key, runs in actual_scores.items():
            if away.lower() in api_key and home.lower() in api_key:
                actual_runs = runs
                break

        if actual_runs is None:
            continue  # Game might not be finished yet or rained out

        # Parse the line from the market (e.g., 'Total_Over_8.5')
        parts = market.split('_')
        pick_type = parts[1]  # 'Over' or 'Under'
        line = float(parts[2])

        total_graded += 1

        # Did the prediction win?
        is_win = False
        if pick_type == 'Over' and actual_runs > line:
            is_win = True
        elif pick_type == 'Under' and actual_runs < line:
            is_win = True

        if is_win:
            correct_picks += 1
            print(
                f"✅ WIN : {away} @ {home} | Pick: {pick_type} {line} | Actual Runs: {actual_runs} (AI Prob: {prob * 100:.1f}%)")
        else:
            print(
                f"❌ LOSS: {away} @ {home} | Pick: {pick_type} {line} | Actual Runs: {actual_runs} (AI Prob: {prob * 100:.1f}%)")

    if total_graded > 0:
        win_rate = (correct_picks / total_graded) * 100
        print("\n========================================")
        print(f" BACKTEST SUMMARY")
        print(f" Total Games Graded: {total_graded}")
        print(f" Correct Predictions: {correct_picks}")
        print(f" Win Rate: {win_rate:.1f}%")
        print("========================================")
    else:
        print("\n[!] Could not grade any games. They might still be playing, or haven't started yet!")


if __name__ == "__main__":
    run_totals_backtest()
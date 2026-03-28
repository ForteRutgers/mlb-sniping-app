# results_tracker.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1487199773256585239/Ci6G7sxUSvvlNPVdT2ng4CpbtTn69t0u7bVn9AupakQa5YYMoZkk7t3GuhnyM4BCWZov"


def fetch_yesterdays_boxscores(yesterday_str):
    """Pings the MLB API for yesterday's locked-in box scores."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={yesterday_str}&hydrate=boxscore"
    print(f"Fetching official MLB Box Scores for {yesterday_str}...")

    try:
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"[!] API Error: {e}")
        return {}

    actuals = {}
    if 'dates' not in data or not data['dates']:
        return actuals

    for game in data['dates'][0].get('games', []):
        # We only want to grade completed games
        if game['status']['statusCode'] not in ['F', 'O']:
            continue

        boxscore = game.get('boxscore', {}).get('teams', {})
        for team_side in ['away', 'home']:
            players = boxscore.get(team_side, {}).get('players', {})
            for pid, pdata in players.items():
                name = pdata['person']['fullName']
                # Clean name to match the ledger
                clean_name = name.replace('*', '').strip()

                stats = pdata.get('stats', {}).get('batting', {})
                if stats:
                    actuals[clean_name] = {
                        'HR': stats.get('homeRuns', 0),
                        'Hit': stats.get('hits', 0),
                        'TB': stats.get('totalBases', 0),
                        'Run': stats.get('runs', 0),
                        'RBI': stats.get('rbi', 0)
                    }
    return actuals


def grade_ledger():
    if not os.path.exists("prediction_ledger.csv"):
        print("[!] No prediction_ledger.csv found. Run daily_bets.py first to build history.")
        return

    df = pd.read_csv("prediction_ledger.csv")

    eastern = pytz.timezone('US/Eastern')
    yesterday_str = (datetime.now(eastern) - timedelta(days=1)).strftime('%Y-%m-%d')

    # Filter the ledger for only yesterday's bets
    yesterday_bets = df[df['Date'] == yesterday_str]
    if yesterday_bets.empty:
        print(f"[!] No predictions found in ledger for {yesterday_str}.")
        return

    actuals = fetch_yesterdays_boxscores(yesterday_str)
    if not actuals:
        print(f"[!] Could not fetch yesterday's box scores. Maybe no games were played?")
        return

    brier_scores = []
    wins = 0
    losses = 0

    for index, row in yesterday_bets.iterrows():
        player = str(row['Player']).replace('*', '').strip()
        market = str(row['Market'])
        prob = float(row['Prob'])

        # If the player wasn't in the box score (late scratch), ignore the bet
        if player not in actuals:
            continue

        p_stats = actuals[player]

        # Determine actual binary outcome (1 = Hit the prop, 0 = Missed the prop)
        actual_outcome = 0
        if market == 'HR' and p_stats.get('HR', 0) >= 1:
            actual_outcome = 1
        elif market == 'Hit' and p_stats.get('Hit', 0) >= 1:
            actual_outcome = 1
        elif market == 'TB' and p_stats.get('TB', 0) >= 2:
            actual_outcome = 1
        elif market == 'Run' and p_stats.get('Run', 0) >= 1:
            actual_outcome = 1
        elif market == 'RBI' and p_stats.get('RBI', 0) >= 1:
            actual_outcome = 1

        # Track raw Win/Loss for fun (Assuming we theoretically bet everything > 50%)
        if actual_outcome == 1 and prob >= 0.50:
            wins += 1
        elif actual_outcome == 0 and prob < 0.50:
            wins += 1
        else:
            losses += 1

        # The ultimate math: Calculate the Brier Score for this prop
        brier = (prob - actual_outcome) ** 2
        brier_scores.append(brier)

    if not brier_scores:
        print("[!] Found bets, but none of the players logged an at-bat yesterday.")
        return

    avg_brier = sum(brier_scores) / len(brier_scores)
    accuracy = (wins / (wins + losses)) * 100

    report = f"📊 **MLB Model Backtest Report ({yesterday_str})** 📊\n"
    report += f"**Brier Score:** {avg_brier:.4f} *(Lower is better, <0.20 is elite)*\n"
    report += f"**Binary Accuracy:** {accuracy:.1f}%\n"
    report += f"**Total Props Graded:** {len(brier_scores)}"

    print(report)

    if DISCORD_WEBHOOK_URL != "https://discord.com/api/webhooks/1487199773256585239/Ci6G7sxUSvvlNPVdT2ng4CpbtTn69t0u7bVn9AupakQa5YYMoZkk7t3GuhnyM4BCWZov":
        requests.post(DISCORD_WEBHOOK_URL, json={"content": report})
        print("Report sent to Discord!")


if __name__ == "__main__":
    grade_ledger()
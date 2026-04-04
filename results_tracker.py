# results_tracker.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os

# 🚨 Paste your webhook URL below! 🚨
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1487199773256585239/Ci6G7sxUSvvlNPVdT2ng4CpbtTn69t0u7bVn9AupakQa5YYMoZkk7t3GuhnyM4BCWZov"


def _fetch_linescore(game_pk: int) -> dict:
    """Fetch the linescore (inning-by-inning runs) for a single game."""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
    try:
        return requests.get(url, timeout=5).json()
    except Exception:
        return {}


def fetch_game_outcomes(yesterday_str: str) -> dict:
    """
    Returns a dict of game-level results keyed by game_pk.
    Each entry has: away_runs, home_runs, away_runs_1st, home_runs_1st,
    away_team, home_team.
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={yesterday_str}"
    print(f"Fetching official MLB game outcomes for {yesterday_str}...")
    try:
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"[!] API Error: {e}")
        return {}

    outcomes = {}
    if 'dates' not in data or not data['dates']:
        return outcomes

    for game in data['dates'][0].get('games', []):
        if game['status']['statusCode'] not in ['F', 'O']:
            continue
        game_pk = game['gamePk']
        away_team = game['teams']['away']['team']['name']
        home_team = game['teams']['home']['team']['name']
        linescore = _fetch_linescore(game_pk)
        innings = linescore.get('innings', [])
        away_total = linescore.get('teams', {}).get('away', {}).get('runs', 0) or 0
        home_total = linescore.get('teams', {}).get('home', {}).get('runs', 0) or 0
        away_1st = home_1st = 0
        if innings:
            away_1st = innings[0].get('away', {}).get('runs', 0) or 0
            home_1st = innings[0].get('home', {}).get('runs', 0) or 0
        outcomes[game_pk] = {
            'away_team': away_team,
            'home_team': home_team,
            'away_runs': int(away_total),
            'home_runs': int(home_total),
            'away_runs_1st': int(away_1st),
            'home_runs_1st': int(home_1st),
        }
    return outcomes


def grade_game_markets(yesterday_bets: pd.DataFrame, game_outcomes: dict) -> dict:
    """
    Grade NRFI/YRFI, Moneyline, and Game Total predictions.
    Returns a summary dict with brier scores and win/loss counts per market type.
    """
    nrfi_brier = []
    ml_brier = []
    gt_brier = []

    for _, row in yesterday_bets.iterrows():
        player = str(row.get('Player', '')).strip()
        market = str(row.get('Market', ''))
        prob = float(row.get('Prob', 0.5))

        if player != 'GAME_TOTAL':
            continue

        # Match game by team names recorded in ledger
        away_t = str(row.get('Away_Team', ''))
        home_t = str(row.get('Home_Team', ''))
        matched = None
        for gk, go in game_outcomes.items():
            if (go['away_team'] == away_t and go['home_team'] == home_t) or \
               (go['away_team'] in away_t or go['home_team'] in home_t):
                matched = go
                break

        if matched is None:
            continue

        if market == 'NRFI':
            actual = int(matched['away_runs_1st'] == 0 and matched['home_runs_1st'] == 0)
            nrfi_brier.append((prob - actual) ** 2)
        elif market.startswith('ML_Away'):
            actual = int(matched['away_runs'] > matched['home_runs'])
            ml_brier.append((prob - actual) ** 2)
        elif market.startswith('ML_Home'):
            actual = int(matched['home_runs'] > matched['away_runs'])
            ml_brier.append((prob - actual) ** 2)
        elif market.startswith('Over_'):
            try:
                line = float(market.split('_')[1])
                actual = int(matched['away_runs'] + matched['home_runs'] > line)
                gt_brier.append((prob - actual) ** 2)
            except (IndexError, ValueError):
                pass

    return {
        'nrfi_brier': sum(nrfi_brier) / len(nrfi_brier) if nrfi_brier else None,
        'nrfi_count': len(nrfi_brier),
        'ml_brier': sum(ml_brier) / len(ml_brier) if ml_brier else None,
        'ml_count': len(ml_brier),
        'gt_brier': sum(gt_brier) / len(gt_brier) if gt_brier else None,
        'gt_count': len(gt_brier),
    }


def fetch_yesterdays_boxscores(yesterday_str):
    """Pings the MLB API for yesterday's games and loops individual boxscores to prevent truncation."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={yesterday_str}"
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
        if game['status']['statusCode'] not in ['F', 'O']:
            continue

        game_pk = game['gamePk']
        # Hit the specific boxscore endpoint for each game to get the un-truncated player stats!
        box_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        try:
            box_data = requests.get(box_url, timeout=5).json()
        except:
            continue

        boxscore = box_data.get('teams', {})
        for team_side in ['away', 'home']:
            players = boxscore.get(team_side, {}).get('players', {})
            for pid, pdata in players.items():
                name = pdata['person']['fullName']
                clean_name = name.replace('*', '').strip()

                stats = pdata.get('stats', {}).get('batting', {})
                if stats:
                    if clean_name not in actuals:
                        actuals[clean_name] = {'HR': 0, 'Hit': 0, 'TB': 0, 'Run': 0, 'RBI': 0}

                    actuals[clean_name]['HR'] += stats.get('homeRuns', 0)
                    actuals[clean_name]['Hit'] += stats.get('hits', 0)
                    actuals[clean_name]['TB'] += stats.get('totalBases', 0)
                    actuals[clean_name]['Run'] += stats.get('runs', 0)
                    actuals[clean_name]['RBI'] += stats.get('rbi', 0)
    return actuals


def grade_ledger():
    if not os.path.exists("prediction_ledger.csv"):
        print("[!] No prediction_ledger.csv found. Run daily_bets.py first to build history.")
        return

    df = pd.read_csv("prediction_ledger.csv")

    eastern = pytz.timezone('US/Eastern')
    yesterday_str = (datetime.now(eastern) - timedelta(days=1)).strftime('%Y-%m-%d')

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

        # Skip the Game Totals for the player prop grader (handled separately)
        if player == 'GAME_TOTAL':
            continue

        prob = float(row['Prob'])

        if player not in actuals:
            continue

        p_stats = actuals[player]

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

        if actual_outcome == 1 and prob >= 0.50:
            wins += 1
        elif actual_outcome == 0 and prob < 0.50:
            wins += 1
        else:
            losses += 1

        brier = (prob - actual_outcome) ** 2
        brier_scores.append(brier)

    if not brier_scores:
        print("[!] Found bets, but none of the players logged an at-bat yesterday.")
        return

    avg_brier = sum(brier_scores) / len(brier_scores)
    accuracy = (wins / (wins + losses)) * 100

    report = f"📊 **MLB Model Backtest Report ({yesterday_str})** 📊\n"
    report += f"**Brier Score (Props):** {avg_brier:.4f} *(Lower is better, <0.20 is elite)*\n"
    report += f"**Binary Accuracy:** {accuracy:.1f}%\n"
    report += f"**Total Props Graded:** {len(brier_scores)}"

    # ---- Grade game-level markets ----
    try:
        game_outcomes = fetch_game_outcomes(yesterday_str)
        game_grades = grade_game_markets(yesterday_bets, game_outcomes)
        if game_grades['nrfi_count']:
            report += f"\n**NRFI Brier Score:** {game_grades['nrfi_brier']:.4f} ({game_grades['nrfi_count']} predictions)"
        if game_grades['ml_count']:
            report += f"\n**Moneyline Brier Score:** {game_grades['ml_brier']:.4f} ({game_grades['ml_count']} predictions)"
        if game_grades['gt_count']:
            report += f"\n**Game Total Brier Score:** {game_grades['gt_brier']:.4f} ({game_grades['gt_count']} predictions)"
    except Exception as e:
        print(f"[!] Could not grade game markets: {e}")

    print(report)

    if DISCORD_WEBHOOK_URL != "YOUR_WEBHOOK_URL_HERE":
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": report})
            print("Report sent to Discord!")
        except Exception as e:
            print(f"Failed to send to Discord: {e}")


if __name__ == "__main__":
    grade_ledger()

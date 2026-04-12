# true_time_machine.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Import your actual prediction engine
from game_markets_predictor import GameMarketsPredictor

try:
    from feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
except ImportError:
    print("[!] FeatureEngineer not found. This backtest requires it.")
    sys.exit(1)


def fetch_historical_boxscores(date_str):
    """Fetches games and then fetches the specific boxscore for each game."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=weather,linescore"
    resp = requests.get(url)
    if resp.status_code != 200: return []

    data = resp.json()
    games_list = []
    if 'dates' not in data: return games_list

    for game in data['dates'][0].get('games', []):
        if game['status']['statusCode'] == 'F' and game['gameType'] == 'R':
            try:
                game_pk = game['gamePk']
                away_team = game['teams']['away']['team']['name']
                home_team = game['teams']['home']['team']['name']
                total_runs = game['teams']['away'].get('score', 0) + game['teams']['home'].get('score', 0)
                stadium = game['venue']['name']

                weather = game.get('weather', {})
                temp = int(weather.get('temp', 72))
                wind_str = weather.get('wind', '0 mph None')
                wind_speed = int(wind_str.split(' ')[0]) if wind_str[0].isdigit() else 0
                wind_dir = 'out' if 'Out' in wind_str else 'none'
                weather_dict = {'temp': temp, 'wind_speed': wind_speed, 'wind_dir': wind_dir}

                linescore = game.get('linescore', {}).get('innings', [])
                actual_yrfi = 0
                if linescore:
                    a_1st = linescore[0].get('away', {}).get('runs', 0)
                    h_1st = linescore[0].get('home', {}).get('runs', 0)
                    if a_1st > 0 or h_1st > 0: actual_yrfi = 1

                # Fetch the exact boxscore for this specific game
                box_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
                box_resp = requests.get(box_url).json()
                box = box_resp.get('teams', {})
                if not box: continue

                def get_lineup(team_side):
                    lineup = []
                    players_dict = box[team_side].get('players', {})
                    for pid in box[team_side].get('battingOrder', [])[:9]:
                        pkey = f"ID{pid}"
                        p_info = players_dict.get(pkey, {})
                        name = p_info.get('person', {}).get('fullName', 'Unknown')
                        lineup.append({'name': name, 'hand': 'R'})  # Default to R for backtest
                    return lineup

                def get_starter(team_side):
                    pitchers = box[team_side].get('pitchers', [])
                    if not pitchers: return 'Unknown'
                    pid = pitchers[0]  # The starter is always listed first
                    pkey = f"ID{pid}"
                    return box[team_side].get('players', {}).get(pkey, {}).get('person', {}).get('fullName', 'Unknown')

                away_lineup = get_lineup('away')
                home_lineup = get_lineup('home')
                away_pitcher = get_starter('away')
                home_pitcher = get_starter('home')

                # If missing data, skip
                if len(away_lineup) < 9 or len(
                        home_lineup) < 9 or away_pitcher == 'Unknown' or home_pitcher == 'Unknown':
                    print(f"   [!] Skipping {away_team} vs {home_team} - Missing Player Data")
                    continue

                games_list.append({
                    'matchup': f"{away_team} @ {home_team}",
                    'stadium': stadium,
                    'weather': weather_dict,
                    'away_lineup': away_lineup,
                    'home_lineup': home_lineup,
                    'away_pitcher': away_pitcher,
                    'home_pitcher': home_pitcher,
                    'actual_total': total_runs,
                    'actual_yrfi': actual_yrfi
                })
            except Exception as e:
                print(f"   [!] Error processing game {game_pk}: {e}")
                continue
    return games_list


def run_true_time_machine():
    print("=====================================================")
    print(" 🕰️  INITIATING TRUE MONTE CARLO TIME MACHINE 🕰️ ")
    print("=====================================================")

    gmp = GameMarketsPredictor(feature_engineer=fe)

    start_date = datetime(2026, 3, 26)
    end_date = datetime.now() - timedelta(days=1)

    tot_wins = tot_loss = 0
    nrfi_wins = nrfi_loss = 0
    games_graded = 0

    print(f" -> Backtesting 2026 Season from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    print(" -> Simulating 100 pitch-by-pitch iterations per game. Please wait...\n")

    current_date = start_date
    while current_date <= end_date:
        d_str = current_date.strftime('%Y-%m-%d')
        games = fetch_historical_boxscores(d_str)

        if games:
            print(f"[{d_str}] Playing {len(games)} games...")

        for g in games:
            # We use 100 sims here so the backtest takes 30 seconds instead of 10 minutes.
            # Your live daily runner uses the full 2,000 sims!
            res = gmp.predict_full_game(
                away_lineup=g['away_lineup'],
                home_lineup=g['home_lineup'],
                away_pitcher=g['away_pitcher'],
                home_pitcher=g['home_pitcher'],
                stadium=g['stadium'],
                weather=g['weather'],
                n_simulations=100
            )

            # Grade Totals
            ai_line = res['game_total_line']
            ai_lean_over = res['game_total_over'] > 0.50

            if ai_lean_over:
                if g['actual_total'] > ai_line:
                    tot_wins += 1
                elif g['actual_total'] < ai_line:
                    tot_loss += 1
            else:
                if g['actual_total'] < ai_line:
                    tot_wins += 1
                elif g['actual_total'] > ai_line:
                    tot_loss += 1

            # Grade NRFI
            if res['nrfi_prob'] > 0.50:
                if g['actual_yrfi'] == 0:
                    nrfi_wins += 1
                else:
                    nrfi_loss += 1
            else:
                if g['actual_yrfi'] == 1:
                    nrfi_wins += 1
                else:
                    nrfi_loss += 1

            games_graded += 1

        current_date += timedelta(days=1)

    print("\n========================================")
    print(" 📊 TRUE 2026 SEASON BACKTEST RESULTS 📊 ")
    print("========================================")
    if tot_wins + tot_loss > 0:
        print(f" TOTALS (O/U) RECORD: {tot_wins} W - {tot_loss} L ({(tot_wins / (tot_wins + tot_loss)) * 100:.1f}%)")
    if nrfi_wins + nrfi_loss > 0:
        print(
            f" NRFI/YRFI RECORD:    {nrfi_wins} W - {nrfi_loss} L ({(nrfi_wins / (nrfi_wins + nrfi_loss)) * 100:.1f}%)")
    print(f" Games Graded: {games_graded}")
    print("========================================")


if __name__ == "__main__":
    run_true_time_machine()
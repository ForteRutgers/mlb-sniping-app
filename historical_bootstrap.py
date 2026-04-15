# historical_bootstrap.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Import the core physics engine from your daily script
from daily_bets import (
    get_prop_matrices, match_player_name, simulate_full_game_with_archetypes,
    generate_pitcher_profile, PARK_FACTORS
)

SIM_GAMES = 2500  # Optimized for fast historical backtesting

league_avg_batter = {
    '1B_Rate': 0.145, '2B_Rate': 0.045, '3B_Rate': 0.004, 'HR_Rate': 0.030,
    'BB_Rate': 0.085, 'K_Rate': 0.225, 'R_Conv': 0.310, 'RBI_Conv': 0.150,
    'SB_Conv': 0.050, 'Barrel_Rate': 0.08, 'xwOBA': 0.320, 'Archetype': 'Balanced', 'Hand': 'R'
}
league_avg_pitcher = {'CALC_HR9': 1.25, 'K_Rate': 0.22, 'BB_Rate': 0.08, 'H_Rate': 0.24, 'BF_per_Start': 22}


def _process_game(game):
    """Processes a single game's data."""
    if game['status']['statusCode'] not in ['F', 'O']:
        return []

    game_pk = game['gamePk']
    # CRITICAL FIX: Pull the individual box score so it doesn't get truncated!
    try:
        box_data = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=5).json()
    except:
        return []

    info = box_data.get('info', [])
    temp, wind_speed, wind_dir = 72, 0, 'none'
    for item in info:
        if 'Weather' in item.get('label', ''):
            val = item.get('value', '').lower()
            try:
                temp = int([x for x in val.split() if 'degrees' in x or 'f' in x or x.isdigit()][0].replace('f', '').replace('degrees', ''))
                if 'mph' in val: wind_speed = int(val.split('mph')[0].split()[-1])
                if 'out' in val: wind_dir = 'out'
            except:
                pass

    stadium = game.get('venue', {}).get('name', 'Unknown')
    boxscore_teams = box_data.get('teams', {})
    game_results = []

    for is_home in [True, False]:
        team_side = 'home' if is_home else 'away'
        opp_side = 'away' if is_home else 'home'

        team_box = boxscore_teams.get(team_side, {})
        opp_box = boxscore_teams.get(opp_side, {})

        batting_order = team_box.get('battingOrder', [])
        players_dict = team_box.get('players', {})
        opp_players_dict = opp_box.get('players', {})

        if not batting_order: continue

        opp_pitchers = opp_box.get('pitchers', [])
        if not opp_pitchers: continue
        sp_id = f"ID{opp_pitchers[0]}"
        sp_data = opp_players_dict.get(sp_id, {})
        sp_name = sp_data.get('person', {}).get('fullName', 'TBD')
        sp_hand = sp_data.get('person', {}).get('pitchHand', {}).get('code', 'R')

        lineup = []
        for pid in batting_order:
            p_key = f"ID{pid}"
            if p_key in players_dict:
                p_data = players_dict[p_key]
                name = p_data['person']['fullName'].replace('*', '').replace('#', '').strip()
                hand = p_data['person'].get('batSide', {}).get('code', 'R')

                stats = p_data.get('stats', {}).get('batting', {})
                actuals = {
                    'HR': stats.get('homeRuns', 0), 'Hit': stats.get('hits', 0),
                    'TB': stats.get('totalBases', 0), 'Run': stats.get('runs', 0),
                    'RBI': stats.get('rbi', 0)
                }
                lineup.append({'name': name, 'hand': hand, 'actuals': actuals})

        game_results.append({
            'stadium': stadium, 'temp': temp, 'wind_speed': wind_speed, 'wind_dir': wind_dir,
            'opp_pitcher': sp_name, 'opp_p_hand': sp_hand, 'lineup': lineup
        })
    return game_results


def fetch_historical_day(date_str):
    """Pings the MLB API for historical boxscores, weather, and lineups using concurrency."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    try:
        data = requests.get(url, timeout=10).json()
    except Exception:
        return []

    if 'dates' not in data or not data['dates']: return []

    games = data['dates'][0].get('games', [])
    games_data = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(_process_game, games))

    for res in results:
        games_data.extend(res)

    return games_data


def bootstrap_season():
    print("\n=========================================================")
    print("🚀 INITIATING HISTORICAL MLB BOOTSTRAPPER (2025 SEASON)")
    print("=========================================================")

    batters_db, pitchers_db = get_prop_matrices()
    batter_keys = list(batters_db.keys())

    start_date = datetime(2025, 3, 27)
    end_date = datetime(2025, 9, 28)

    training_rows = []

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        sys.stdout.write(f"\rCrunching {date_str}... ")
        sys.stdout.flush()

        games = fetch_historical_day(date_str)

        for g in games:
            park_hr, park_avg = PARK_FACTORS.get(g['stadium'], [1.0, 1.0])
            w_boost = 1 + ((g['temp'] - 70) * 0.01) + ((g['wind_speed'] / 5) * 0.05 if g['wind_dir'] == 'out' else 0)

            p_name = g['opp_pitcher']
            p_hand = g['opp_p_hand']
            p_stats = pitchers_db.get(match_player_name(p_name, list(pitchers_db.keys())), league_avg_pitcher)
            p_hr9 = p_stats['CALC_HR9']
            p_archetype, _ = generate_pitcher_profile(p_hr9)

            for order_idx, b_dict in enumerate(g['lineup']):
                b_name = b_dict['name']
                b_hand = b_dict['hand']
                b_actuals = b_dict['actuals']

                b_stats = batters_db.get(match_player_name(b_name, batter_keys), league_avg_batter).copy()
                b_stats['Hand'] = b_hand
                b_stats['Archetype'] = b_stats.get('Archetype', 'Balanced')
                b_xwoba = b_stats.get('xwOBA', 0.320)

                has_platoon = (b_hand == 'S') or (p_hand != b_hand)

                tracker = {'HR': 0, 'Hit': 0, 'TB': 0, 'Run': 0, 'RBI': 0}
                for _ in range(SIM_GAMES):
                    hr, hits, tb, runs, rbis, _, _, _, _ = simulate_full_game_with_archetypes(
                        b_stats, p_hr9, p_hand, w_boost, park_hr, park_avg, order_idx
                    )
                    if hr >= 1: tracker['HR'] += 1
                    if hits >= 1: tracker['Hit'] += 1
                    if tb >= 2: tracker['TB'] += 1
                    if runs >= 1: tracker['Run'] += 1
                    if rbis >= 1: tracker['RBI'] += 1

                base_row = {
                    'Date': date_str, 'Stadium': g['stadium'], 'Temp': g['temp'], 'Wind_Speed': g['wind_speed'],
                    'Lineup_Spot': order_idx + 1, 'Batter': b_name, 'Batter_Hand': b_hand,
                    'Batter_Archetype': b_stats['Archetype'], 'Batter_xwOBA': b_xwoba, 'Pitcher': p_name,
                    'Pitcher_Hand': p_hand, 'Pitcher_Archetype': p_archetype, 'Pitcher_HR9': p_hr9,
                    'Platoon_Adv': 1 if has_platoon else 0, 'Player': b_name
                }

                for market in ['HR', 'Hit', 'TB', 'Run', 'RBI']:
                    prob = tracker[market] / SIM_GAMES
                    actual = 1 if b_actuals[market] >= (2 if market == 'TB' else 1) else 0

                    row = base_row.copy()
                    row['Market'] = market
                    row['Prob'] = prob
                    row['Actual_Outcome'] = actual
                    training_rows.append(row)

        current_date += timedelta(days=1)
        time.sleep(0.1)

    print("\n\n[SUCCESS] 2025 Season completely simulated and graded!")

    df = pd.DataFrame(training_rows)
    df.to_csv("historical_training_data.csv", index=False)
    print(f"Saved {len(df)} heavily engineered training rows to historical_training_data.csv")


if __name__ == "__main__":
    bootstrap_season()
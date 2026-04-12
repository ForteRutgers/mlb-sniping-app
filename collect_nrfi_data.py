# collect_nrfi_data.py
import requests
import pandas as pd
from datetime import datetime
import os

PARK_FACTORS = {
    "Coors Field": 1.35, "Great American Ball Park": 1.18, "Yankee Stadium": 1.05,
    "Wrigley Field": 1.04, "Truist Park": 1.04, "Dodger Stadium": 1.08,
    "Citizens Bank Park": 1.08, "Minute Maid Park": 1.05, "American Family Field": 1.03,
    "Fenway Park": 1.06, "Guaranteed Rate Field": 1.01, "Angel Stadium": 1.02,
    "Globe Life Field": 1.02, "Oriole Park at Camden Yards": 1.02, "Rogers Centre": 1.00,
    "Target Field": 1.00, "Nationals Park": 0.99, "Progressive Field": 0.99,
    "Chase Field": 0.99, "Kauffman Stadium": 0.98, "Citi Field": 0.97,
    "Tropicana Field": 0.97, "loanDepot park": 0.95, "Comerica Park": 0.96,
    "Oakland Coliseum": 0.94, "Petco Park": 0.97, "T-Mobile Park": 0.95,
    "PNC Park": 0.92, "Oracle Park": 0.88, "Busch Stadium": 0.97
}


def get_nrfi_games(start_date, end_date):
    print(f" -> Fetching 1st Inning (FTTO) Data from {start_date} to {end_date}...")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}&hydrate=game(content(summary)),decisions,probablePitcher,weather,linescore"

    response = requests.get(url)
    if response.status_code != 200: return []

    data = response.json()
    games_list = []

    if 'dates' not in data: return games_list

    for date_obj in data['dates']:
        for game in date_obj['games']:
            if game['status']['statusCode'] == 'F' and game['gameType'] == 'R':
                try:
                    linescore = game.get('linescore', {}).get('innings', [])
                    if not linescore: continue

                    # Check 1st Inning Runs
                    first_inning = linescore[0]
                    away_1st = first_inning.get('away', {}).get('runs', 0)
                    home_1st = first_inning.get('home', {}).get('runs', 0)

                    # 0 = NRFI, 1 = YRFI
                    yrfi = 1 if (away_1st > 0 or home_1st > 0) else 0

                    stadium = game['venue']['name']
                    park_factor = PARK_FACTORS.get(stadium, 1.00)

                    weather = game.get('weather', {})
                    temp = int(weather.get('temp', 72))
                    wind_str = weather.get('wind', '0 mph None')
                    wind_speed = int(wind_str.split(' ')[0]) if wind_str[0].isdigit() else 0
                    wind_out = 1 if 'Out' in wind_str else 0

                    away_pitcher_id = game['teams']['away'].get('probablePitcher', {}).get('id', 0)
                    home_pitcher_id = game['teams']['home'].get('probablePitcher', {}).get('id', 0)

                    if away_pitcher_id == 0 or home_pitcher_id == 0: continue

                    games_list.append({
                        'park_factor': park_factor,
                        'temp': temp,
                        'wind_speed': wind_speed,
                        'wind_out': wind_out,
                        'yrfi_outcome': yrfi
                    })
                except Exception:
                    continue

    return games_list


def run_collection():
    print("========================================")
    print("   MLB NRFI/YRFI HISTORICAL COLLECTOR   ")
    print("========================================")

    season_2024 = get_nrfi_games("2024-03-28", "2024-09-29")
    season_2025 = get_nrfi_games("2025-03-27", "2025-09-28")
    all_games = season_2024 + season_2025

    df = pd.DataFrame(all_games)
    file_name = "nrfi_training_data.csv"
    df.to_csv(file_name, index=False)

    print("\n[SUCCESS] NRFI Data Collection Complete!")
    print(f" -> Saved {len(df)} historical 1st innings to {file_name}")


if __name__ == "__main__":
    run_collection()
# fast_time_machine.py
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Park Factors & Config
# ---------------------------------------------------------------------------
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


def load_models():
    totals_model = None
    totals_cols = []
    nrfi_model = None
    nrfi_cols = []

    if os.path.exists("totals_model.json"):
        totals_model = xgb.XGBRegressor()
        totals_model.load_model("totals_model.json")
        with open("totals_model_features.txt", "r") as f:
            totals_cols = f.read().strip().split(",")

    if os.path.exists("nrfi_model.json"):
        nrfi_model = xgb.XGBClassifier()
        nrfi_model.load_model("nrfi_model.json")
        with open("nrfi_model_features.txt", "r") as f:
            nrfi_cols = f.read().strip().split(",")

    return totals_model, totals_cols, nrfi_model, nrfi_cols


def fetch_historical_day(date_str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=game(content(summary)),weather,linescore"
    resp = requests.get(url)
    if resp.status_code != 200: return []

    data = resp.json()
    games = []
    if 'dates' not in data: return games

    for game in data['dates'][0].get('games', []):
        if game['status']['statusCode'] == 'F' and game['gameType'] == 'R':
            try:
                away_team = game['teams']['away']['team']['name']
                home_team = game['teams']['home']['team']['name']
                away_runs = game['teams']['away'].get('score', 0)
                home_runs = game['teams']['home'].get('score', 0)
                total_runs = away_runs + home_runs

                linescore = game.get('linescore', {}).get('innings', [])
                yrfi_outcome = 0
                if linescore:
                    a_1st = linescore[0].get('away', {}).get('runs', 0)
                    h_1st = linescore[0].get('home', {}).get('runs', 0)
                    if a_1st > 0 or h_1st > 0: yrfi_outcome = 1

                stadium = game['venue']['name']
                weather = game.get('weather', {})
                temp = int(weather.get('temp', 72))
                wind_str = weather.get('wind', '0 mph None')
                wind_speed = int(wind_str.split(' ')[0]) if wind_str[0].isdigit() else 0
                wind_out = 1 if 'Out' in wind_str else 0

                games.append({
                    'matchup': f"{away_team} @ {home_team}",
                    'stadium': stadium,
                    'park_factor': PARK_FACTORS.get(stadium, 1.0),
                    'temp': temp,
                    'wind_speed': wind_speed,
                    'wind_out': wind_out,
                    'actual_total': total_runs,
                    'actual_yrfi': yrfi_outcome,
                    # We use league average fallbacks for past pitchers/hitters in this fast script
                    'away_top3_xwoba': 0.320, 'home_top3_xwoba': 0.320,
                    'away_pitcher_k': 0.22, 'home_pitcher_k': 0.22
                })
            except Exception:
                continue
    return games


def run_time_machine():
    print("========================================")
    print(" 🚀 INITIATING XGBOOST TIME MACHINE 🚀 ")
    print("========================================")

    t_mod, t_cols, n_mod, n_cols = load_models()
    if not t_mod or not n_mod:
        print("[!] Missing AI models! Run your training scripts first.")
        return

    # 2026 Opening Day to Yesterday
    start_date = datetime(2026, 3, 26)
    end_date = datetime.now() - timedelta(days=1)

    current_date = start_date

    tot_wins = tot_loss = 0
    nrfi_wins = nrfi_loss = 0

    print(
        f" -> Scanning historical MLB APIs from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...\n")

    while current_date <= end_date:
        d_str = current_date.strftime('%Y-%m-%d')
        games = fetch_historical_day(d_str)

        for g in games:
            # Predict Totals
            t_feat = {c: g.get(c, 0) for c in t_cols}
            t_pred = float(t_mod.predict(pd.DataFrame([t_feat]))[0])
            # Round prediction to nearest half-run for a standard Vegas line
            line = round(t_pred * 2) / 2

            # Did it go Over or Under our predicted line?
            if t_pred > g['actual_total']:
                tot_loss += 1  # We predicted higher, actual was lower (Over missed)
            else:
                tot_wins += 1  # We predicted lower, actual was higher (Under hit)

            # Predict NRFI (Isolating the environment & average player expectations)
            n_feat = {c: g.get(c, 0) for c in n_cols}
            n_pred_yrfi = float(n_mod.predict_proba(pd.DataFrame([n_feat]))[0, 1])

            # Grade NRFI/YRFI (Taking whichever side is > 50%)
            if n_pred_yrfi > 0.50:
                if g['actual_yrfi'] == 1:
                    nrfi_wins += 1
                else:
                    nrfi_loss += 1
            else:
                if g['actual_yrfi'] == 0:
                    nrfi_wins += 1
                else:
                    nrfi_loss += 1

        current_date += timedelta(days=1)

    print("========================================")
    print(" 📊 2026 SEASON BACKTEST RESULTS 📊 ")
    print("========================================")
    if tot_wins + tot_loss > 0:
        print(f" TOTALS (O/U) RECORD: {tot_wins} W - {tot_loss} L ({(tot_wins / (tot_wins + tot_loss)) * 100:.1f}%)")
    if nrfi_wins + nrfi_loss > 0:
        print(
            f" NRFI/YRFI RECORD:    {nrfi_wins} W - {nrfi_loss} L ({(nrfi_wins / (nrfi_wins + nrfi_loss)) * 100:.1f}%)")
    print("========================================")


if __name__ == "__main__":
    run_time_machine()
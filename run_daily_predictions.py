# run_daily_predictions.py
"""
Main script for generating comprehensive daily MLB predictions.
Saves results to enhanced_predictions_report.txt and mlb_predictions.db
Focuses exclusively on Game Totals (O/U) and NRFI/YRFI models.
"""

import sqlite3
from datetime import datetime
import pytz

from live_scraper import get_todays_matchups
from game_markets_predictor import GameMarketsPredictor

# ---------------------------------------------------------------------------
# League Average Fallbacks
# ---------------------------------------------------------------------------
try:
    from feature_engineering import FeatureEngineer, LEAGUE_AVG_BATTER, LEAGUE_AVG_PITCHER

    _FE_AVAILABLE = True
except ImportError:
    _FE_AVAILABLE = False
    LEAGUE_AVG_BATTER = {
        "1B_Rate": 0.145, "2B_Rate": 0.045, "3B_Rate": 0.004, "HR_Rate": 0.030,
        "BB_Rate": 0.085, "K_Rate": 0.225, "R_Conv": 0.310, "RBI_Conv": 0.150,
        "SB_Conv": 0.050, "Barrel_Rate": 0.080, "xwOBA": 0.320,
        "Archetype": "Balanced", "Hand": "R",
    }
    LEAGUE_AVG_PITCHER = {
        "CALC_HR9": 1.25, "K_Rate": 0.22, "BB_Rate": 0.08, "H_Rate": 0.24, "BF_per_Start": 22
    }


# ---------------------------------------------------------------------------
# Database & Formatting Logic
# ---------------------------------------------------------------------------

def _write_to_sqlite(game_ledger_data):
    db_path = 'mlb_predictions.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS game_predictions
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY,
                          date
                          TEXT,
                          away_team
                          TEXT,
                          home_team
                          TEXT,
                          stadium
                          TEXT,
                          market
                          TEXT,
                          probability
                          REAL,
                          fair_odds
                          TEXT,
                          game_total_line
                          REAL
                      )''')

    eastern = pytz.timezone("US/Eastern")
    run_date = datetime.now(eastern).strftime('%Y-%m-%d')

    for entry in game_ledger_data:
        cursor.execute("DELETE FROM game_predictions WHERE date=? AND away_team=? AND home_team=?",
                       (run_date, entry['away_team'], entry['home_team']))

        game_res = entry['game_result']
        total_line = game_res.get('game_total_line', 8.5)

        markets = [
            ('NRFI', game_res.get('nrfi_prob', 0.5), "N/A"),
            ('Total_Over', game_res.get('game_total_over', 0.5), "N/A"),
            ('Total_Under', game_res.get('game_total_under', 0.5), "N/A")
        ]

        for m_name, prob, odds in markets:
            cursor.execute(
                '''INSERT INTO game_predictions
                   (date, away_team, home_team, stadium, market, probability, fair_odds, game_total_line)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (run_date, entry['away_team'], entry['home_team'], entry['stadium'], m_name, float(prob), str(odds),
                 float(total_line))
            )

    conn.commit()
    conn.close()


def _recommendation(nrfi_prob: float) -> str:
    if nrfi_prob >= 0.65: return "🔥🔥🔥 STRONG NRFI"
    if nrfi_prob >= 0.58: return "🔥🔥 LEAN NRFI"
    if nrfi_prob <= 0.35: return "💥💥💥 STRONG YRFI"
    if nrfi_prob <= 0.42: return "💥💥 LEAN YRFI"
    return "➖ NEUTRAL"

    def _format_game_report(away_t, home_t, game_r, game_time) -> str:
        over_p = game_r.get('game_total_over', 0.5) * 100
        under_p = game_r.get('game_total_under', 0.5) * 100
        total_line = game_r.get('game_total_line', 8.5)
        nrfi_p = game_r.get('nrfi_prob', 0.5) * 100
        yrfi_p = (1 - (nrfi_p / 100)) * 100

        # Clean, modern Discord formatting using blockquotes
        return (
            f"⚾ **{away_t} @ {home_t}** ({game_time})\n"
            f"> 📊 **O/U {total_line}:** Over {over_p:.0f}% | Under {under_p:.0f}%\n"
            f"> 🕒 **1st Inn:** NRFI {nrfi_p:.0f}% | YRFI {yrfi_p:.0f}%\n"
            f"> 🎯 **Edge:** {_recommendation(nrfi_p / 100)}\n\n"
        )
    )


# ---------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------------------------

def run_daily_predictions():
    print("🚀 Running Streamlined MLB Prediction Engine...")

    all_matchups = get_todays_matchups()
    if not all_matchups:
        print("[!] No games found.")
        return

    predictor = GameMarketsPredictor()
    full_report = "# ⚾ MLB Daily System 2 Predictions\n\n"
    game_ledger = []

    # Process in pairs (Away then Home) because live_scraper returns 2 entries per game
    for i in range(0, len(all_matchups), 2):
        if i + 1 >= len(all_matchups): break

        away_data = all_matchups[i]
        home_data = all_matchups[i + 1]

        away_name = away_data['team']
        home_name = home_data['team']
        stadium = home_data['home_stadium']
        game_time = home_data['game_time']
        weather = home_data['weather']

        away_lineup = away_data['lineup']
        home_lineup = home_data['lineup']

        # The opposing pitcher for the away team is the home pitcher
        home_pitcher = away_data['opposing_pitcher']
        home_pitcher_hand = away_data['opposing_pitcher_hand']
        away_pitcher = home_data['opposing_pitcher']
        away_pitcher_hand = home_data['opposing_pitcher_hand']

        print(f"Analyzing {away_name} @ {home_name}...")

        # Run ONE Simulation that captures both Game Totals and NRFI
        game_results = predictor.predict_full_game(
            away_lineup=away_lineup,
            home_lineup=home_lineup,
            away_pitcher=away_pitcher,
            home_pitcher=home_pitcher,
            stadium=stadium,
            away_pitcher_hand=away_pitcher_hand,
            home_pitcher_hand=home_pitcher_hand,
            weather=weather
        )

        full_report += _format_game_report(
            away_name, home_name,
            game_results, game_time
        )

        game_ledger.append({
            'away_team': away_name,
            'home_team': home_name,
            'stadium': stadium,
            'game_result': game_results
        })

    with open("enhanced_predictions_report.txt", "w") as f:
        f.write(full_report)

    _write_to_sqlite(game_ledger)
    print(f"✅ Success! Report generated.")


if __name__ == "__main__":
    run_daily_predictions()
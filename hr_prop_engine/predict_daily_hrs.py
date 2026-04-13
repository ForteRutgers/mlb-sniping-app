import pandas as pd
import joblib
import os
from datetime import datetime


# -------------------------------------------------------------------
# ISOLATION SAFEGUARD:
# This script only reads the isolated model from the sandbox.
# It generates a standalone HR prop report to the console or a separate
# text file, completely avoiding your main betting dashboard logic.
# -------------------------------------------------------------------

def get_todays_matchups() -> pd.DataFrame:
    """
    Simulates pulling today's live MLB matchups, weather, and stadium data.
    In production, this would hit your OpenWeather API and PyBaseball stats.
    """
    print("Fetching today's batter vs. pitcher matchups...")

    # Simulated data for today's slate
    data = pd.DataFrame({
        'player_name': ['Aaron Judge', 'Shohei Ohtani', 'Kyle Schwarber', 'Steven Kwan'],
        'batter_barrel_pct': [0.22, 0.19, 0.18, 0.03],  # Statcast
        'batter_launch_angle': [15.5, 13.0, 18.2, 5.0],
        'pitcher_hr_per_9': [1.8, 1.1, 2.2, 0.8],  # Matchup target
        'park_factor_hr': [115, 105, 112, 95],  # Park factor
        'weather_temp': [85, 72, 90, 60],  # Weather
        'weather_wind_outward': [10, 0, 15, -5]  # Wind
    })
    return data


def generate_hr_predictions():
    """
    Loads the ensemble model and generates probabilities for today's slate.
    """
    model_path = 'hr_prop_engine/models/hr_ensemble.pkl'

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please run train_hr_model.py first.")
        return

    # Load the safely isolated models
    print("Loading HR Ensemble Models...")
    models = joblib.load(model_path)
    lr_model = models['lr']
    xgb_model = models['xgb']
    features = models['features']

    # Get today's data
    todays_data = get_todays_matchups()
    X_today = todays_data[features]

    # Predict using both models and average the probabilities
    lr_probs = lr_model.predict_proba(X_today)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_today)[:, 1]
    ensemble_probs = (lr_probs + xgb_probs) / 2

    todays_data['hr_probability'] = ensemble_probs

    # Sort by highest probability
    todays_data = todays_data.sort_values(by='hr_probability', ascending=False)

    print("\n" + "=" * 40)
    print(f"🔥 DAILY HR PROP PREDICTIONS ({datetime.now().strftime('%Y-%m-%d')}) 🔥")
    print("=" * 40)

    for index, row in todays_data.iterrows():
        prob_pct = row['hr_probability'] * 100
        # Simple threshold for betting signals
        signal = "⭐ PLAY" if prob_pct > 15.0 else "PASS"
        print(f"{row['player_name']:<15} | HR Prob: {prob_pct:>5.1f}% | {signal}")

    print("=" * 40 + "\n")


if __name__ == "__main__":
    generate_hr_predictions()
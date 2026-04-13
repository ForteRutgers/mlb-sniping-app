import pandas as pd
import joblib
import os
from datetime import datetime


# -------------------------------------------------------------------
# ISOLATION SAFEGUARD:
# This script loads our new real-data model and feeds it aligned
# pre-game averages.
# -------------------------------------------------------------------

def get_todays_matchups() -> pd.DataFrame:
    """
    Simulates today's pre-game rolling averages.
    In full production, PyBaseball would pull these averages live.
    """
    print("Fetching today's batter vs. pitcher matchups...")

    data = pd.DataFrame({
        'player_name': ['Aaron Judge', 'Shohei Ohtani', 'Kyle Schwarber', 'Steven Kwan'],
        # We now feed the model their AVERAGE expected stats before the game
        'batter_exit_velocity': [95.5, 94.0, 92.5, 85.1],
        'batter_launch_angle': [15.5, 13.0, 18.2, 5.0],
        'pitcher_release_speed': [94.0, 92.5, 96.0, 90.0],
        'pitcher_spin_rate': [2400, 2250, 2500, 2100],
        'park_factor_hr': [115, 105, 112, 95],
        'weather_temp': [85, 72, 90, 60],
        'weather_wind_outward': [10, 0, 15, -5]
    })
    return data


def generate_hr_predictions():
    model_path = 'hr_prop_engine/models/hr_ensemble.pkl'

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please run train_hr_model.py first.")
        return

    print("Loading REAL DATA HR Ensemble Models...")
    models = joblib.load(model_path)
    lr_model = models['lr']
    xgb_model = models['xgb']
    features = models['features']  # These are the exact columns the model demands

    todays_data = get_todays_matchups()

    # Isolate only the specific columns the model was trained on
    X_today = todays_data[features]

    # Predict
    lr_probs = lr_model.predict_proba(X_today)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_today)[:, 1]
    ensemble_probs = (lr_probs + xgb_probs) / 2

    todays_data['hr_probability'] = ensemble_probs
    todays_data = todays_data.sort_values(by='hr_probability', ascending=False)

    print("\n" + "=" * 40)
    print(f"🔥 DAILY HR PROP PREDICTIONS ({datetime.now().strftime('%Y-%m-%d')}) 🔥")
    print("=" * 40)

    for index, row in todays_data.iterrows():
        prob_pct = row['hr_probability'] * 100
        signal = "⭐ PLAY" if prob_pct > 15.0 else "PASS"
        print(f"{row['player_name']:<15} | HR Prob: {prob_pct:>5.1f}% | {signal}")

    print("=" * 40 + "\n")


if __name__ == "__main__":
    generate_hr_predictions()
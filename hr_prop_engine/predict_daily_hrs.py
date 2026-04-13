import pandas as pd
import joblib
import os
from datetime import datetime

# -------------------------------------------------------------------
# DISCORD INTEGRATION:
# This script now saves its top picks to 'hr_prop_engine/discord_snippet.txt'
# so your main Discord notifier can find and include them.
# -------------------------------------------------------------------

def get_todays_matchups() -> pd.DataFrame:
    """Simulates today's slate using real 2023 season averages."""
    data = pd.DataFrame({
        'player_name': ['Aaron Judge', 'Shohei Ohtani', 'Kyle Schwarber', 'Steven Kwan'],
        'batter_exit_velocity': [97.6, 94.4, 92.3, 85.5],
        'batter_launch_angle': [17.1, 13.2, 19.2, 6.0],
        'pitcher_release_speed': [93.5, 93.5, 93.5, 93.5],
        'pitcher_spin_rate': [2250, 2250, 2250, 2250],
        'park_factor_hr': [115, 105, 112, 95],
        'weather_temp': [85, 72, 90, 60],
        'weather_wind_outward': [10, 0, 15, -5]
    })
    return data

def generate_hr_predictions():
    model_path = 'hr_prop_engine/models/hr_ensemble.pkl'
    if not os.path.exists(model_path):
        return

    models = joblib.load(model_path)
    lr_model = models['lr']
    xgb_model = models['xgb']
    features = models['features']

    todays_data = get_todays_matchups()
    X_today = todays_data[features]

    lr_probs = lr_model.predict_proba(X_today)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_today)[:, 1]
    ensemble_probs = (lr_probs + xgb_probs) / 2

    todays_data['hr_probability'] = ensemble_probs
    todays_data = todays_data.sort_values(by='hr_probability', ascending=False)

    # --- NEW: Build Discord Snippet ---
    discord_lines = ["\n**🚀 EXPERIMENTAL HR PROP PICKS**"]
    has_plays = False

    for index, row in todays_data.iterrows():
        prob_pct = row['hr_probability'] * 100
        if prob_pct > 5.0: # Only send players with a decent edge to Discord
            has_plays = True
            discord_lines.append(f"• **{row['player_name']}**: {prob_pct:.1f}% chance")

    if not has_plays:
        discord_lines.append("• No high-value HR props identified for today.")

    # Save the snippet for the main script to grab
    with open('hr_prop_engine/discord_snippet.txt', 'w') as f:
        f.write("\n".join(discord_lines))

    # Also keep the console printout for debugging
    print("\n" + "="*40)
    print(f"🔥 DAILY HR PROP PREDICTIONS ({datetime.now().strftime('%Y-%m-%d')}) 🔥")
    print("="*40)
    for index, row in todays_data.iterrows():
        prob_pct = row['hr_probability'] * 100
        signal = "⭐ PLAY" if prob_pct > 5.0 else "PASS"
        print(f"{row['player_name']:<15} | HR Prob: {prob_pct:>5.1f}% | {signal}")
    print("="*40 + "\n")

if __name__ == "__main__":
    generate_hr_predictions()
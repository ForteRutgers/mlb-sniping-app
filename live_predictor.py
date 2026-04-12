import pandas as pd
import xgboost as xgb

def run_live_predictions():
    print("1. Loading the Artificial Brain...")
    # Initialize an empty model, then load your saved weights
    model = xgb.XGBClassifier()
    model.load_model("home_run_brain.json")
    print("Brain loaded successfully!\n")

    print("2. Setting up our Matchup Scenarios...")
    # We are simulating Bryce Harper hitting a perfect barrel.
    # We will test the EXACT SAME swing in two different stadiums.
    scenarios = pd.DataFrame({
        'batter': ['Harper (in Cincy)', 'Harper (in Detroit)'],
        'launch_speed': [104.0, 104.0],  # 104 mph exit velocity
        'launch_angle': [28.0, 28.0],    # 28 degree launch angle (perfect barrel)
        'release_speed': [95.0, 95.0],   # 95 mph pitch
        'hr_park_factor': [134, 82]      # 134 = Great American Ball Park, 82 = Comerica Park
    })

    # Calculate our custom kinetic features for these swings
    scenarios['is_hard_hit'] = (scenarios['launch_speed'] >= 95.0).astype(int)
    scenarios['is_barrel'] = ((scenarios['launch_speed'] >= 98.0) & (scenarios['launch_angle'] >= 26.0) & (scenarios['launch_angle'] <= 30.0)).astype(int)
    scenarios['is_blast'] = ((scenarios['launch_speed'] >= 105.0) & (scenarios['launch_angle'] >= 20.0) & (scenarios['launch_angle'] <= 35.0)).astype(int)

    # Reorder columns to match EXACTLY what the AI was trained on
    features = ['launch_speed', 'launch_angle', 'release_speed', 'is_hard_hit', 'is_barrel', 'is_blast', 'hr_park_factor']
    X_live = scenarios[features]

    print("3. Asking the AI for Probabilities...")
    # predict_proba returns the percentage chance of a home run
    probabilities = model.predict_proba(X_live)[:, 1]
    scenarios['single_ab_hr_prob'] = probabilities

    print("\n--- LIVE AI PREDICTIONS ---")
    for index, row in scenarios.iterrows():
        
        # Apply the Game-Level Math from your original blueprint!
        single_prob = row['single_ab_hr_prob']
        game_prob = 1 - (1 - single_prob)**3.8
        
        # Convert to Implied Fair American Odds
        fair_odds = (100 / game_prob) - 100

        print(f"\nScenario: {row['batter']}")
        print(f"  - Swing: {row['launch_speed']} mph at {row['launch_angle']} degrees")
        print(f"  - Single At-Bat HR Probability: {single_prob * 100:.1f}%")
        print(f"  - Full Game HR Probability (3.8 ABs): {game_prob * 100:.1f}%")
        print(f"  - Implied Fair Odds for your bet: +{fair_odds:.0f}")

if __name__ == "__main__":
    run_live_predictions()
# Complete train_brain.py content
import pandas as pd
import numpy as np
import xgboost as xgb
from pybaseball import statcast
from datetime import datetime, timedelta
import os

# Statcast Home Run Park Factors to match the main engine
PARK_FACTORS = {
    'COL': 113, 'CIN': 126, 'NYY': 110, 'SF': 81, 'SEA': 96, 'PIT': 82,
    'CHC': 106, 'ATL': 105, 'LAD': 118, 'PHI': 115, 'MIL': 108, 'HOU': 112,
    'BOS': 109, 'CWS': 103, 'LAA': 105, 'TEX': 104, 'BAL': 105, 'TOR': 100,
    'TB': 95, 'MIA': 93, 'DET': 94, 'OAK': 92, 'SD': 96, 'CLE': 98,
    'KC': 97, 'MIN': 99, 'NYM': 95, 'STL': 95, 'WSH': 98, 'ARI': 98
}

def fetch_yesterdays_data():
    """Downloads yesterday's exact pitch-by-pitch Statcast data."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Downloading Statcast data for {yesterday}...")
    
    try:
        df = statcast(start_dt=yesterday, end_dt=yesterday)
        if df is None or df.empty:
            print("No data found for yesterday. Games may not have been played.")
            return None
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def engineer_features(df):
    """Processes raw Statcast data into the exact format the ML brain expects."""
    print("Processing physics and park factors...")
    
    # Filter only for batted balls (ignore walks, strikeouts, fouls)
    hit_events = ['single', 'double', 'triple', 'home_run', 'field_out', 'line_out', 'grounded_into_dp', 'force_out']
    df = df[df['events'].isin(hit_events)].copy()
    
    # Drop rows missing crucial physics data
    df = df.dropna(subset=['launch_speed', 'launch_angle', 'release_speed'])
    
    # 1. Target Variable: Was it a Home Run? (1 = Yes, 0 = No)
    df['is_hr'] = np.where(df['events'] == 'home_run', 1, 0)
    
    # 2. Hard Hit (Exit Velo >= 95 mph)
    df['is_hard_hit'] = np.where(df['launch_speed'] >= 95.0, 1, 0)
    
    # 3. Barrel (Strict Definition: EV >= 98, LA between 26-30 degrees)
    df['is_barrel'] = np.where(
        (df['launch_speed'] >= 98.0) & (df['launch_angle'] >= 26.0) & (df['launch_angle'] <= 30.0), 1, 0
    )
    
    # 4. Blast/Bomb (Elite Contact: EV >= 100, LA between 26-30 degrees)
    df['is_blast'] = np.where(
        (df['launch_speed'] >= 100.0) & (df['launch_angle'] >= 26.0) & (df['launch_angle'] <= 30.0), 1, 0
    )
    
    # 5. Park Factor Injection
    # Statcast uses abbreviations (e.g., 'NYY') for home_team
    df['hr_park_factor'] = df['home_team'].map(PARK_FACTORS).fillna(100)
    
    # Select only the features the model was trained on, in the exact order
    feature_cols = ['launch_speed', 'launch_angle', 'release_speed', 'is_hard_hit', 'is_barrel', 'is_blast', 'hr_park_factor']
    
    X = df[feature_cols]
    y = df['is_hr']
    
    return X, y

def update_brain():
    model_path = "home_run_brain.json"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. You must run the initial training first.")
        return

    # 1. Get Yesterday's Data
    raw_data = fetch_yesterdays_data()
    if raw_data is None:
        return
        
    # 2. Process into ML Features
    X_new, y_new = engineer_features(raw_data)
    
    if len(X_new) == 0:
        print("No valid batted ball events found to train on.")
        return
        
    print(f"Training on {len(X_new)} new batted ball events...")

    # 3. Load the existing Brain and update it with true incremental boosting
    booster = xgb.Booster()
    booster.load_model(model_path)

    dtrain = xgb.DMatrix(X_new, label=y_new)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    updated_booster = xgb.train(
        params,
        dtrain,
        num_boost_round=10,
        xgb_model=booster,
        verbose_eval=False,
    )

    # 4. Save the smarter brain back to disk
    updated_booster.save_model(model_path)
    print(f"[SUCCESS] The Home Run Brain has successfully learned from yesterday's games and updated its weights.")

if __name__ == "__main__":
    update_brain()
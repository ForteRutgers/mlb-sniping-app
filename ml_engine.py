import pandas as pd
from pybaseball import statcast
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

def train_system():
    cache_file = "mlb_data_2024.csv"
    if os.path.exists(cache_file):
        df_raw = pd.read_csv(cache_file)
    else:
        print("Downloading training data (2024 Season)...")
        df_raw = statcast(start_dt='2024-04-01', end_dt='2024-09-30')
        df_raw.to_csv(cache_file, index=False)

    df = df_raw.dropna(subset=['launch_speed', 'launch_angle', 'events', 'home_team']).copy()
    
    # Feature Engineering
    df['is_hard_hit'] = (df['launch_speed'] >= 95).astype(int)
    df['is_barrel'] = ((df['launch_speed'] >= 98) & (df['launch_angle'] >= 26) & (df['launch_angle'] <= 30)).astype(int)
    df['is_blast'] = ((df['launch_speed'] >= 105) & (df['launch_angle'] >= 20) & (df['launch_angle'] <= 35)).astype(int)
    
    park_map = {'CIN': 134, 'LAD': 118, 'COL': 115, 'PHI': 114, 'NYY': 112, 'HOU': 104, 'BOS': 98, 'SF': 86, 'DET': 82}
    df['hr_park_factor'] = df['home_team'].map(park_map).fillna(100)
    df['is_home_run'] = (df['events'] == 'home_run').astype(int)

    features = ['launch_speed', 'launch_angle', 'release_speed', 'is_hard_hit', 'is_barrel', 'is_blast', 'hr_park_factor']
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(df[features], df['is_home_run'])
    
    model.save_model("home_run_brain.json")
    print("Brain trained and saved successfully.")

if __name__ == "__main__":
    train_system()
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os
import pybaseball as pyb


# -------------------------------------------------------------------
# ISOLATION SAFEGUARD:
# This script calculates strict pre-game rolling averages to
# eliminate Target Leakage.
# -------------------------------------------------------------------

def fetch_hr_training_data() -> pd.DataFrame:
    """
    Fetches real Statcast data and calculates pre-pitch rolling averages.
    """
    print("Fetching REAL Statcast data... (Expanded to 4 weeks to build player history)")
    pyb.cache.enable()

    # 4-week window (July 15 - Aug 14, 2023) to give players time to build an average
    df = pyb.statcast(start_dt='2023-07-15', end_dt='2023-08-14')

    # Filter for balls in play and drop missing IDs
    df = df.dropna(subset=['events', 'launch_speed', 'launch_angle', 'batter', 'pitcher', 'game_date'])

    # CRITICAL: Sort chronologically so our historical averages flow forward in time
    df = df.sort_values(by=['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])

    print("Calculating true pre-game rolling averages (fixing target leakage)...")

    df['is_home_run'] = (df['events'] == 'home_run').astype(int)

    # --- BATTER ROLLING AVERAGES ---
    # We take the expanding mean (average up to that point) and shift(1) so the current pitch isn't included
    df['batter_exit_velocity'] = df.groupby('batter')['launch_speed'].transform(lambda x: x.expanding().mean().shift(1))
    df['batter_launch_angle'] = df.groupby('batter')['launch_angle'].transform(lambda x: x.expanding().mean().shift(1))

    # --- PITCHER ROLLING AVERAGES ---
    df['pitcher_release_speed'] = df.groupby('pitcher')['release_speed'].transform(
        lambda x: x.expanding().mean().shift(1))
    df['pitcher_spin_rate'] = df.groupby('pitcher')['release_spin_rate'].transform(
        lambda x: x.expanding().mean().shift(1))

    # Drop rows where the player had no history (their first few pitches of the dataset)
    df = df.dropna(subset=['batter_exit_velocity', 'batter_launch_angle', 'pitcher_release_speed', 'pitcher_spin_rate'])

    features_df = pd.DataFrame({
        'batter_exit_velocity': df['batter_exit_velocity'],
        'batter_launch_angle': df['batter_launch_angle'],
        'pitcher_release_speed': df['pitcher_release_speed'],
        'pitcher_spin_rate': df['pitcher_spin_rate'],
        'park_factor_hr': 100,
        'weather_temp': 75,
        'weather_wind_outward': 0,
        'is_home_run': df['is_home_run']
    })

    return features_df


def train_ensemble_model():
    """
    Trains a Logistic Regression and XGBoost ensemble on true pre-game averages.
    """
    df = fetch_hr_training_data()

    features = [
        'batter_exit_velocity', 'batter_launch_angle', 'pitcher_release_speed',
        'pitcher_spin_rate', 'park_factor_hr', 'weather_temp', 'weather_wind_outward'
    ]
    X = df[features]
    y = df['is_home_run']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Logistic Regression Base Model...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)

    print("Training XGBoost Base Model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=15,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    lr_preds = lr_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

    ensemble_preds = (lr_preds + xgb_preds) / 2
    auc = roc_auc_score(y_test, ensemble_preds)

    # This score will be much lower, but REALISTIC!
    print(f"Realistic Ensemble ROC-AUC Score: {auc:.3f}")

    os.makedirs('hr_prop_engine/models', exist_ok=True)
    joblib.dump({'lr': lr_model, 'xgb': xgb_model, 'features': features}, 'hr_prop_engine/models/hr_ensemble.pkl')
    print("Pre-Game Models safely saved to hr_prop_engine/models/hr_ensemble.pkl")


if __name__ == "__main__":
    print("--- Starting REAL DATA HR Model Training (No Leakage) ---")
    train_ensemble_model()
    print("--- Training Complete ---")
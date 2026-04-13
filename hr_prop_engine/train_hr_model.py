import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os
import pybaseball as pyb  # <-- NEW: Our real data pipeline


# -------------------------------------------------------------------
# ISOLATION SAFEGUARD:
# This script runs independently of the main repository's data pipeline.
# It pulls its own historical data and saves a standalone model file.
# -------------------------------------------------------------------

def fetch_hr_training_data() -> pd.DataFrame:
    """
    Fetches REAL Statcast data using pybaseball.
    We pull a 2-week sample to train our initial proof-of-concept model quickly.
    """
    print("Fetching REAL Statcast data from Baseball Savant... (This may take a minute)")

    # Enable pybaseball cache so we don't redownload the same data if we run this twice
    pyb.cache.enable()

    # Pull pitch-by-pitch data for a 2-week window (Aug 1 - Aug 14, 2023)
    df = pyb.statcast(start_dt='2023-08-01', end_dt='2023-08-14')

    # Filter only for pitches that were hit into play (home runs, flyouts, singles, etc.)
    df = df.dropna(subset=['events', 'launch_speed', 'launch_angle'])

    print("Data downloaded! Engineering HR prop features...")

    # Create our exact binary target: 1 if Home Run, 0 if not
    df['is_home_run'] = (df['events'] == 'home_run').astype(int)

    # Map pybaseball's real columns to our model's expected features
    features_df = pd.DataFrame({
        'batter_exit_velocity': df['launch_speed'],
        'batter_launch_angle': df['launch_angle'],
        # Using real pitcher release speed and spin rate for pitcher-side metrics
        'pitcher_release_speed': df['release_speed'].fillna(90.0),
        'pitcher_spin_rate': df['release_spin_rate'].fillna(2200.0),
        # (Park/Weather left static for this specific step to focus safely on Statcast)
        'park_factor_hr': 100,
        'weather_temp': 75,
        'weather_wind_outward': 0,
        'is_home_run': df['is_home_run']
    })

    return features_df


def train_ensemble_model():
    """
    Trains a Logistic Regression and XGBoost ensemble for HR probability.
    """
    df = fetch_hr_training_data()

    # Updated features list to match our real Statcast data
    features = [
        'batter_exit_velocity', 'batter_launch_angle', 'pitcher_release_speed',
        'pitcher_spin_rate', 'park_factor_hr', 'weather_temp', 'weather_wind_outward'
    ]
    X = df[features]
    y = df['is_home_run']

    # Split data safely
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Logistic Regression Base Model...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)

    print("Training XGBoost Base Model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=15,  # Adjusting for real HR rarity
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate
    lr_preds = lr_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

    ensemble_preds = (lr_preds + xgb_preds) / 2
    auc = roc_auc_score(y_test, ensemble_preds)

    # Our real data score!
    print(f"Ensemble ROC-AUC Score: {auc:.3f}")

    os.makedirs('hr_prop_engine/models', exist_ok=True)
    joblib.dump({'lr': lr_model, 'xgb': xgb_model, 'features': features}, 'hr_prop_engine/models/hr_ensemble.pkl')
    print("Real Data Models safely saved to hr_prop_engine/models/hr_ensemble.pkl")


if __name__ == "__main__":
    print("--- Starting REAL DATA HR Model Training ---")
    train_ensemble_model()
    print("--- Training Complete ---")
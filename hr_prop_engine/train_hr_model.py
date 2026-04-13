import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os


# -------------------------------------------------------------------
# ISOLATION SAFEGUARD:
# This script runs independently of the main repository's data pipeline.
# It pulls its own historical data and saves a standalone model file
# (.pkl) that the daily predictor will load. It does not overwrite
# any existing model weights or team caches.
# -------------------------------------------------------------------

def fetch_hr_training_data() -> pd.DataFrame:
    """
    Simulates fetching historical Statcast, Pitcher HR/FB, Weather,
    and Park Factor data. In production, connect this to your DB or PyBaseball.
    """
    print("Fetching historical batter-pitcher matchup data...")
    # Simulated dataframe mimicking the research variables
    np.random.seed(42)
    data = pd.DataFrame({
        'batter_barrel_pct': np.random.uniform(0.02, 0.20, 1000),
        'batter_launch_angle': np.random.uniform(5, 25, 1000),
        'pitcher_hr_per_9': np.random.uniform(0.5, 2.5, 1000),
        'park_factor_hr': np.random.uniform(90, 115, 1000),  # 100 is average
        'weather_temp': np.random.uniform(50, 95, 1000),
        'weather_wind_outward': np.random.uniform(-10, 15, 1000),  # negative is blowing in
        'is_home_run': np.random.choice([0, 1], p=[0.95, 0.05], size=1000)  # Highly imbalanced binary target
    })
    return data


def train_ensemble_model():
    """
    Trains a Logistic Regression and XGBoost ensemble for HR probability.
    """
    df = fetch_hr_training_data()

    features = [
        'batter_barrel_pct', 'batter_launch_angle', 'pitcher_hr_per_9',
        'park_factor_hr', 'weather_temp', 'weather_wind_outward'
    ]
    X = df[features]
    y = df['is_home_run']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Logistic Regression Base Model...")
    lr_model = LogisticRegression(class_weight='balanced')
    lr_model.fit(X_train, y_train)

    print("Training XGBoost Base Model...")
    # XGBoost handles non-linear interactions (like high temp + high barrel %)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=10,  # Handle the 5% HR imbalance
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate
    lr_preds = lr_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

    # Simple average ensemble
    ensemble_preds = (lr_preds + xgb_preds) / 2
    auc = roc_auc_score(y_test, ensemble_preds)

    print(f"Ensemble ROC-AUC Score: {auc:.3f}")

    # Ensure output directory exists safely
    os.makedirs('hr_prop_engine/models', exist_ok=True)

    # Save models safely to the isolated sandbox
    joblib.dump({'lr': lr_model, 'xgb': xgb_model, 'features': features}, 'hr_prop_engine/models/hr_ensemble.pkl')
    print("Models safely saved to hr_prop_engine/models/hr_ensemble.pkl")


if __name__ == "__main__":
    print("--- Starting Isolated HR Model Training ---")
    train_ensemble_model()
    print("--- Training Complete ---")
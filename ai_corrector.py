# ai_corrector.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import os


def train_historical_ai():
    print("🧠 Initiating XGBoost Machine Learning Sequence...")

    if not os.path.exists("historical_training_data.csv"):
        print("[!] No historical_training_data.csv found. Run historical_bootstrap.py first.")
        return

    print("   -> Loading 2025 Historical Data...")
    df = pd.read_csv("historical_training_data.csv")

    # Drop any nulls just in case
    df = df.dropna(subset=['Actual_Outcome']).copy()

    print(f"   -> Structuring {len(df)} graded predictions for the Neural Net...")

    # Define the Features the AI will learn from
    features = ['Temp', 'Wind_Speed', 'Lineup_Spot', 'Batter_xwOBA', 'Pitcher_HR9', 'Platoon_Adv', 'Prob']
    categorical_features = ['Batter_Archetype', 'Pitcher_Archetype', 'Market']

    # Convert text categories into binary math (One-Hot Encoding)
    X = pd.get_dummies(df[features + categorical_features], columns=categorical_features)
    y = df['Actual_Outcome']

    # Split data to test the AI's accuracy (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("   -> Training XGBoost Decision Trees...")
    # The AI Model
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Grade the AI's homework
    raw_mc_brier = brier_score_loss(y_test, X_test['Prob'])

    ai_predictions = model.predict_proba(X_test)[:, 1]
    ai_brier = brier_score_loss(y_test, ai_predictions)

    print("\n==================================================")
    print("🤖 XGBOOST AI PERFORMANCE REPORT (2025 SEASON)")
    print("==================================================")
    print(f"Raw Monte Carlo Brier Score : {raw_mc_brier:.4f}")
    print(f"XGBoost AI Brier Score      : {ai_brier:.4f}")

    if ai_brier < raw_mc_brier:
        print("\n✅ THE AI IS SMARTER THAN THE RAW SIMULATOR.")
    else:
        print("\n⏳ The AI needs more tuning. The Raw Simulator is currently more accurate.")

    # Save the brain to the hard drive
    model.save_model("mlb_xgboost_brain.json")

    # Save the exact column structure so our daily bet script knows how to format the data tomorrow
    with open("xgboost_columns.txt", "w") as f:
        f.write(",".join(X.columns))

    print("\n[SUCCESS] AI Brain saved to mlb_xgboost_brain.json")


if __name__ == "__main__":
    train_historical_ai()
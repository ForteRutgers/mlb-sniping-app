# train_nrfi_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
import xgboost as xgb


def train_nrfi():
    print("========================================")
    print("   TRAINING FTTO-Top3 NRFI MODEL        ")
    print("========================================")

    if not os.path.exists("nrfi_training_data.csv"):
        print("[!] Error: nrfi_training_data.csv not found!")
        return

    print(" -> Loading 1st Inning data...")
    df = pd.read_csv("nrfi_training_data.csv").dropna()

    # The exact features our AI is looking for
    feature_cols = [
        "park_factor", "temp", "wind_speed", "wind_out",
        "away_top3_xwoba", "home_top3_xwoba", "away_pitcher_k", "home_pitcher_k"
    ]

    # Inject dummy columns for the player features (the predictor handles these live)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols]
    y = df["yrfi_outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f" -> Training XGBoost Classifier on {len(X_train)} games...")

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss"
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict_proba(X_test)[:, 1]
    binary_preds = model.predict(X_test)

    acc = accuracy_score(y_test, binary_preds)
    bs = brier_score_loss(y_test, preds)

    print(f"\n[MODEL PERFORMANCE]")
    print(f" -> Brier Score: {bs:.4f} (Closer to 0 is better)")
    print(f" -> Accuracy: {acc * 100:.1f}%")

    model.save_model("nrfi_model.json")
    with open("nrfi_model_features.txt", "w") as f:
        f.write(",".join(feature_cols))

    print("\n[SUCCESS] FTTO-Top3 Model saved to nrfi_model.json!")
    print(" -> You are clear to run the Time Machine.")


if __name__ == "__main__":
    train_nrfi()

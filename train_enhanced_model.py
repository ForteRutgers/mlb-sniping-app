# train_enhanced_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("[!] xgboost not installed. Run: pip install xgboost")

try:
    from feature_engineering import FeatureEngineer, LEAGUE_AVG_BATTER, LEAGUE_AVG_PITCHER

    _FE_AVAILABLE = True
except ImportError:
    _FE_AVAILABLE = False

MARKETS = ["HR", "Hit", "TB", "Run", "RBI"]

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=20,
)


def _apply_features_to_row(row: pd.Series, fe: "FeatureEngineer") -> dict:
    context = {
        "temp": float(row.get("Temp", 72)),
        "wind_speed": float(row.get("Wind_Speed", 0)),
        "lineup_spot": int(row.get("Lineup_Spot", 5)),
        "platoon_adv": int(row.get("Platoon_Adv", 0)),
        "park_hr_factor": 1.0,
        "park_avg_factor": 1.0,
        "park_runs_factor": 1.0,
    }
    batter = str(row.get("Batter", row.get("Player", "")))
    pitcher = str(row.get("Pitcher", row.get("Opposing_Pitcher", "")))
    vector = fe.build_feature_vector(batter, pitcher, context)
    vector["Prob"] = float(row.get("Prob", 0.5))
    for arch in ["Slugger", "Contact", "Balanced"]:
        vector[f"Batter_Archetype_{arch}"] = int(str(row.get("Batter_Archetype", "Balanced")) == arch)
        vector[f"Pitcher_Archetype_{arch}"] = int(str(row.get("Pitcher_Archetype", "Balanced")) == arch)
    return vector


def build_enhanced_training_data(fe: "FeatureEngineer") -> pd.DataFrame:
    if not os.path.exists("historical_training_data.csv"):
        raise FileNotFoundError("historical_training_data.csv not found.")
    df = pd.read_csv("historical_training_data.csv")
    df = df.dropna(subset=["Actual_Outcome"]).copy()
    print(f" -> Loaded {len(df):,} rows from historical_training_data.csv")
    vectors = []
    for _, row in df.iterrows():
        v = _apply_features_to_row(row, fe)
        v["Market"] = str(row.get("Market", "HR"))
        v["Actual_Outcome"] = int(row.get("Actual_Outcome", 0))
        vectors.append(v)
    enhanced = pd.DataFrame(vectors)
    enhanced = enhanced.fillna(0)
    return enhanced


def train_market_model(df: pd.DataFrame, market: str, feature_cols: list) -> tuple:
    subset = df[df["Market"] == market].copy()
    if len(subset) < 50:
        print(f" [!] Skipping {market} - only {len(subset)} samples.")
        return None, None, None
    X = subset[feature_cols].fillna(0)
    y = subset["Actual_Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # FIX: The Over-Correction Bug
    # We use max(1.0, ...) to ensure we only scale up rare events, and never shrink
    # the value of common events like Hits.
    num_negatives = (y_train == 0).sum()
    num_positives = (y_train == 1).sum()
    if num_positives > 0:
        weight = max(1.0, num_negatives / num_positives)
    else:
        weight = 1.0

    market_params = XGB_PARAMS.copy()
    market_params["scale_pos_weight"] = weight

    model = xgb.XGBClassifier(**market_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict_proba(X_test)[:, 1]
    bs = brier_score_loss(y_test, preds)
    raw_bs = brier_score_loss(y_test, X_test["Prob"]) if "Prob" in X_test.columns else None
    return model, bs, raw_bs


def run_training():
    print("=" * 60)
    print("   ENHANCED MODEL TRAINING")
    print("=" * 60)
    if not _XGB_AVAILABLE:
        print("[!] xgboost required. Aborting.")
        return
    if not _FE_AVAILABLE:
        print("[!] feature_engineering.py required. Aborting.")
        return
    print("\n[1/3] Loading FeatureEngineer...")
    fe = FeatureEngineer()
    print("[2/3] Building enhanced feature matrix...")
    try:
        enhanced_df = build_enhanced_training_data(fe)
    except FileNotFoundError as exc:
        print(f"[!] {exc}")
        return

    non_feature = {"Market", "Actual_Outcome"}
    feature_cols = [c for c in enhanced_df.columns if c not in non_feature]

    with open("enhanced_model_features.txt", "w") as f:
        f.write(",".join(feature_cols))

    print(f" -> {len(feature_cols)} feature columns.")
    print("[3/3] Training per-market XGBoost models...")
    results = []
    for market in MARKETS:
        print(f" -> {market} ", end="", flush=True)
        model, bs, raw_bs = train_market_model(enhanced_df, market, feature_cols)
        if model is None:
            continue
        model_path = f"enhanced_model_{market.lower()}.json"
        model.save_model(model_path)
        improvement = ""
        if raw_bs is not None:
            delta = raw_bs - bs
            improvement = f" (Delta: {delta:+.4f} vs raw MC)"
        print(f"Brier={bs:.4f}{improvement}")
        results.append(
            {"market": market, "brier_score": round(bs, 4), "raw_mc_brier": round(raw_bs, 4) if raw_bs else None})

    results_df = pd.DataFrame(results)
    results_df.to_csv("enhanced_model_results.csv", index=False)
    print("\n[SUCCESS] Models saved:")
    for market in MARKETS:
        path = f"enhanced_model_{market.lower()}.json"
        if os.path.exists(path):
            print(f"  {path}")
    print("  enhanced_model_features.txt")
    print("  enhanced_model_results.csv")
    print("\nSummary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_training()
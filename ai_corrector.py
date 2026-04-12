# ai_corrector.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import os


def _load_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to merge advanced Statcast metrics into the training DataFrame.
    Falls back silently if the enhanced metrics files are not available.
    """
    # Batter advanced metrics
    if os.path.exists("batter_advanced_metrics_2024_2025.csv"):
        try:
            b_adv = pd.read_csv("batter_advanced_metrics_2024_2025.csv")
            b_adv = b_adv.rename(columns={"player_name": "Batter"})
            adv_cols = [
                "Batter", "barrel_rate", "hard_hit_rate", "sweet_spot_rate",
                "avg_exit_velo", "gb_rate", "ld_rate", "fb_rate",
                "whiff_rate", "chase_rate", "zone_contact_rate",
                "woba_vs_fastball", "woba_vs_breaking", "woba_vs_offspeed",
            ]
            available = [c for c in adv_cols if c in b_adv.columns]
            if len(available) > 1:
                df = df.merge(b_adv[available], on="Batter", how="left")
        except Exception:
            pass

    # Pitcher advanced metrics
    if os.path.exists("pitcher_advanced_metrics_2024_2025.csv"):
        try:
            p_adv = pd.read_csv("pitcher_advanced_metrics_2024_2025.csv")
            p_adv = p_adv.rename(columns={"player_name": "Pitcher"})
            adv_cols = [
                "Pitcher", "whiff_rate", "csw_rate", "k_rate", "bb_rate",
                "hard_hit_against", "barrel_against", "avg_exit_velo_against",
                "fastball_pct", "breaking_pct", "offspeed_pct", "avg_fastball_velo",
            ]
            # Prefix pitcher columns to avoid collisions
            available = [c for c in adv_cols if c in p_adv.columns]
            rename_map = {c: f"p_{c}" for c in available if c != "Pitcher"}
            p_adv_sub = p_adv[available].rename(columns=rename_map)
            p_adv_sub = p_adv_sub.rename(columns={"p_Pitcher": "Pitcher"}) if "p_Pitcher" in p_adv_sub.columns else p_adv_sub
            df = df.merge(p_adv_sub, on="Pitcher", how="left")
        except Exception:
            pass

    return df


def train_historical_ai(use_enhanced_features: bool = True):
    print("🧠 Initiating XGBoost Machine Learning Sequence...")

    if not os.path.exists("historical_training_data.csv"):
        print("[!] No historical_training_data.csv found. Run historical_bootstrap.py first.")
        return

    print("   -> Loading 2025 Historical Data...")
    df = pd.read_csv("historical_training_data.csv")

    # Drop any nulls just in case
    df = df.dropna(subset=['Actual_Outcome']).copy()

    print(f"   -> Structuring {len(df)} graded predictions for the Neural Net...")

    # Optionally merge advanced Statcast metrics
    if use_enhanced_features:
        print("   -> Merging advanced Statcast metrics (if available)...")
        df = _load_enhanced_features(df)

    # Define the core features the AI will learn from
    base_features = ['Temp', 'Wind_Speed', 'Lineup_Spot', 'Batter_xwOBA', 'Pitcher_HR9', 'Platoon_Adv', 'Prob']

    # Advanced Statcast features (included when available)
    advanced_batter_features = [
        'barrel_rate', 'hard_hit_rate', 'sweet_spot_rate', 'avg_exit_velo',
        'gb_rate', 'ld_rate', 'fb_rate', 'whiff_rate', 'chase_rate', 'zone_contact_rate',
        'woba_vs_fastball', 'woba_vs_breaking', 'woba_vs_offspeed',
    ]
    advanced_pitcher_features = [
        'p_whiff_rate', 'p_csw_rate', 'p_k_rate', 'p_bb_rate',
        'p_hard_hit_against', 'p_barrel_against', 'p_avg_exit_velo_against',
        'p_fastball_pct', 'p_breaking_pct', 'p_offspeed_pct', 'p_avg_fastball_velo',
    ]

    # Only include advanced columns that are actually present in the DataFrame
    available_advanced = [
        c for c in advanced_batter_features + advanced_pitcher_features
        if c in df.columns
    ]
    if available_advanced:
        print(f"   -> Including {len(available_advanced)} advanced Statcast features.")

    features = base_features + available_advanced
    categorical_features = ['Batter_Archetype', 'Pitcher_Archetype', 'Market']

    # Convert text categories into binary math (One-Hot Encoding)
    X = pd.get_dummies(df[features + categorical_features], columns=categorical_features)
    X = X.fillna(0)
    y = df['Actual_Outcome']

    # -------------------------------------------------------------------------------------
    # FIX APPLIED HERE: We changed shuffle to False so it learns chronologically!
    # -------------------------------------------------------------------------------------
    # Split data to test the AI's accuracy (Oldest 80% Training, Newest 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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
    if available_advanced:
        print(f"Advanced Statcast features  : {len(available_advanced)} columns merged")

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
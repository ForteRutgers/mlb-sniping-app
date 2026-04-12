# train_totals_model.py
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_totals_model():
    print("========================================")
    print("   TRAINING GAME TOTALS (O/U) MODEL     ")
    print("========================================")

    file_path = "totals_training_data.csv"
    if not os.path.exists(file_path):
        print(f"[!] {file_path} not found. Run collect_totals_data.py first.")
        return

    df = pd.read_csv(file_path)

    # Define the core environmental features we collected
    features = ['park_factor', 'temp', 'wind_speed', 'wind_out']
    target = 'total_runs'

    # Drop any corrupted rows
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    # Split the 4,800+ games: 80% for learning, 20% for testing itself
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f" -> Training XGBoost Regressor on {len(X_train)} games...")

    # Initialize the Regressor (Predicting a number, not a category)
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )

    # Train the brain
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Test the brain against the 20% of games it was hiding from itself
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print("\n[MODEL PERFORMANCE]")
    print(f" -> Mean Absolute Error (MAE): {mae:.2f} runs")
    print("    (This means the AI's baseline run prediction is typically accurate within this margin)")

    # Save the brain to a JSON file
    model_name = "totals_model.json"
    model.save_model(model_name)

    # Save the features it needs to look for
    with open("totals_model_features.txt", "w") as f:
        f.write(",".join(features))

    print(f"\n[SUCCESS] Model saved to {model_name}!")
    print(" -> Your AI Totals Brain is fully trained and ready.")


if __name__ == "__main__":
    train_totals_model()
"""
send_brier_report.py
Reads enhanced_model_results.csv, appends to brier_score_history.csv,
and sends a formatted Brier score performance report to Discord.
"""

import os
import requests
import pandas as pd
from datetime import date


RESULTS_FILE = "enhanced_model_results.csv"
HISTORY_FILE = "brier_score_history.csv"
HISTORY_COLUMNS = ["date", "market", "brier_score", "raw_mc_brier", "improvement_pct"]

MARKET_ORDER = ["HR", "Hit", "TB", "Run", "RBI"]


def load_results() -> pd.DataFrame:
    """Load today's Brier scores from enhanced_model_results.csv."""
    df = pd.read_csv(RESULTS_FILE)
    required = {"market", "brier_score", "raw_mc_brier"}
    if not required.issubset(df.columns):
        raise ValueError(f"{RESULTS_FILE} is missing required columns: {required - set(df.columns)}")
    return df


def load_history() -> pd.DataFrame:
    """Load existing Brier score history, or return an empty DataFrame."""
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def append_today(results: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Append today's results to the history DataFrame."""
    today_str = date.today().isoformat()
    rows = []
    for _, row in results.iterrows():
        market = row["market"]
        brier = float(row["brier_score"])
        raw_mc = float(row["raw_mc_brier"])
        improvement_pct = round((raw_mc - brier) / raw_mc * 100, 2) if raw_mc else 0.0
        rows.append({
            "date": today_str,
            "market": market,
            "brier_score": brier,
            "raw_mc_brier": raw_mc,
            "improvement_pct": improvement_pct,
        })
    new_rows = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return pd.concat([history, new_rows], ignore_index=True)


def build_discord_message(results: pd.DataFrame, history: pd.DataFrame) -> str:
    """Build a formatted Discord message with Brier score results and trends."""
    today_str = date.today().isoformat()

    # Ordered results for display
    ordered = []
    for market in MARKET_ORDER:
        match = results[results["market"] == market]
        if not match.empty:
            ordered.append(match.iloc[0])
    # Append any markets not in MARKET_ORDER
    for _, row in results.iterrows():
        if row["market"] not in MARKET_ORDER:
            ordered.append(row)

    # Count improvements
    improving = 0
    total = len(ordered)

    header = "📊 **MLB Model Performance Report** 📊\n\n"
    header += f"📅 **Date**: {today_str}\n\n"
    header += "🎯 **Brier Scores** (Lower = Better)\n"
    header += "```\n"
    header += f"{'Market':<8} {'XGBoost':>8} {'Raw MC':>8} {'Improve':>10}\n"
    header += "-" * 38 + "\n"

    rows_text = ""
    for row in ordered:
        market = row["market"]
        brier = float(row["brier_score"])
        raw_mc = float(row["raw_mc_brier"])
        improvement_pct = (raw_mc - brier) / raw_mc * 100 if raw_mc else 0.0
        if improvement_pct > 0:
            trend = f"+{improvement_pct:.1f}% ✅"
            improving += 1
        else:
            trend = f"{improvement_pct:.1f}% ❌"
        rows_text += f"{market:<8} {brier:>8.4f} {raw_mc:>8.4f} {trend:>10}\n"

    header += rows_text
    header += "```\n"

    # Trend summary
    if total > 0:
        header += f"\n📈 **Trend**: Model improving on {improving}/{total} markets\n"
    if improving == total:
        header += "🏆 **Status**: XGBoost outperforming raw Monte Carlo on all markets"
    elif improving > total // 2:
        header += "✅ **Status**: XGBoost outperforming raw Monte Carlo on most markets"
    else:
        header += "⚠️ **Status**: Model needs improvement — more training data recommended"

    # Historical trend (last 7 days vs today)
    if not history.empty and "date" in history.columns:
        recent = history[history["date"] < today_str].tail(5 * len(MARKET_ORDER))
        if not recent.empty:
            avg_prev = recent["brier_score"].mean()
            avg_today = results["brier_score"].mean()
            delta = avg_prev - avg_today
            if delta > 0:
                header += f"\n📊 **vs Previous**: Avg Brier improved by {delta:.4f} ({delta / avg_prev * 100:.1f}%) since last report"
            elif delta < 0:
                header += f"\n📊 **vs Previous**: Avg Brier worsened by {abs(delta):.4f} ({abs(delta) / avg_prev * 100:.1f}%) since last report"
            else:
                header += "\n📊 **vs Previous**: No change since last report"

    return header


def send_to_discord(message: str, webhook_url: str) -> None:
    """Send a message to Discord via webhook."""
    payload = {"content": message}
    response = requests.post(webhook_url, json=payload, timeout=10)
    if response.status_code in (200, 204):
        print("[SUCCESS] Brier score report sent to Discord.")
    else:
        print(f"[ERROR] Discord returned status {response.status_code}: {response.text}")
        response.raise_for_status()


def main():
    webhook_url = os.environ.get("DISCORD_WEBHOOK", "").strip()

    if not os.path.exists(RESULTS_FILE):
        print(f"[INFO] {RESULTS_FILE} not found — skipping Brier report.")
        return

    print(f"[INFO] Loading results from {RESULTS_FILE}...")
    results = load_results()

    print(f"[INFO] Loading history from {HISTORY_FILE}...")
    history = load_history()

    print("[INFO] Appending today's scores to history...")
    updated_history = append_today(results, history)
    updated_history.to_csv(HISTORY_FILE, index=False)
    print(f"[INFO] History saved to {HISTORY_FILE} ({len(updated_history)} rows total).")

    message = build_discord_message(results, history)
    print("\n--- Discord Message Preview ---")
    print(message)
    print("-------------------------------\n")

    if not webhook_url:
        print("[INFO] DISCORD_WEBHOOK not set — message not sent.")
        return

    send_to_discord(message, webhook_url)


if __name__ == "__main__":
    main()

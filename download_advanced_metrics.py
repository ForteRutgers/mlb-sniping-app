# download_advanced_metrics.py
import pybaseball
import pandas as pd


def fix_name(name_string):
    """Takes 'Last, First' and flips it to 'First Last'."""
    if pd.isna(name_string):
        return name_string
    parts = str(name_string).split(', ')
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return name_string


def download_stats():
    print("⚾ Connecting to the official Statcast database...")
    print("Fetching Advanced Batter Data for 2025 (Baseball Savant)...")

    try:
        # 1. Download Batter Stats
        batters = pybaseball.statcast_batter_exitvelo_barrels(2025)

        # Check how the database gave us the names and fix them
        if 'last_name, first_name' in batters.columns:
            batters['player_name'] = batters['last_name, first_name'].apply(fix_name)
        elif 'first_name' in batters.columns and 'last_name' in batters.columns:
            batters['player_name'] = batters['first_name'] + ' ' + batters['last_name']
        else:
            print("⚠️ Warning: Could not find player names in the batter data!")

        # Translate the Statcast columns to match our AI's language
        if 'brl_percent' in batters.columns:
            batters['barrel_rate'] = batters['brl_percent']
        if 'ev95percent' in batters.columns:
            batters['hard_hit_rate'] = batters['ev95percent']
        if 'avg_hit_speed' in batters.columns:
            batters['avg_exit_velo'] = batters['avg_hit_speed']
        if 'anglesweetspotpercent' in batters.columns:
            batters['sweet_spot_rate'] = batters['anglesweetspotpercent']

        batters.to_csv("batter_advanced_metrics_2024_2025.csv", index=False)
        print("✅ Saved batter_advanced_metrics_2024_2025.csv successfully!")

        print("\nFetching Advanced Pitcher Data for 2025 (Baseball Savant)...")

        # 2. Download Pitcher Stats
        pitchers = pybaseball.statcast_pitcher_exitvelo_barrels(2025)

        # Check how the database gave us the names and fix them
        if 'last_name, first_name' in pitchers.columns:
            pitchers['player_name'] = pitchers['last_name, first_name'].apply(fix_name)
        elif 'first_name' in pitchers.columns and 'last_name' in pitchers.columns:
            pitchers['player_name'] = pitchers['first_name'] + ' ' + pitchers['last_name']
        else:
            print("⚠️ Warning: Could not find player names in the pitcher data!")

        # Translate Pitcher columns
        if 'brl_percent' in pitchers.columns:
            pitchers['barrel_against'] = pitchers['brl_percent']
        if 'ev95percent' in pitchers.columns:
            pitchers['hard_hit_against'] = pitchers['ev95percent']
        if 'avg_hit_speed' in pitchers.columns:
            pitchers['avg_exit_velo_against'] = pitchers['avg_hit_speed']

        pitchers.to_csv("pitcher_advanced_metrics_2024_2025.csv", index=False)
        print("✅ Saved pitcher_advanced_metrics_2024_2025.csv successfully!")

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    download_stats()
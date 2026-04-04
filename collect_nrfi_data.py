# collect_nrfi_data.py
"""
Collect historical first-inning data and enrich it with advanced features
for NRFI model training.

Usage:
    python collect_nrfi_data.py
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from typing import Dict, List, Any

try:
    from feature_engineering import FeatureEngineer, LEAGUE_AVG_BATTER, LEAGUE_AVG_PITCHER
    _FE_AVAILABLE = True
except ImportError:
    _FE_AVAILABLE = False

from game_markets_predictor import _get_park


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = 10, retries: int = 3) -> Dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return {}


def _get_schedule(date_str: str) -> List[Dict]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    data = _fetch_json(url)
    games = []
    for date_obj in data.get("dates", []):
        games.extend(date_obj.get("games", []))
    return games


def _get_linescore(game_pk: int) -> Dict:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
    return _fetch_json(url)


def _get_boxscore(game_pk: int) -> Dict:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    return _fetch_json(url)


# ---------------------------------------------------------------------------
# First-inning data collection
# ---------------------------------------------------------------------------

def collect_first_inning_data(
    start_year: int = 2023, end_year: int = 2024
) -> pd.DataFrame:
    """
    Fetch first-inning box-score data for *start_year* through *end_year*.
    Saves raw results to ``nrfi_raw_data.csv`` and returns the DataFrame.
    """
    print(f"[1/2] Collecting first-inning data ({start_year}–{end_year})…")
    records = []

    for year in range(start_year, end_year + 1):
        # Season spans roughly April through October
        dates = pd.date_range(f"{year}-04-01", f"{year}-10-15", freq="D")
        print(f"   -> {year}: scanning {len(dates)} dates…")

        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            games = _get_schedule(date_str)
            if not games:
                continue
            time.sleep(0.15)

            for game in games:
                status = game.get("status", {}).get("statusCode", "")
                if status not in ["F", "O"]:
                    continue

                game_pk = game["gamePk"]
                venue = game.get("venue", {}).get("name", "Unknown")
                home_team = game["teams"]["home"]["team"]["name"]
                away_team = game["teams"]["away"]["team"]["name"]

                linescore = _get_linescore(game_pk)
                innings = linescore.get("innings", [])
                if not innings:
                    continue
                first = innings[0]

                away_runs_1st = first.get("away", {}).get("runs", None)
                home_runs_1st = first.get("home", {}).get("runs", None)
                if away_runs_1st is None or home_runs_1st is None:
                    continue

                # Starting pitchers
                boxscore = _get_boxscore(game_pk)
                teams_box = boxscore.get("teams", {})
                away_pitchers = teams_box.get("away", {}).get("pitchers", [])
                home_pitchers = teams_box.get("home", {}).get("pitchers", [])
                away_players = teams_box.get("away", {}).get("players", {})
                home_players = teams_box.get("home", {}).get("players", {})

                def _pitcher_name(pitcher_list, players_dict):
                    if not pitcher_list:
                        return "TBD"
                    pid = f"ID{pitcher_list[0]}"
                    return players_dict.get(pid, {}).get("person", {}).get("fullName", "TBD")

                away_sp = _pitcher_name(away_pitchers, away_players)
                home_sp = _pitcher_name(home_pitchers, home_players)

                nrfi = int(away_runs_1st == 0 and home_runs_1st == 0)

                records.append({
                    "game_date": date_str,
                    "game_pk": game_pk,
                    "venue": venue,
                    "home_team": home_team,
                    "away_team": away_team,
                    "away_sp": away_sp,
                    "home_sp": home_sp,
                    "away_runs_1st": int(away_runs_1st),
                    "home_runs_1st": int(home_runs_1st),
                    "nrfi_outcome": nrfi,
                })

            if len(records) % 500 == 0 and records:
                pd.DataFrame(records).to_csv("nrfi_raw_data.csv", index=False)

    df = pd.DataFrame(records)
    df.to_csv("nrfi_raw_data.csv", index=False)
    nrfi_rate = df["nrfi_outcome"].mean() if len(df) else 0.0
    print(f"   -> Collected {len(df):,} games.  NRFI rate: {nrfi_rate:.3f}")
    return df


# ---------------------------------------------------------------------------
# Feature enrichment
# ---------------------------------------------------------------------------

def enrich_nrfi_data_with_features() -> pd.DataFrame:
    """
    Enrich ``nrfi_raw_data.csv`` with pitcher and lineup advanced features.
    Saves to ``nrfi_training_data.csv``.
    """
    print("[2/2] Enriching NRFI data with advanced features…")

    if not os.path.exists("nrfi_raw_data.csv"):
        print("[!] nrfi_raw_data.csv not found.  Run collect_first_inning_data() first.")
        return pd.DataFrame()

    df = pd.read_csv("nrfi_raw_data.csv")
    if df.empty:
        return df

    if not _FE_AVAILABLE:
        print("[!] feature_engineering.py not available.  Saving raw data as training data.")
        df.to_csv("nrfi_training_data.csv", index=False)
        return df

    fe = FeatureEngineer()

    rows = []
    for _, row in df.iterrows():
        away_sp = str(row.get("away_sp", ""))
        home_sp = str(row.get("home_sp", ""))
        venue = str(row.get("venue", "Unknown"))
        park = _get_park(venue)

        # Pitcher features
        apf = fe.get_pitcher_features(away_sp)
        hpf = fe.get_pitcher_features(home_sp)

        # We don't have historical lineup data here — use league averages as proxy
        la_b = LEAGUE_AVG_BATTER

        enriched = {
            "game_date": row.get("game_date"),
            "venue": venue,
            "away_team": row.get("away_team"),
            "home_team": row.get("home_team"),
            "away_sp": away_sp,
            "home_sp": home_sp,
            # Away pitcher features
            "away_whiff_rate": apf.get("whiff_rate", 0.248),
            "away_k_rate": apf.get("k_rate", 0.220),
            "away_bb_rate": apf.get("bb_rate", 0.080),
            "away_hard_hit_against": apf.get("hard_hit_against", 0.380),
            "away_gb_rate": apf.get("gb_rate", 0.435),
            "away_stuff_plus": apf.get("stuff_plus", 100.0),
            # Home pitcher features
            "home_whiff_rate": hpf.get("whiff_rate", 0.248),
            "home_k_rate": hpf.get("k_rate", 0.220),
            "home_bb_rate": hpf.get("bb_rate", 0.080),
            "home_hard_hit_against": hpf.get("hard_hit_against", 0.380),
            "home_gb_rate": hpf.get("gb_rate", 0.435),
            "home_stuff_plus": hpf.get("stuff_plus", 100.0),
            # Lineup proxy features (league averages — improved if real lineups are available)
            "away_top4_xwoba": la_b["xwOBA"],
            "away_leadoff_xwoba": la_b["xwOBA"],
            "away_top4_hard_hit": la_b["hard_hit_rate"],
            "away_top4_barrel": la_b["Barrel_Rate"],
            "home_top4_xwoba": la_b["xwOBA"],
            "home_leadoff_xwoba": la_b["xwOBA"],
            "home_top4_hard_hit": la_b["hard_hit_rate"],
            "home_top4_barrel": la_b["Barrel_Rate"],
            # Park & weather context
            "park_runs_factor": park[2],
            "temp": 72,  # historical weather unavailable without external API
            "wind_out": 0,
            # Target
            "nrfi_outcome": int(row.get("nrfi_outcome", 0)),
        }
        rows.append(enriched)

    out = pd.DataFrame(rows)
    out.to_csv("nrfi_training_data.csv", index=False)
    print(f"   -> Saved {len(out):,} enriched records to nrfi_training_data.csv")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = collect_first_inning_data(start_year=2023, end_year=2024)
    enrich_nrfi_data_with_features()

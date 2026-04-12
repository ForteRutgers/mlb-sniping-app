# enhanced_historical_bootstrap.py
"""
Advanced Statcast data pipeline.
Pulls complete 2024-2025 pitch-by-pitch data and calculates advanced batter/pitcher metrics.

Usage:
    python enhanced_historical_bootstrap.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, date

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_divide(num, denom, default=0.0):
    """Divide two Series/scalars, returning *default* when denom is zero."""
    if isinstance(denom, pd.Series):
        return np.where(denom > 0, num / denom.replace(0, np.nan), default)
    return num / denom if denom else default


def _build_name(row):
    first = str(row.get("player_name", "")).split(",")
    if len(first) == 2:
        return f"{first[1].strip()} {first[0].strip()}"
    return str(row.get("player_name", "Unknown"))


def _build_name_vectorized(series: pd.Series) -> pd.Series:
    """Vectorized version of _build_name: converts 'Last, First' → 'First Last'."""
    s = series.fillna("Unknown").astype(str)
    split = s.str.split(",", n=1, expand=True)
    if split.shape[1] >= 2:
        has_comma = split.iloc[:, 1].notna()
        result = split.iloc[:, 1].str.strip() + " " + split.iloc[:, 0].str.strip()
        return pd.Series(np.where(has_comma, result, s), index=series.index)
    return s


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_statcast_data(start_year: int = 2024, end_year: int = 2025) -> pd.DataFrame:
    """
    Pull pitch-by-pitch Statcast data for *start_year* through *end_year*.
    Falls back to an empty DataFrame with the expected columns when pybaseball
    is unavailable or the API is rate-limited.
    """
    print(f"[1/6] Fetching Statcast data for {start_year}-{end_year}...")

    raw_csv = "statcast_raw_2024_2025.csv"
    if os.path.exists(raw_csv):
        print(f"      -> Loading existing {raw_csv} (skipping download)")
        return pd.read_csv(raw_csv, low_memory=False)

    frames = []
    try:
        import pybaseball as _pyb  # type: ignore
        _pyb.cache.enable()
        from pybaseball import statcast  # type: ignore
        for year in range(start_year, end_year + 1):
            start_dt = f"{year}-03-01"
            end_dt = f"{year}-11-30"
            print(f"      -> Downloading {year} season…")
            chunk = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
            if chunk is not None and not chunk.empty:
                frames.append(chunk)
                print(f"         {len(chunk):,} pitches received.")
    except Exception as exc:
        print(f"[!] Statcast fetch failed: {exc}")

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_csv("statcast_raw_2024_2025.csv", index=False)
        print(f"      -> Saved {len(df):,} rows to statcast_raw_2024_2025.csv")
        return df

    print("[!] No data fetched.  Creating empty placeholder files.")
    _create_placeholder_files()
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Batter Metrics
# ---------------------------------------------------------------------------

def calculate_batter_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive advanced batter metrics from raw Statcast data.

    Returns one row per batter with all advanced metrics.
    """
    print("[2/6] Calculating batter advanced metrics…")
    if df.empty:
        return _empty_batter_df()

    # Group by player name (Statcast uses "player_name" for batter)
    if "player_name" not in df.columns:
        print("[!] player_name column not found – skipping batter metrics.")
        return _empty_batter_df()

    # Build full name from Statcast "Last, First" format – computed ONCE using
    # vectorized string ops BEFORE any filtering so all subsets inherit the column.
    df = df.copy()
    df["batter_full_name"] = _build_name_vectorized(df["player_name"])

    batted = df[df["launch_speed"].notna() & df["launch_angle"].notna()].copy()

    batted["is_barrel"] = (
        (batted["launch_speed"] >= 98) &
        (batted["launch_angle"] >= 26) & (batted["launch_angle"] <= 30)
    ).astype(int)
    batted["is_hard_hit"] = (batted["launch_speed"] >= 95).astype(int)
    batted["is_sweet_spot"] = (
        (batted["launch_angle"] >= 8) & (batted["launch_angle"] <= 32)
    ).astype(int)

    # Batted-ball types using launch angle bins
    batted["is_gb"] = (batted["launch_angle"] < 10).astype(int)
    batted["is_ld"] = (
        (batted["launch_angle"] >= 10) & (batted["launch_angle"] <= 25)
    ).astype(int)
    batted["is_fb"] = (batted["launch_angle"] > 25).astype(int)

    # Swings and whiffs – all inherit batter_full_name from df
    swings = df[df["description"].isin(
        ["hit_into_play", "swinging_strike", "foul", "swinging_strike_blocked", "foul_tip"]
    )]
    whiffs = swings[swings["description"].isin(
        ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
    )]

    # Chase rate: swings at pitches outside zone (zone 11-14 are ball zones in Statcast)
    out_zone = df[df["zone"].isin([11, 12, 13, 14])]
    out_zone_swings = out_zone[out_zone["description"].isin(
        ["hit_into_play", "swinging_strike", "foul", "swinging_strike_blocked", "foul_tip"]
    )]

    # Zone contact (zone 1-9 are strike zones)
    in_zone = df[df["zone"].between(1, 9)]
    in_zone_contact = in_zone[in_zone["description"].isin(
        ["hit_into_play", "foul", "foul_tip"]
    )]
    in_zone_swings = in_zone[in_zone["description"].isin(
        ["hit_into_play", "swinging_strike", "foul", "swinging_strike_blocked", "foul_tip"]
    )]

    # wOBA vs pitch type  (use estimated_woba_using_speedangle where available)
    woba_col = "estimated_woba_using_speedangle" if "estimated_woba_using_speedangle" in df.columns else None

    # Aggregate counts per batter – groupby directly; no redundant apply() calls
    bb_agg = batted.groupby("batter_full_name").agg(
        total_batted=("is_barrel", "count"),
        barrels=("is_barrel", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        sweet_spots=("is_sweet_spot", "sum"),
        avg_ev=("launch_speed", "mean"),
        max_ev=("launch_speed", "max"),
        gb=("is_gb", "sum"),
        ld=("is_ld", "sum"),
        fb=("is_fb", "sum"),
    )

    sw_agg = swings.groupby("batter_full_name").size().rename("total_swings")
    wh_agg = whiffs.groupby("batter_full_name").size().rename("total_whiffs")
    oz_agg = out_zone.groupby("batter_full_name").size().rename("oz_pitches")
    ozs_agg = out_zone_swings.groupby("batter_full_name").size().rename("oz_swings")
    iz_s_agg = in_zone_swings.groupby("batter_full_name").size().rename("iz_swings")
    iz_c_agg = in_zone_contact.groupby("batter_full_name").size().rename("iz_contacts")

    combined = bb_agg.join(sw_agg, how="left").join(wh_agg, how="left") \
                     .join(oz_agg, how="left").join(ozs_agg, how="left") \
                     .join(iz_s_agg, how="left").join(iz_c_agg, how="left")
    combined = combined.fillna(0)

    combined["barrel_rate"] = _safe_divide(combined["barrels"], combined["total_batted"])
    combined["hard_hit_rate"] = _safe_divide(combined["hard_hits"], combined["total_batted"])
    combined["sweet_spot_rate"] = _safe_divide(combined["sweet_spots"], combined["total_batted"])
    combined["avg_exit_velo"] = combined["avg_ev"]
    combined["max_exit_velo"] = combined["max_ev"]
    combined["gb_rate"] = _safe_divide(combined["gb"], combined["total_batted"])
    combined["ld_rate"] = _safe_divide(combined["ld"], combined["total_batted"])
    combined["fb_rate"] = _safe_divide(combined["fb"], combined["total_batted"])
    combined["whiff_rate"] = _safe_divide(combined["total_whiffs"], combined["total_swings"])
    combined["chase_rate"] = _safe_divide(combined["oz_swings"], combined["oz_pitches"])
    combined["zone_contact_rate"] = _safe_divide(combined["iz_contacts"], combined["iz_swings"])

    # wOBA vs pitch type – df already has batter_full_name, inherited by df_pt
    if woba_col and "pitch_type" in df.columns:
        pt_map = {"FF": "fastball", "SI": "fastball", "FC": "fastball",
                  "SL": "breaking", "CU": "breaking", "KC": "breaking", "CS": "breaking",
                  "CH": "offspeed", "FS": "offspeed", "FO": "offspeed"}
        df_pt = df[df[woba_col].notna()].copy()
        df_pt["pitch_cat"] = df_pt["pitch_type"].map(pt_map)
        for cat in ["fastball", "breaking", "offspeed"]:
            subset = df_pt[df_pt["pitch_cat"] == cat]
            woba_avg = subset.groupby("batter_full_name")[woba_col].mean().rename(f"woba_vs_{cat}")
            combined = combined.join(woba_avg, how="left")
        for col in ["woba_vs_fastball", "woba_vs_breaking", "woba_vs_offspeed"]:
            if col not in combined.columns:
                combined[col] = 0.320
            combined[col] = combined[col].fillna(0.320)
    else:
        combined["woba_vs_fastball"] = 0.320
        combined["woba_vs_breaking"] = 0.320
        combined["woba_vs_offspeed"] = 0.320

    combined = combined.reset_index().rename(columns={"batter_full_name": "player_name"})
    out_path = "batter_advanced_metrics_2024_2025.csv"
    combined.to_csv(out_path, index=False)
    print(f"      -> Saved {len(combined):,} batter records to {out_path}")
    return combined


# ---------------------------------------------------------------------------
# Pitcher Metrics
# ---------------------------------------------------------------------------

def calculate_pitcher_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive advanced pitcher metrics from raw Statcast data.
    """
    print("[3/6] Calculating pitcher advanced metrics…")
    if df.empty:
        return _empty_pitcher_df()

    df = df.copy()
    # Statcast pitcher name is stored in "player_name" for the pitcher perspective
    # when queried via statcast(); use pitcher column (MLBAM id) + a name lookup.
    # Build pitcher full name from "pitcher_name" or reconstruct if needed.
    if "pitcher_name" in df.columns:
        df["pitcher_full_name"] = df["pitcher_name"]
    else:
        # player_name in statcast() is the pitcher's name
        df["pitcher_full_name"] = _build_name_vectorized(df["player_name"])

    batted = df[df["launch_speed"].notna() & df["launch_angle"].notna()].copy()
    batted["is_barrel"] = (
        (batted["launch_speed"] >= 98) &
        (batted["launch_angle"] >= 26) & (batted["launch_angle"] <= 30)
    ).astype(int)
    batted["is_hard_hit"] = (batted["launch_speed"] >= 95).astype(int)
    batted["is_gb"] = (batted["launch_angle"] < 10).astype(int)
    batted["is_fb"] = (batted["launch_angle"] > 25).astype(int)

    swings = df[df["description"].isin(
        ["hit_into_play", "swinging_strike", "foul", "swinging_strike_blocked", "foul_tip"]
    )].copy()
    whiffs = swings[swings["description"].isin(
        ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
    )]
    called_strikes = df[df["description"] == "called_strike"]

    # Pitch type breakdown
    pt_map = {"FF": "fastball", "SI": "fastball", "FC": "fastball",
              "SL": "breaking", "CU": "breaking", "KC": "breaking", "CS": "breaking",
              "CH": "offspeed", "FS": "offspeed", "FO": "offspeed"}

    # Aggregate
    bb_agg = batted.groupby("pitcher_full_name").agg(
        total_batted=("is_barrel", "count"),
        barrels_allowed=("is_barrel", "sum"),
        hard_hits_allowed=("is_hard_hit", "sum"),
        avg_ev_against=("launch_speed", "mean"),
        gb=("is_gb", "sum"),
        fb=("is_fb", "sum"),
    )
    sw_agg = swings.groupby("pitcher_full_name").size().rename("total_swings")
    wh_agg = whiffs.groupby("pitcher_full_name").size().rename("total_whiffs")
    cs_agg = called_strikes.groupby("pitcher_full_name").size().rename("called_strikes")

    # Total batters faced
    total_bf = df.groupby("pitcher_full_name").size().rename("total_bf")

    # K and BB
    k_events = df[df["events"].isin(["strikeout", "strikeout_double_play"])]
    bb_events = df[df["events"].isin(["walk", "intent_walk"])]
    hr_events = df[df["events"] == "home_run"]
    k_agg = k_events.groupby("pitcher_full_name").size().rename("strikeouts")
    bb_agg2 = bb_events.groupby("pitcher_full_name").size().rename("walks")
    hr_agg = hr_events.groupby("pitcher_full_name").size().rename("hrs_allowed")

    combined = bb_agg.join(sw_agg, how="left").join(wh_agg, how="left") \
                     .join(cs_agg, how="left").join(total_bf, how="left") \
                     .join(k_agg, how="left").join(bb_agg2, how="left").join(hr_agg, how="left")
    combined = combined.fillna(0)

    combined["whiff_rate"] = _safe_divide(combined["total_whiffs"], combined["total_swings"])
    combined["csw_rate"] = _safe_divide(
        combined["total_whiffs"] + combined["called_strikes"], combined["total_bf"]
    )
    combined["k_rate"] = _safe_divide(combined["strikeouts"], combined["total_bf"])
    combined["bb_rate"] = _safe_divide(combined["walks"], combined["total_bf"])
    combined["hard_hit_against"] = _safe_divide(combined["hard_hits_allowed"], combined["total_batted"])
    combined["barrel_against"] = _safe_divide(combined["barrels_allowed"], combined["total_batted"])
    combined["avg_exit_velo_against"] = combined["avg_ev_against"].fillna(88.5)
    combined["gb_rate"] = _safe_divide(combined["gb"], combined["total_batted"])
    combined["fb_rate"] = _safe_divide(combined["fb"], combined["total_batted"])
    combined["hr_rate_allowed"] = _safe_divide(combined["hrs_allowed"], combined["total_bf"])

    # Pitch arsenal
    if "pitch_type" in df.columns:
        df_pt = df.copy()
        df_pt["pitch_cat"] = df_pt["pitch_type"].map(pt_map)
        for cat, col in [("fastball", "fastball_pct"), ("breaking", "breaking_pct"), ("offspeed", "offspeed_pct")]:
            sub = df_pt[df_pt["pitch_cat"] == cat].groupby("pitcher_full_name").size().rename(col)
            combined = combined.join(sub, how="left")
        for col in ["fastball_pct", "breaking_pct", "offspeed_pct"]:
            if col not in combined.columns:
                combined[col] = 0
            combined[col] = combined[col].fillna(0)
        total_typed = combined["fastball_pct"] + combined["breaking_pct"] + combined["offspeed_pct"]
        for col in ["fastball_pct", "breaking_pct", "offspeed_pct"]:
            combined[col] = np.where(total_typed > 0, combined[col] / total_typed, 1 / 3)

        # Fastball velocity
        fb_pitches = df_pt[df_pt["pitch_cat"] == "fastball"]
        avg_fb_velo = fb_pitches.groupby("pitcher_full_name")["release_speed"].mean().rename("avg_fastball_velo")
        combined = combined.join(avg_fb_velo, how="left")
        combined["avg_fastball_velo"] = combined["avg_fastball_velo"].fillna(93.0)
    else:
        combined["fastball_pct"] = 0.50
        combined["breaking_pct"] = 0.30
        combined["offspeed_pct"] = 0.20
        combined["avg_fastball_velo"] = 93.0

    combined = combined.reset_index().rename(columns={"pitcher_full_name": "player_name"})
    out_path = "pitcher_advanced_metrics_2024_2025.csv"
    combined.to_csv(out_path, index=False)
    print(f"      -> Saved {len(combined):,} pitcher records to {out_path}")
    return combined


# ---------------------------------------------------------------------------
# Recent Form (Rolling 14-day)
# ---------------------------------------------------------------------------

def calculate_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling 14-day batter performance metrics.
    """
    print("[4/6] Calculating rolling 14-day form data…")
    if df.empty or "game_date" not in df.columns:
        return _empty_recent_form_df()

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["batter_full_name"] = _build_name_vectorized(df["player_name"])

    cutoff = df["game_date"].max() - pd.Timedelta(days=14)
    recent = df[df["game_date"] >= cutoff].copy()
    if recent.empty:
        return _empty_recent_form_df()

    batted = recent[recent["launch_speed"].notna()].copy()
    batted["is_hard_hit"] = (batted["launch_speed"] >= 95).astype(int)
    batted["is_barrel"] = (
        (batted["launch_speed"] >= 98) &
        (batted["launch_angle"] >= 26) & (batted["launch_angle"] <= 30)
    ).astype(int)

    bb_agg = batted.groupby("batter_full_name").agg(
        recent_avg_ev=("launch_speed", "mean"),
        recent_hard_hit=("is_hard_hit", "mean"),
        recent_barrel_rate=("is_barrel", "mean"),
        recent_batted=("is_barrel", "count"),
    )

    total_pa = recent.groupby("batter_full_name").size().rename("recent_pa")
    hits = recent[recent["events"].isin(["single", "double", "triple", "home_run"])]
    hit_agg = hits.groupby("batter_full_name").size().rename("recent_hits")
    hr_agg = recent[recent["events"] == "home_run"].groupby("batter_full_name").size().rename("recent_hrs")

    combined = bb_agg.join(total_pa, how="left").join(hit_agg, how="left").join(hr_agg, how="left").fillna(0)
    combined["recent_avg"] = _safe_divide(combined["recent_hits"], combined["recent_pa"])
    combined["recent_hr_rate"] = _safe_divide(combined["recent_hrs"], combined["recent_pa"])

    combined = combined.reset_index().rename(columns={"batter_full_name": "player_name"})
    out_path = "recent_form_data.csv"
    combined.to_csv(out_path, index=False)
    print(f"      -> Saved {len(combined):,} recent-form records to {out_path}")
    return combined


# ---------------------------------------------------------------------------
# Expected Stats (xBA, xSLG, xwOBA)
# ---------------------------------------------------------------------------

def calculate_expected_stats(df: pd.DataFrame) -> tuple:
    """
    Extract or approximate xBA, xSLG, xwOBA and luck factors.
    Returns (batter_df, pitcher_df).
    """
    print("[5/6] Calculating expected stats (xBA, xSLG, xwOBA)…")
    if df.empty:
        return _empty_expected_batter_df(), _empty_expected_pitcher_df()

    df = df.copy()
    df["batter_full_name"] = _build_name_vectorized(df["player_name"])
    if "pitcher_name" in df.columns:
        df["pitcher_full_name"] = df["pitcher_name"]
    else:
        df["pitcher_full_name"] = _build_name_vectorized(df["player_name"])

    xwoba_col = "estimated_woba_using_speedangle" if "estimated_woba_using_speedangle" in df.columns else None
    xba_col = "estimated_ba_using_speedangle" if "estimated_ba_using_speedangle" in df.columns else None

    # ---------- Batter expected stats ----------
    if xwoba_col and xba_col:
        b_exp = df.groupby("batter_full_name").agg(
            xwoba=(xwoba_col, "mean"),
            xba=(xba_col, "mean"),
        ).reset_index()
    else:
        # Approximate using hard-hit rate
        batted = df[df["launch_speed"].notna()].copy()
        batted["is_hard_hit"] = (batted["launch_speed"] >= 95).astype(int)
        hh = batted.groupby("batter_full_name")["is_hard_hit"].mean().reset_index()
        hh.columns = ["batter_full_name", "hard_hit_rate"]
        hh["xwoba"] = 0.250 + hh["hard_hit_rate"] * 0.40
        hh["xba"] = 0.200 + hh["hard_hit_rate"] * 0.20
        b_exp = hh[["batter_full_name", "xwoba", "xba"]]

    # Add xSLG approximation
    batted2 = df[df["launch_speed"].notna()].copy()
    batted2["tb_estimate"] = np.where(
        batted2["events"] == "home_run", 4,
        np.where(batted2["events"] == "triple", 3,
        np.where(batted2["events"] == "double", 2,
        np.where(batted2["events"] == "single", 1, 0)))
    )
    pa_count = df.groupby("batter_full_name").size().rename("pa")
    tb_sum = batted2.groupby("batter_full_name")["tb_estimate"].sum().rename("total_tb")
    b_slg = (tb_sum / pa_count).rename("xslg").reset_index()

    b_exp = b_exp.rename(columns={"batter_full_name": "player_name"}) if "batter_full_name" in b_exp.columns else b_exp
    if "player_name" not in b_exp.columns and len(b_exp.columns) > 0:
        b_exp = b_exp.rename(columns={b_exp.columns[0]: "player_name"})

    b_exp.to_csv("batter_expected_stats.csv", index=False)
    print(f"      -> Saved {len(b_exp):,} batter expected-stats rows.")

    # ---------- Pitcher expected stats ----------
    if xwoba_col:
        p_exp = df.groupby("pitcher_full_name").agg(
            xwoba_against=(xwoba_col, "mean"),
        ).reset_index().rename(columns={"pitcher_full_name": "player_name"})
    else:
        p_exp = pd.DataFrame(columns=["player_name", "xwoba_against"])

    p_exp.to_csv("pitcher_expected_stats.csv", index=False)
    print(f"      -> Saved {len(p_exp):,} pitcher expected-stats rows.")
    return b_exp, p_exp


# ---------------------------------------------------------------------------
# Placeholder helpers (empty DataFrames with the right columns)
# ---------------------------------------------------------------------------

def _empty_batter_df():
    cols = [
        "player_name", "barrel_rate", "hard_hit_rate", "sweet_spot_rate",
        "avg_exit_velo", "max_exit_velo", "gb_rate", "ld_rate", "fb_rate",
        "whiff_rate", "chase_rate", "zone_contact_rate",
        "woba_vs_fastball", "woba_vs_breaking", "woba_vs_offspeed",
    ]
    return pd.DataFrame(columns=cols)


def _empty_pitcher_df():
    cols = [
        "player_name", "whiff_rate", "csw_rate", "k_rate", "bb_rate",
        "hard_hit_against", "barrel_against", "avg_exit_velo_against",
        "gb_rate", "fb_rate", "fastball_pct", "breaking_pct", "offspeed_pct",
        "avg_fastball_velo", "hr_rate_allowed",
    ]
    return pd.DataFrame(columns=cols)


def _empty_recent_form_df():
    cols = [
        "player_name", "recent_avg_ev", "recent_hard_hit", "recent_barrel_rate",
        "recent_batted", "recent_pa", "recent_hits", "recent_hrs",
        "recent_avg", "recent_hr_rate",
    ]
    return pd.DataFrame(columns=cols)


def _empty_expected_batter_df():
    return pd.DataFrame(columns=["player_name", "xwoba", "xba", "xslg"])


def _empty_expected_pitcher_df():
    return pd.DataFrame(columns=["player_name", "xwoba_against"])


def _create_placeholder_files():
    """Write empty CSV stubs so downstream modules can still load them."""
    for df, path in [
        (_empty_batter_df(), "batter_advanced_metrics_2024_2025.csv"),
        (_empty_pitcher_df(), "pitcher_advanced_metrics_2024_2025.csv"),
        (_empty_recent_form_df(), "recent_form_data.csv"),
        (_empty_expected_batter_df(), "batter_expected_stats.csv"),
        (_empty_expected_pitcher_df(), "pitcher_expected_stats.csv"),
        (pd.DataFrame(), "statcast_raw_2024_2025.csv"),
    ]:
        if not os.path.exists(path):
            df.to_csv(path, index=False)
            print(f"      -> Created placeholder {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(start_year: int = 2024, end_year: int = 2025):
    print("=" * 70)
    print("   ENHANCED STATCAST DATA PIPELINE")
    print(f"   Seasons: {start_year} – {end_year}")
    print("=" * 70)

    raw_df = fetch_statcast_data(start_year, end_year)
    batter_df = calculate_batter_metrics(raw_df)
    pitcher_df = calculate_pitcher_metrics(raw_df)
    recent_df = calculate_recent_form(raw_df)
    b_exp, p_exp = calculate_expected_stats(raw_df)

    print("\n[6/6] Pipeline complete.  Output files:")
    for f in [
        "statcast_raw_2024_2025.csv",
        "batter_advanced_metrics_2024_2025.csv",
        "pitcher_advanced_metrics_2024_2025.csv",
        "recent_form_data.csv",
        "batter_expected_stats.csv",
        "pitcher_expected_stats.csv",
    ]:
        size = f"{os.path.getsize(f):,} bytes" if os.path.exists(f) else "not created"
        print(f"   {f:<45} {size}")

    return batter_df, pitcher_df, recent_df, b_exp, p_exp


if __name__ == "__main__":
    run_pipeline()

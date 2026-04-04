# feature_engineering.py
"""
Feature engineering module for the Enhanced MLB Prediction System.

Provides the FeatureEngineer class which loads pre-computed advanced metrics
and builds feature vectors for ML models.

Usage:
    from feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    batter_feats  = fe.get_batter_features("Aaron Judge")
    pitcher_feats = fe.get_pitcher_features("Gerrit Cole")
    matchup        = fe.calculate_matchup_factors(batter_feats, pitcher_feats)
    vector         = fe.build_feature_vector("Aaron Judge", "Gerrit Cole", context)
"""

import os
import difflib
import unicodedata
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# League-average defaults  (used when a player is not in the metrics files)
# ---------------------------------------------------------------------------

LEAGUE_AVG_BATTER: Dict[str, Any] = {
    "barrel_rate": 0.076,
    "hard_hit_rate": 0.380,
    "sweet_spot_rate": 0.335,
    "avg_exit_velo": 88.5,
    "max_exit_velo": 108.0,
    "gb_rate": 0.435,
    "ld_rate": 0.215,
    "fb_rate": 0.350,
    "whiff_rate": 0.248,
    "chase_rate": 0.295,
    "zone_contact_rate": 0.863,
    "woba_vs_fastball": 0.330,
    "woba_vs_breaking": 0.290,
    "woba_vs_offspeed": 0.305,
    # basic rates (from daily_bets.py legacy pipeline)
    "1B_Rate": 0.145,
    "2B_Rate": 0.045,
    "3B_Rate": 0.004,
    "HR_Rate": 0.030,
    "BB_Rate": 0.085,
    "K_Rate": 0.225,
    "R_Conv": 0.310,
    "RBI_Conv": 0.150,
    "SB_Conv": 0.050,
    "Barrel_Rate": 0.080,
    "xwOBA": 0.320,
    "Archetype": "Balanced",
    "Hand": "R",
}

LEAGUE_AVG_PITCHER: Dict[str, Any] = {
    "whiff_rate": 0.248,
    "csw_rate": 0.300,
    "k_rate": 0.220,
    "bb_rate": 0.080,
    "hard_hit_against": 0.380,
    "barrel_against": 0.076,
    "avg_exit_velo_against": 88.5,
    "gb_rate": 0.435,
    "fb_rate": 0.350,
    "fastball_pct": 0.500,
    "breaking_pct": 0.300,
    "offspeed_pct": 0.200,
    "avg_fastball_velo": 93.5,
    "hr_rate_allowed": 0.030,
    # legacy fields used by daily_bets.py
    "CALC_HR9": 1.25,
    "K_Rate": 0.220,
    "BB_Rate": 0.080,
    "H_Rate": 0.240,
    "BF_per_Start": 22,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = "".join(
        c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn"
    )
    return name.lower().replace(".", "").replace("*", "").replace("#", "").strip()


def _fuzzy_match(raw: str, keys, cutoff: float = 0.70) -> Optional[str]:
    norm = _normalize(raw)
    if norm in keys:
        return norm
    matches = difflib.get_close_matches(norm, keys, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def _load_csv_as_dict(path: str, name_col: str = "player_name") -> Dict[str, Dict]:
    """Load a CSV and index it by normalised player name."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if name_col not in df.columns:
            return {}
        df["_key"] = df[name_col].apply(_normalize)
        df = df.drop_duplicates(subset=["_key"])
        return df.set_index("_key").drop(columns=[name_col]).to_dict("index")
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Loads pre-computed advanced Statcast metrics and assembles feature vectors
    for XGBoost models.
    """

    def __init__(self):
        self._batter_adv: Dict[str, Dict] = _load_csv_as_dict(
            "batter_advanced_metrics_2024_2025.csv"
        )
        self._pitcher_adv: Dict[str, Dict] = _load_csv_as_dict(
            "pitcher_advanced_metrics_2024_2025.csv"
        )
        self._recent_form: Dict[str, Dict] = _load_csv_as_dict("recent_form_data.csv")
        self._batter_exp: Dict[str, Dict] = _load_csv_as_dict("batter_expected_stats.csv")
        self._pitcher_exp: Dict[str, Dict] = _load_csv_as_dict("pitcher_expected_stats.csv")

        self._batter_keys = set(self._batter_adv.keys())
        self._pitcher_keys = set(self._pitcher_adv.keys())

        loaded = sum([
            bool(self._batter_adv), bool(self._pitcher_adv),
            bool(self._recent_form), bool(self._batter_exp),
        ])
        print(f"[FeatureEngineer] Loaded {loaded}/4 metrics files. "
              f"({len(self._batter_adv)} batters, {len(self._pitcher_adv)} pitchers)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batter_features(
        self, player_name: str, game_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Return a feature dict for *player_name*.
        Falls back to league-average defaults for any missing fields.
        """
        features = LEAGUE_AVG_BATTER.copy()
        key = _fuzzy_match(player_name, self._batter_keys)
        if key:
            adv = self._batter_adv[key]
            features.update({k: v for k, v in adv.items() if not pd.isna(v) if isinstance(v, (int, float, str))})

        # Merge recent form (overrides season averages for volatile metrics)
        rf_key = _fuzzy_match(player_name, set(self._recent_form.keys()))
        if rf_key:
            rf = self._recent_form[rf_key]
            for src_col, dst_col in [
                ("recent_avg_ev", "avg_exit_velo"),
                ("recent_hard_hit", "hard_hit_rate"),
                ("recent_barrel_rate", "barrel_rate"),
            ]:
                if src_col in rf and not pd.isna(rf[src_col]):
                    # Blend: 40% recent, 60% season
                    features[dst_col] = (
                        0.40 * float(rf[src_col]) + 0.60 * features[dst_col]
                    )

        # Merge expected stats
        exp_key = _fuzzy_match(player_name, set(self._batter_exp.keys()))
        if exp_key:
            exp = self._batter_exp[exp_key]
            if "xwoba" in exp and not pd.isna(exp["xwoba"]):
                features["xwOBA"] = float(exp["xwoba"])

        # Derived metrics
        features["power_speed_score"] = self._power_speed_score(features)
        features["contact_quality_score"] = self._contact_quality_score(features)
        features["iso"] = max(0.0, features["xwOBA"] - 0.310)
        features["xiso"] = (
            features["barrel_rate"] * 0.6 + features["hard_hit_rate"] * 0.1
        )
        return features

    def get_pitcher_features(self, player_name: str) -> Dict[str, Any]:
        """
        Return a feature dict for pitcher *player_name*.
        Falls back to league-average defaults for missing fields.
        """
        features = LEAGUE_AVG_PITCHER.copy()
        key = _fuzzy_match(player_name, self._pitcher_keys)
        if key:
            adv = self._pitcher_adv[key]
            features.update({k: v for k, v in adv.items() if not pd.isna(v) if isinstance(v, (int, float, str))})

        exp_key = _fuzzy_match(player_name, set(self._pitcher_exp.keys()))
        if exp_key:
            exp = self._pitcher_exp[exp_key]
            if "xwoba_against" in exp and not pd.isna(exp["xwoba_against"]):
                features["xwoba_against"] = float(exp["xwoba_against"])

        # Derived
        features["stuff_plus"] = self._stuff_plus(features)
        features["pitcher_quality_score"] = self._pitcher_quality_score(features)
        return features

    def calculate_matchup_factors(
        self,
        batter_features: Dict[str, Any],
        pitcher_features: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Return matchup-specific adjustment factors given pre-fetched feature dicts.
        All values are multiplicative adjustments (1.0 = neutral).
        """
        # Pitch-type matchup: batter wOBA vs pitcher's pitch mix
        b_woba_vs_fb = float(batter_features.get("woba_vs_fastball", 0.330))
        b_woba_vs_br = float(batter_features.get("woba_vs_breaking", 0.290))
        b_woba_vs_os = float(batter_features.get("woba_vs_offspeed", 0.305))
        p_fb_pct = float(pitcher_features.get("fastball_pct", 0.50))
        p_br_pct = float(pitcher_features.get("breaking_pct", 0.30))
        p_os_pct = float(pitcher_features.get("offspeed_pct", 0.20))
        weighted_woba = (
            b_woba_vs_fb * p_fb_pct
            + b_woba_vs_br * p_br_pct
            + b_woba_vs_os * p_os_pct
        )
        pitch_type_matchup = weighted_woba / 0.320  # normalise to league avg

        # Velo matchup: high-velo pitchers suppress high-chase-rate batters
        avg_fb_velo = float(pitcher_features.get("avg_fastball_velo", 93.5))
        chase_rate = float(batter_features.get("chase_rate", 0.295))
        velo_matchup = 1.0 - max(0.0, (avg_fb_velo - 93.5) / 100) * chase_rate

        # Contact vs whiff
        b_zone_contact = float(batter_features.get("zone_contact_rate", 0.863))
        p_whiff = float(pitcher_features.get("whiff_rate", 0.248))
        contact_vs_whiff = b_zone_contact * (1.0 - p_whiff)
        contact_vs_whiff /= (0.863 * 0.752)  # normalise

        # HR suppression: ground ball pitchers suppress HRs
        p_gb_rate = float(pitcher_features.get("gb_rate", 0.435))
        hr_suppression = 1.0 - max(0.0, (p_gb_rate - 0.435) * 1.5)

        # Overall matchup (weighted composite)
        overall_matchup = (
            0.35 * pitch_type_matchup
            + 0.25 * velo_matchup
            + 0.25 * contact_vs_whiff
            + 0.15 * hr_suppression
        )

        return {
            "pitch_type_matchup": round(pitch_type_matchup, 4),
            "velo_matchup": round(velo_matchup, 4),
            "contact_vs_whiff": round(contact_vs_whiff, 4),
            "hr_suppression": round(hr_suppression, 4),
            "overall_matchup": round(overall_matchup, 4),
        }

    def build_feature_vector(
        self,
        batter_name: str,
        pitcher_name: str,
        game_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a complete flat feature dict suitable for XGBoost prediction.

        *game_context* should contain keys like:
            temp, wind_speed, wind_dir, lineup_spot, park_hr_factor,
            park_avg_factor, park_runs_factor, platoon_adv
        """
        bf = self.get_batter_features(batter_name, game_context.get("game_date"))
        pf = self.get_pitcher_features(pitcher_name)
        mf = self.calculate_matchup_factors(bf, pf)

        vector: Dict[str, Any] = {}

        # ---- Batter features ----
        for key in [
            "barrel_rate", "hard_hit_rate", "sweet_spot_rate",
            "avg_exit_velo", "gb_rate", "ld_rate", "fb_rate",
            "whiff_rate", "chase_rate", "zone_contact_rate",
            "woba_vs_fastball", "woba_vs_breaking", "woba_vs_offspeed",
            "power_speed_score", "contact_quality_score", "iso", "xiso",
            "xwOBA", "HR_Rate", "K_Rate", "BB_Rate", "Barrel_Rate",
        ]:
            vector[f"b_{key}"] = float(bf.get(key, 0.0))

        # ---- Pitcher features ----
        for key in [
            "whiff_rate", "csw_rate", "k_rate", "bb_rate",
            "hard_hit_against", "barrel_against", "avg_exit_velo_against",
            "gb_rate", "fb_rate", "fastball_pct", "breaking_pct", "offspeed_pct",
            "avg_fastball_velo", "hr_rate_allowed", "stuff_plus", "pitcher_quality_score",
            "CALC_HR9",
        ]:
            vector[f"p_{key}"] = float(pf.get(key, 0.0))

        # ---- Matchup features ----
        for key, val in mf.items():
            vector[f"matchup_{key}"] = val

        # ---- Game context ----
        vector["Temp"] = float(game_context.get("temp", 72))
        vector["Wind_Speed"] = float(game_context.get("wind_speed", 0))
        vector["Lineup_Spot"] = int(game_context.get("lineup_spot", 5))
        vector["Platoon_Adv"] = int(game_context.get("platoon_adv", 0))
        vector["park_hr_factor"] = float(game_context.get("park_hr_factor", 1.0))
        vector["park_avg_factor"] = float(game_context.get("park_avg_factor", 1.0))
        vector["park_runs_factor"] = float(game_context.get("park_runs_factor", 1.0))

        # Legacy columns expected by existing XGBoost model
        vector["Batter_xwOBA"] = vector["b_xwOBA"]
        vector["Pitcher_HR9"] = vector["p_CALC_HR9"]

        return vector

    # ------------------------------------------------------------------
    # Private derived-metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _power_speed_score(bf: Dict) -> float:
        barrel = float(bf.get("barrel_rate", LEAGUE_AVG_BATTER["barrel_rate"]))
        hard_hit = float(bf.get("hard_hit_rate", LEAGUE_AVG_BATTER["hard_hit_rate"]))
        ev = float(bf.get("avg_exit_velo", LEAGUE_AVG_BATTER["avg_exit_velo"]))
        ev_norm = max(0.0, (ev - 80.0) / 30.0)
        return round(0.40 * barrel / 0.076 + 0.35 * hard_hit / 0.380 + 0.25 * ev_norm, 4)

    @staticmethod
    def _contact_quality_score(bf: Dict) -> float:
        sweet = float(bf.get("sweet_spot_rate", LEAGUE_AVG_BATTER["sweet_spot_rate"]))
        zone_ct = float(bf.get("zone_contact_rate", LEAGUE_AVG_BATTER["zone_contact_rate"]))
        whiff = float(bf.get("whiff_rate", LEAGUE_AVG_BATTER["whiff_rate"]))
        return round(0.40 * sweet / 0.335 + 0.35 * zone_ct / 0.863 + 0.25 * (1.0 - whiff) / 0.752, 4)

    @staticmethod
    def _stuff_plus(pf: Dict) -> float:
        """Approximate Stuff+ from whiff, CSW, and fastball velocity."""
        whiff = float(pf.get("whiff_rate", LEAGUE_AVG_PITCHER["whiff_rate"]))
        csw = float(pf.get("csw_rate", LEAGUE_AVG_PITCHER["csw_rate"]))
        velo = float(pf.get("avg_fastball_velo", LEAGUE_AVG_PITCHER["avg_fastball_velo"]))
        score = (whiff / 0.248) * 40 + (csw / 0.300) * 35 + ((velo - 90) / 6) * 25
        return round(score, 2)

    @staticmethod
    def _pitcher_quality_score(pf: Dict) -> float:
        """Overall pitcher quality for run-suppression purposes (higher = better)."""
        k_rate = float(pf.get("k_rate", LEAGUE_AVG_PITCHER["k_rate"]))
        bb_rate = float(pf.get("bb_rate", LEAGUE_AVG_PITCHER["bb_rate"]))
        hard_hit = float(pf.get("hard_hit_against", LEAGUE_AVG_PITCHER["hard_hit_against"]))
        barrel = float(pf.get("barrel_against", LEAGUE_AVG_PITCHER["barrel_against"]))
        score = (
            (k_rate / 0.220) * 0.35
            - (bb_rate / 0.080 - 1.0) * 0.20
            - (hard_hit / 0.380 - 1.0) * 0.25
            - (barrel / 0.076 - 1.0) * 0.20
        )
        return round(score, 4)

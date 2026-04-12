# game_markets_predictor.py
"""
Game-level markets predictor: NRFI/YRFI, Moneyline, Run Line, Game/Team Totals, F5.

Usage:
    from game_markets_predictor import GameMarketsPredictor
    gmp = GameMarketsPredictor()
    result = gmp.predict_full_game(away_lineup, home_lineup,
                                   away_pitcher, home_pitcher, ...)
"""

import os
import random
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

try:
    from feature_engineering import FeatureEngineer, LEAGUE_AVG_BATTER, LEAGUE_AVG_PITCHER
except ImportError:
    FeatureEngineer = None  # type: ignore
    LEAGUE_AVG_BATTER = {}
    LEAGUE_AVG_PITCHER = {}

# ---------------------------------------------------------------------------
# Load XGBoost (For Totals and NRFI)
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb

    _XGB = True
except ImportError:
    _XGB = False

TOTALS_MODEL = None
TOTALS_COLS = []


def _load_totals_model():
    global TOTALS_MODEL, TOTALS_COLS
    if not _XGB: return
    if os.path.exists("totals_model.json") and os.path.exists("totals_model_features.txt"):
        try:
            TOTALS_MODEL = xgb.XGBRegressor()
            TOTALS_MODEL.load_model("totals_model.json")
            with open("totals_model_features.txt", "r") as f:
                TOTALS_COLS = f.read().strip().split(",")
        except Exception as e:
            print(f"[!] Could not load totals model: {e}")


_load_totals_model()

# ---------------------------------------------------------------------------
# Park factors  {team_abbr / stadium: [hr_factor, avg_factor, runs_factor]}
# ---------------------------------------------------------------------------

PARK_FACTORS_FULL: Dict[str, List[float]] = {
    # Extreme hitter parks
    "Colorado Rockies": [1.38, 1.15, 1.35],
    "Coors Field": [1.38, 1.15, 1.35],
    "Cincinnati Reds": [1.26, 1.05, 1.18],
    "Great American Ball Park": [1.26, 1.05, 1.18],
    # Hitter-friendly
    "New York Yankees": [1.10, 0.98, 1.05],
    "Yankee Stadium": [1.10, 0.98, 1.05],
    "Chicago Cubs": [1.06, 1.01, 1.04],
    "Wrigley Field": [1.06, 1.01, 1.04],
    "Atlanta Braves": [1.05, 1.02, 1.04],
    "Truist Park": [1.05, 1.02, 1.04],
    "Los Angeles Dodgers": [1.18, 1.01, 1.08],
    "Dodger Stadium": [1.18, 1.01, 1.08],
    "Philadelphia Phillies": [1.15, 1.02, 1.08],
    "Citizens Bank Park": [1.15, 1.02, 1.08],
    "Houston Astros": [1.12, 1.00, 1.05],
    "Minute Maid Park": [1.12, 1.00, 1.05],
    "Milwaukee Brewers": [1.08, 1.00, 1.03],
    "American Family Field": [1.08, 1.00, 1.03],
    "Boston Red Sox": [1.09, 1.05, 1.06],
    "Fenway Park": [1.09, 1.05, 1.06],
    "Chicago White Sox": [1.03, 1.00, 1.01],
    "Guaranteed Rate Field": [1.03, 1.00, 1.01],
    "Los Angeles Angels": [1.05, 1.00, 1.02],
    "Angel Stadium": [1.05, 1.00, 1.02],
    "Texas Rangers": [1.04, 1.00, 1.02],
    "Globe Life Field": [1.04, 1.00, 1.02],
    "Baltimore Orioles": [1.05, 1.00, 1.02],
    "Oriole Park at Camden Yards": [1.05, 1.00, 1.02],
    # Neutral parks
    "Toronto Blue Jays": [1.00, 1.00, 1.00],
    "Rogers Centre": [1.00, 1.00, 1.00],
    "Minnesota Twins": [0.99, 1.00, 1.00],
    "Target Field": [0.99, 1.00, 1.00],
    "Washington Nationals": [0.98, 1.00, 0.99],
    "Nationals Park": [0.98, 1.00, 0.99],
    "Cleveland Guardians": [0.98, 1.00, 0.99],
    "Progressive Field": [0.98, 1.00, 0.99],
    "Arizona Diamondbacks": [0.98, 1.00, 0.99],
    "Chase Field": [0.98, 1.00, 0.99],
    # Pitcher-friendly
    "Kansas City Royals": [0.97, 1.00, 0.98],
    "Kauffman Stadium": [0.97, 1.00, 0.98],
    "New York Mets": [0.95, 1.00, 0.97],
    "Citi Field": [0.95, 1.00, 0.97],
    "Tampa Bay Rays": [0.95, 1.00, 0.97],
    "Tropicana Field": [0.95, 1.00, 0.97],
    "Miami Marlins": [0.93, 0.99, 0.95],
    "loanDepot park": [0.93, 0.99, 0.95],
    "Detroit Tigers": [0.94, 0.99, 0.96],
    "Comerica Park": [0.94, 0.99, 0.96],
    "Athletics": [0.92, 0.98, 0.94],
    "Oakland Coliseum": [0.92, 0.98, 0.94],
    "San Diego Padres": [0.96, 0.99, 0.97],
    "Petco Park": [0.96, 0.99, 0.97],
    "Seattle Mariners": [0.96, 0.95, 0.95],
    "T-Mobile Park": [0.96, 0.95, 0.95],
    "Pittsburgh Pirates": [0.82, 0.99, 0.92],
    "PNC Park": [0.82, 0.99, 0.92],
    # Extreme pitcher park
    "San Francisco Giants": [0.81, 0.99, 0.88],
    "Oracle Park": [0.81, 0.99, 0.88],
    "St. Louis Cardinals": [0.95, 1.00, 0.97],
    "Busch Stadium": [0.95, 1.00, 0.97],
}

DEFAULT_PARK = [1.0, 1.0, 1.0]


def _get_park(stadium: str) -> List[float]:
    return PARK_FACTORS_FULL.get(stadium, DEFAULT_PARK)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def format_odds(probability: float) -> str:
    """Convert win probability to American moneyline odds."""
    if probability <= 0.001:
        return "+9999"
    if probability >= 0.999:
        return "-9999"
    if probability > 0.50:
        return str(int((probability / (1 - probability)) * -100))
    return f"+{int((100 / probability) - 100)}"


def _calculate_sharp_line(values: List[float], target_median: Optional[float] = None) -> tuple:
    """
    Find the optimal O/U line (half-point increments).
    If target_median is provided (from XGBoost), it shifts the simulation array
    to match the blended AI prediction before calculating the line.
    """
    if not values:
        return 8.5, 0.5, 0.5
    arr = np.array(values, dtype=float)

    if target_median is not None:
        current_median = float(np.median(arr))
        shift = target_median - current_median
        arr = arr + shift

    # Use MEDIAN instead of MEAN to find the true center and ignore 15-run blowout outliers
    base_val = float(np.median(arr))

    # Round to nearest 0.5
    line = round(base_val * 2) / 2

    # Push to half-point if we land on a whole number to prevent "pushes" (ties)
    if line % 1 == 0:
        mean_val = float(np.mean(arr))
        line += 0.5 if mean_val >= line else -0.5

    over = float(np.mean(arr > line))
    under = float(np.mean(arr < line))
    return line, over, under


def get_edge_rating(our_prob: float, market_implied_prob: float) -> str:
    """Return a human-readable edge rating."""
    edge = our_prob - market_implied_prob
    if edge >= 0.06:
        return "🔥🔥🔥 STRONG EDGE"
    if edge >= 0.04:
        return "🔥🔥 MODERATE EDGE"
    if edge >= 0.02:
        return "🔥 SLIGHT EDGE"
    if edge <= -0.06:
        return "❌ STRONG FADE"
    if edge <= -0.04:
        return "⚠️ MODERATE FADE"
    return "➖ NEUTRAL"


# ---------------------------------------------------------------------------
# GameMarketsPredictor
# ---------------------------------------------------------------------------

class GameMarketsPredictor:
    """
    Simulates full MLB games (and first innings) to produce probabilistic
    predictions for game-level betting markets.
    """

    SIM_GAMES = 2_000  # Reduced from 10,000 for faster execution

    def __init__(self, feature_engineer: Optional[Any] = None):
        if feature_engineer is not None:
            self.fe = feature_engineer
        elif FeatureEngineer is not None:
            try:
                self.fe = FeatureEngineer()
            except Exception as exc:
                print(f"[GameMarketsPredictor] FeatureEngineer init failed: {exc}.  Using league-average defaults.")
                self.fe = None
        else:
            self.fe = None

    # ------------------------------------------------------------------
    # NRFI / YRFI
    # ------------------------------------------------------------------

    def calculate_pitcher_first_inning_score(self, pitcher_name: str) -> float:
        if self.fe:
            pf = self.fe.get_pitcher_features(pitcher_name)
        else:
            pf = LEAGUE_AVG_PITCHER.copy()

        k_rate = float(pf.get("k_rate", 0.220)) * 1.08
        bb_rate = float(pf.get("bb_rate", 0.080)) * 1.05
        hard_hit = float(pf.get("hard_hit_against", 0.380)) * 0.93
        barrel = float(pf.get("barrel_against", 0.076)) * 0.90
        stuff = float(pf.get("stuff_plus", 100.0)) if "stuff_plus" in pf else 100.0

        k_score = k_rate / 0.240
        bb_penalty = max(0.0, bb_rate - 0.080) * 2.0
        contact_penalty = (hard_hit - 0.380) * 1.5 + (barrel - 0.076) * 2.5
        stuff_bonus = (stuff - 100) / 200 if stuff > 100 else 0.0

        raw = 0.50 + 0.15 * (k_score - 1.0) - 0.10 * bb_penalty - 0.15 * contact_penalty + 0.10 * stuff_bonus
        return float(np.clip(raw, 0.05, 0.95))

    def calculate_lineup_first_inning_threat(self, lineup: List[Dict[str, Any]], pitcher_name: str,
                                             pitcher_hand: str) -> float:
        if not lineup:
            return 0.30

        weights = [0.35, 0.25, 0.22, 0.18]
        top4 = lineup[:4]
        threat = 0.0
        for i, batter in enumerate(top4):
            w = weights[i] if i < len(weights) else 0.10
            if self.fe:
                bf = self.fe.get_batter_features(batter.get("name", "Unknown"))
            else:
                bf = LEAGUE_AVG_BATTER.copy()
            b_hand = batter.get("hand", bf.get("Hand", "R"))
            platoon_mult = 1.05 if (b_hand != pitcher_hand and b_hand != "S") else 0.97
            obp_proxy = (float(bf.get("BB_Rate", 0.085)) + float(bf.get("HR_Rate", 0.030)) + float(
                bf.get("1B_Rate", 0.145)) + float(bf.get("2B_Rate", 0.045)))
            xwoba = float(bf.get("xwOBA", 0.320))
            batter_threat = (obp_proxy * 0.40 + xwoba / 0.320 * 0.60) * platoon_mult
            threat += w * batter_threat

        return float(np.clip(threat, 0.05, 0.95))

    def predict_nrfi_probability(self, away_lineup: List[Dict], home_lineup: List[Dict], away_pitcher: str,
                                 home_pitcher: str, stadium: str = "Unknown", away_pitcher_hand: str = "R",
                                 home_pitcher_hand: str = "R", weather: Optional[Dict] = None,
                                 n_simulations: int = 10_000) -> Dict[str, Any]:
        if weather is None:
            weather = {"temp": 72, "wind_speed": 0, "wind_dir": "none"}

        park = _get_park(stadium)
        runs_factor = park[2]

        away_p_score = self.calculate_pitcher_first_inning_score(away_pitcher)
        home_p_score = self.calculate_pitcher_first_inning_score(home_pitcher)
        away_threat = self.calculate_lineup_first_inning_threat(away_lineup, home_pitcher, home_pitcher_hand)
        home_threat = self.calculate_lineup_first_inning_threat(home_lineup, away_pitcher, away_pitcher_hand)

        weather_mult = 1.0 + max(0.0, (weather["temp"] - 70) * 0.003)
        if weather.get("wind_dir") == "out":
            weather_mult += weather.get("wind_speed", 0) * 0.004

        away_scores_analytical = float(
            np.clip(away_threat * (1.0 - away_p_score) * runs_factor * weather_mult, 0.05, 0.90))
        home_scores_analytical = float(
            np.clip(home_threat * (1.0 - home_p_score) * runs_factor * weather_mult, 0.05, 0.90))
        nrfi_analytical = (1.0 - away_scores_analytical) * (1.0 - home_scores_analytical)

        nrfi_count = 0
        away_scored_count = 0
        home_scored_count = 0
        for _ in range(n_simulations):
            a_runs = self._simulate_first_inning(away_lineup[:4], home_pitcher, home_pitcher_hand, park, weather)
            h_runs = self._simulate_first_inning(home_lineup[:4], away_pitcher, away_pitcher_hand, park, weather)
            if a_runs == 0 and h_runs == 0: nrfi_count += 1
            if a_runs > 0: away_scored_count += 1
            if h_runs > 0: home_scored_count += 1

        nrfi_mc = nrfi_count / n_simulations
        away_scores = 0.30 * away_scores_analytical + 0.70 * (away_scored_count / n_simulations)
        home_scores = 0.30 * home_scores_analytical + 0.70 * (home_scored_count / n_simulations)
        nrfi_prob = 0.30 * nrfi_analytical + 0.70 * nrfi_mc
        yrfi_prob = 1.0 - nrfi_prob

        return {
            "nrfi_prob": round(nrfi_prob, 4), "yrfi_prob": round(yrfi_prob, 4),
            "away_scores_prob": round(away_scores, 4), "home_scores_prob": round(home_scores, 4),
            "away_pitcher_score": round(away_p_score, 4), "home_pitcher_score": round(home_p_score, 4),
            "nrfi_odds": format_odds(nrfi_prob), "yrfi_odds": format_odds(yrfi_prob),
        }

    # ------------------------------------------------------------------
    # Full-game simulation
    # ------------------------------------------------------------------

    def _simulate_first_inning(self, batters: List[Dict], pitcher_name: str, pitcher_hand: str,
                               park_factors: List[float], weather: Dict) -> int:
        return self._simulate_half_inning(batters, pitcher_name, pitcher_hand, park_factors, weather, inning=1)

    def _simulate_half_inning(self, batters: List[Dict], pitcher_name: str, pitcher_hand: str,
                              park_factors: List[float], weather: Dict, inning: int = 1,
                              is_starter: bool = True) -> int:
        park_hr, park_avg, park_runs = park_factors
        if self.fe:
            pf = self.fe.get_pitcher_features(pitcher_name)
        else:
            pf = LEAGUE_AVG_PITCHER.copy()

        inning_k_adj = 1.08 if inning == 1 else (0.92 if inning >= 6 else 1.0)
        inning_hit_adj = 0.94 if inning == 1 else (1.08 if inning >= 6 else 1.0)

        p_k_rate = float(pf.get("k_rate", 0.220)) * inning_k_adj
        p_bb_rate = float(pf.get("bb_rate", 0.080))
        p_hr_rate = float(pf.get("hr_rate_allowed", 0.030))
        p_gb_rate = float(pf.get("gb_rate", 0.435))

        w_mult = 1.0 + max(0.0, (weather.get("temp", 72) - 70) * 0.008)
        if weather.get("wind_dir") == "out":
            w_mult += weather.get("wind_speed", 0) * 0.005

        outs = 0
        runners = [False, False, False]
        runs = 0
        batter_idx = 0
        n_batters = max(1, len(batters))

        while outs < 3:
            batter = batters[batter_idx % n_batters]
            batter_idx += 1

            if self.fe:
                bf = self.fe.get_batter_features(batter.get("name", "Unknown"))
            else:
                bf = LEAGUE_AVG_BATTER.copy()

            b_hand = batter.get("hand", "R")
            platoon_mult = 1.05 if (b_hand != pitcher_hand and b_hand != "S") else 0.97

            hr_rate = (float(bf.get("HR_Rate",
                                    0.030)) * park_hr * w_mult * platoon_mult * inning_hit_adj + p_hr_rate * park_hr * w_mult) / 2
            d2_rate = float(bf.get("2B_Rate", 0.045)) * park_avg * platoon_mult * inning_hit_adj
            d3_rate = float(bf.get("3B_Rate", 0.004)) * platoon_mult * inning_hit_adj
            s1_rate = float(bf.get("1B_Rate", 0.145)) * park_avg * platoon_mult * inning_hit_adj
            bb_rate = (float(bf.get("BB_Rate", 0.085)) + p_bb_rate) / 2
            k_rate = (float(bf.get("K_Rate", 0.225)) * p_k_rate / 0.220 + p_k_rate) / 2

            roll = random.random()
            cumulative = 0.0

            cumulative += k_rate
            if roll < cumulative:
                outs += 1
                continue

            cumulative += bb_rate
            if roll < cumulative:
                if runners[0] and runners[1] and runners[2]:
                    runs += 1
                elif runners[0] and runners[1]:
                    runners[2] = True
                elif runners[0]:
                    runners[1] = True
                else:
                    runners[0] = True
                continue

            cumulative += hr_rate
            if roll < cumulative:
                runs += 1 + sum(runners)
                runners = [False, False, False]
                continue

            cumulative += d3_rate
            if roll < cumulative:
                runs += sum(runners)
                runners = [False, False, True]
                continue

            cumulative += d2_rate
            if roll < cumulative:
                runs += runners[1] + runners[2]
                if runners[0]:
                    if random.random() < 0.50:
                        runs += 1;
                        runners = [False, True, False]
                    else:
                        runners = [False, True, True]
                else:
                    runners = [False, True, False]
                continue

            cumulative += s1_rate
            if roll < cumulative:
                new_runners = [True, runners[0], False]
                runs += runners[2]
                if runners[1]:
                    if random.random() < 0.65:
                        runs += 1
                    else:
                        new_runners[2] = True
                runners = new_runners
                continue

            if p_gb_rate > 0.50 and runners[0] and random.random() < 0.45:
                outs += 2
            else:
                outs += 1
            if outs < 3 and runners[2]:
                runs += 1
                runners[2] = False

        return max(0, runs)

    def predict_full_game(self, away_lineup: List[Dict], home_lineup: List[Dict], away_pitcher: str, home_pitcher: str,
                          stadium: str = "Unknown", away_pitcher_hand: str = "R", home_pitcher_hand: str = "R",
                          weather: Optional[Dict] = None, n_simulations: Optional[int] = None) -> Dict[str, Any]:
        if weather is None: weather = {"temp": 72, "wind_speed": 0, "wind_dir": "none"}
        if n_simulations is None: n_simulations = self.SIM_GAMES

        park = _get_park(stadium)
        ap_stuff = self.fe.get_pitcher_features(away_pitcher).get("stuff_plus", 100) if self.fe else 100.0
        hp_stuff = self.fe.get_pitcher_features(home_pitcher).get("stuff_plus", 100) if self.fe else 100.0

        away_starter_innings = max(4, min(7, int(5 + (ap_stuff - 100) / 40)))
        home_starter_innings = max(4, min(7, int(5 + (hp_stuff - 100) / 40)))

        tracker: Dict[str, Any] = {
            "away_wins": 0, "home_wins": 0, "away_covers_15": 0, "home_covers_15": 0,
            "nrfi": 0, "yrfi": 0, "f5_away_leads": 0, "f5_home_leads": 0, "f5_tie": 0,
            "game_totals": [], "f5_totals": [], "away_totals": [], "home_totals": [],
            "away_f5_totals": [], "home_f5_totals": [],
        }

        for _ in range(n_simulations):
            away_runs_by_inning = []
            home_runs_by_inning = []

            for inning in range(1, 10):
                is_a_starter = inning <= away_starter_innings
                is_h_starter = inning <= home_starter_innings
                a_runs = self._simulate_half_inning(away_lineup, home_pitcher, home_pitcher_hand, park, weather,
                                                    inning=inning, is_starter=is_h_starter)
                h_runs = self._simulate_half_inning(home_lineup, away_pitcher, away_pitcher_hand, park, weather,
                                                    inning=inning, is_starter=is_a_starter)
                away_runs_by_inning.append(a_runs)
                home_runs_by_inning.append(h_runs)

            total_away = sum(away_runs_by_inning)
            total_home = sum(home_runs_by_inning)

            extra = 0
            while total_away == total_home and extra < 3:
                total_away += self._simulate_half_inning(away_lineup, home_pitcher, home_pitcher_hand, park, weather,
                                                         inning=10)
                total_home += self._simulate_half_inning(home_lineup, away_pitcher, away_pitcher_hand, park, weather,
                                                         inning=10)
                extra += 1

            if total_away == total_home:
                if random.random() < 0.53:
                    total_home += 1
                else:
                    total_away += 1

            f5_away = sum(away_runs_by_inning[:5])
            f5_home = sum(home_runs_by_inning[:5])

            if away_runs_by_inning[0] == 0 and home_runs_by_inning[0] == 0:
                tracker["nrfi"] += 1
            else:
                tracker["yrfi"] += 1

            if total_away > total_home:
                tracker["away_wins"] += 1
            else:
                tracker["home_wins"] += 1

            if total_away - total_home >= 2: tracker["away_covers_15"] += 1
            if total_home - total_away >= 2: tracker["home_covers_15"] += 1

            if f5_away > f5_home:
                tracker["f5_away_leads"] += 1
            elif f5_home > f5_away:
                tracker["f5_home_leads"] += 1
            else:
                tracker["f5_tie"] += 1

            tracker["game_totals"].append(total_away + total_home)
            tracker["f5_totals"].append(f5_away + f5_home)
            tracker["away_totals"].append(total_away)
            tracker["home_totals"].append(total_home)
            tracker["away_f5_totals"].append(f5_away)
            tracker["home_f5_totals"].append(f5_home)

        return self._calculate_market_predictions(tracker, n_simulations, weather, park)

    # ------------------------------------------------------------------
    # Market calculations
    # ------------------------------------------------------------------

    def _calculate_market_predictions(self, tracker: Dict[str, Any], n_sims: int, weather: Dict, park: List[float]) -> \
    Dict[str, Any]:
        """Convert raw simulation counts to market probabilities using XGBoost blending."""

        xgb_target = None
        if TOTALS_MODEL is not None and TOTALS_COLS:
            wind_out = 1 if weather.get("wind_dir") == "out" else 0
            feat_dict = {
                'park_factor': park[2],
                'temp': weather.get("temp", 72),
                'wind_speed': weather.get("wind_speed", 0),
                'wind_out': wind_out
            }
            row = {col: feat_dict.get(col, 0) for col in TOTALS_COLS}
            X = pd.DataFrame([row])
            xgb_pred = float(TOTALS_MODEL.predict(X)[0])
            mc_median = float(np.median(tracker["game_totals"]))

            # The Magic Sauce: Blend specific matchups (MC) with environmental history (XGB)
            xgb_target = (xgb_pred + mc_median) / 2.0

        # We pass the blended target to shift the distribution properly
        gt_line, gt_over, gt_under = _calculate_sharp_line(tracker["game_totals"], target_median=xgb_target)

        f5_line, f5_over, f5_under = _calculate_sharp_line(tracker["f5_totals"])
        a_tt_line, a_tt_over, a_tt_under = _calculate_sharp_line(tracker["away_totals"])
        h_tt_line, h_tt_over, h_tt_under = _calculate_sharp_line(tracker["home_totals"])
        a_f5_line, a_f5_over, a_f5_under = _calculate_sharp_line(tracker["away_f5_totals"])
        h_f5_line, h_f5_over, h_f5_under = _calculate_sharp_line(tracker["home_f5_totals"])

        away_ml = tracker["away_wins"] / n_sims
        home_ml = tracker["home_wins"] / n_sims
        away_rl = tracker["away_covers_15"] / n_sims
        home_rl = tracker["home_covers_15"] / n_sims
        nrfi_p = tracker["nrfi"] / n_sims
        yrfi_p = tracker["yrfi"] / n_sims
        f5_away = tracker["f5_away_leads"] / n_sims
        f5_home = tracker["f5_home_leads"] / n_sims
        f5_tie = tracker["f5_tie"] / n_sims

        return {
            "away_ml_prob": round(away_ml, 4), "home_ml_prob": round(home_ml, 4),
            "away_ml_odds": format_odds(away_ml), "home_ml_odds": format_odds(home_ml),
            "away_rl_prob": round(away_rl, 4), "home_rl_prob": round(home_rl, 4),
            "away_rl_odds": format_odds(away_rl), "home_rl_odds": format_odds(home_rl),

            "game_total_line": gt_line,
            "game_total_over": round(gt_over, 4), "game_total_under": round(gt_under, 4),
            "game_total_mean": round(xgb_target if xgb_target else float(np.mean(tracker["game_totals"])), 2),

            "away_tt_line": a_tt_line, "away_tt_over": round(a_tt_over, 4), "away_tt_under": round(a_tt_under, 4),
            "home_tt_line": h_tt_line, "home_tt_over": round(h_tt_over, 4), "home_tt_under": round(h_tt_under, 4),

            "f5_away_prob": round(f5_away, 4), "f5_home_prob": round(f5_home, 4), "f5_tie_prob": round(f5_tie, 4),
            "f5_total_line": f5_line, "f5_total_over": round(f5_over, 4), "f5_total_under": round(f5_under, 4),
            "away_f5_line": a_f5_line, "away_f5_over": round(a_f5_over, 4),
            "home_f5_line": h_f5_line, "home_f5_over": round(h_f5_over, 4),

            "nrfi_prob": round(nrfi_p, 4), "yrfi_prob": round(yrfi_p, 4),
            "nrfi_odds": format_odds(nrfi_p), "yrfi_odds": format_odds(yrfi_p),
        }


# ---------------------------------------------------------------------------
# NRFI model training
# ---------------------------------------------------------------------------
def train_nrfi_model():
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import brier_score_loss

    try:
        import xgboost as xgb
    except ImportError:
        return

    if not os.path.exists("nrfi_training_data.csv"): return

    df = pd.read_csv("nrfi_training_data.csv").dropna()
    feature_cols = [
        "away_whiff_rate", "away_k_rate", "away_bb_rate", "away_hard_hit_against", "away_gb_rate", "away_stuff_plus",
        "home_whiff_rate", "home_k_rate", "home_bb_rate", "home_hard_hit_against", "home_gb_rate", "home_stuff_plus",
        "away_top4_xwoba", "away_leadoff_xwoba", "away_top4_hard_hit", "away_top4_barrel",
        "home_top4_xwoba", "home_leadoff_xwoba", "home_top4_hard_hit", "home_top4_barrel",
        "park_runs_factor", "temp", "wind_out",
    ]
    available = [c for c in feature_cols if c in df.columns]
    if not available: return

    X = df[available]
    y = df["nrfi_outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict_proba(X_test)[:, 1]
    bs = brier_score_loss(y_test, preds)
    print(f"[NRFI Model] Brier Score: {bs:.4f}")
    model.save_model("nrfi_model.json")
    with open("nrfi_model_features.txt", "w") as f:
        f.write(",".join(available))
    print("[SUCCESS] NRFI model saved to nrfi_model.json")
# daily_bets.py
import pandas as pd
import numpy as np
import random
import difflib
import unicodedata
import sys
import os
from live_scraper import get_todays_matchups
from datetime import datetime

# Load the AI Brain
try:
    import xgboost as xgb

    AI_BRAIN = xgb.XGBClassifier()
    AI_BRAIN.load_model("mlb_xgboost_brain.json")
    with open("xgboost_columns.txt", "r") as f:
        AI_COLS = f.read().strip().split(",")
    AI_ACTIVE = True
except Exception:
    AI_ACTIVE = False

NAME_ALIASES = {
    'Jazz Chisholm': 'Jazz Chisholm Jr', 'Luis Robert': 'Luis Robert Jr', 'Shohei Ohtani': 'Shohei Ohtani',
}

PARK_FACTORS = {
    'Colorado Rockies': [1.13, 1.15], 'Cincinnati Reds': [1.26, 1.05], 'New York Yankees': [1.10, 0.98],
    'San Francisco Giants': [0.81, 0.99], 'Seattle Mariners': [0.96, 0.95], 'Pittsburgh Pirates': [0.82, 0.99],
    'Chicago Cubs': [1.06, 1.01], 'Atlanta Braves': [1.05, 1.02], 'Los Angeles Dodgers': [1.18, 1.01]
}

SIM_GAMES = 10000


def normalize_name(name):
    if not isinstance(name, str): return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    return name.lower().replace(".", "").replace("*", "").replace("#", "").strip()


def match_player_name(raw_name, db_keys):
    norm_raw = normalize_name(raw_name)
    if raw_name in NAME_ALIASES: norm_raw = normalize_name(NAME_ALIASES[raw_name])
    if norm_raw in db_keys: return norm_raw
    matches = difflib.get_close_matches(norm_raw, db_keys, n=1, cutoff=0.70)
    if matches: return matches[0]
    return norm_raw


def categorize_batter_archetype(k_rate, hr_rate):
    if hr_rate >= 0.035 and k_rate >= 0.21:
        return 'Slugger'
    elif k_rate <= 0.19 and hr_rate < 0.035:
        return 'Contact'
    else:
        return 'Balanced'


def get_col_safe(df, col_name, default_val=0):
    if col_name in df.columns: return pd.to_numeric(df[col_name], errors='coerce').fillna(default_val)
    return pd.Series(default_val, index=df.index)


def fetch_best_available_data():
    try:
        from pybaseball import batting_stats, pitching_stats
        b, p = batting_stats(2025, qual=30), pitching_stats(2025, qual=50)
        if b is not None and not b.empty:
            b.columns, p.columns = b.columns.str.upper(), p.columns.str.upper()
            if 'NAME' in b.columns: return b, p, "FanGraphs"
    except Exception:
        pass
    try:
        from pybaseball import batting_stats_bref, pitching_stats_bref
        b, p = batting_stats_bref(2025), pitching_stats_bref(2025)
        if b is not None and not b.empty:
            b.columns, p.columns = b.columns.str.upper(), p.columns.str.upper()
            if 'NAME' in b.columns:
                b['NAME'] = b['NAME'].astype(str).str.replace(r'[*#]', '', regex=True).str.strip()
                p['NAME'] = p['NAME'].astype(str).str.replace(r'[*#]', '', regex=True).str.strip()
                return b, p, "Baseball-Reference"
    except Exception:
        pass
    try:
        from pybaseball import statcast_batter_expected_stats, statcast_pitcher_expected_stats
        b, p = statcast_batter_expected_stats(2025, 50), statcast_pitcher_expected_stats(2025, 50)
        if b is not None and not b.empty:
            b.columns, p.columns = b.columns.str.upper(), p.columns.str.upper()
            b['NAME'], p['NAME'] = b['FIRST_NAME'] + ' ' + b['LAST_NAME'], p['FIRST_NAME'] + ' ' + p['LAST_NAME']
            b['PA'] = get_col_safe(b, 'PA', 1)
            b['AVG'], b['SLG'] = get_col_safe(b, 'BA', 0.240), get_col_safe(b, 'SLG', 0.400)
            b['HR'], b['H'] = (b['SLG'] - b['AVG']) * 0.15 * b['PA'], b['AVG'] * b['PA']
            b['2B'], b['3B'], b['BB'], b['SO'], b['SB'] = b['H'] * 0.20, b['H'] * 0.02, b['PA'] * 0.085, b[
                'PA'] * 0.225, b['PA'] * 0.02
            b['R'], b['RBI'], b['BARREL%'], b['XWOBA'] = b['H'] * 0.4, b['H'] * 0.4, 0.08, b[
                'EST_WOBA'] if 'EST_WOBA' in b.columns else 0.320
            p['IP'] = get_col_safe(p, 'PA', 500) / 4.2
            p['HR'] = get_col_safe(p, 'EST_SLG', 0.400) * 15
            p['SO'], p['BB'], p['H'] = (get_col_safe(p, 'K_PERCENT', 22.0) / 100) * get_col_safe(p, 'PA', 500), (
                        get_col_safe(p, 'BB_PERCENT', 8.0) / 100) * get_col_safe(p, 'PA', 500), get_col_safe(p,
                                                                                                             'EST_BA',
                                                                                                             0.240) * get_col_safe(
                p, 'PA', 500)
            p['BF'], p['GS'] = get_col_safe(p, 'PA', 500), p['IP'] / 5.0
            return b, p, "MLB Statcast Savant"
    except Exception:
        pass
    return None, None, "NONE"


def get_prop_matrices():
    print("\n[1/3] Running Cascading Tank Fetcher for 2025 True DNA...")
    try:
        b_df, p_df, source = fetch_best_available_data()
        if b_df is None: raise Exception("Rate Limit Jail: All providers blocked.")
        print(f"      -> CONNECTION ESTABLISHED via {source}!")
        b_df, p_df = b_df.reset_index(), p_df.reset_index()
        b_df['NAME_NORM'], p_df['NAME_NORM'] = b_df['NAME'].apply(normalize_name), p_df['NAME'].apply(normalize_name)
        b_df, p_df = b_df.drop_duplicates(subset=['NAME_NORM']), p_df.drop_duplicates(subset=['NAME_NORM'])

        pa, h, d2, d3, hr, bb, so, r, rbi, sb = get_col_safe(b_df, 'PA', 1).replace(0, 1), get_col_safe(b_df, 'H',
                                                                                                        0), get_col_safe(
            b_df, '2B', 0), get_col_safe(b_df, '3B', 0), get_col_safe(b_df, 'HR', 0), get_col_safe(b_df, 'BB',
                                                                                                   0), get_col_safe(
            b_df, 'SO', 0), get_col_safe(b_df, 'R', 0), get_col_safe(b_df, 'RBI', 0), get_col_safe(b_df, 'SB', 0)
        b_df['1B_CALC'] = h - d2 - d3 - hr
        b_df['1B_Rate'], b_df['2B_Rate'], b_df['3B_Rate'], b_df['HR_Rate'], b_df['BB_Rate'], b_df['K_Rate'] = b_df[
                                                                                                                  '1B_CALC'] / pa, d2 / pa, d3 / pa, hr / pa, bb / pa, so / pa
        b_df['Barrel_Rate'], b_df['xwOBA'] = get_col_safe(b_df, 'BARREL%', 0.08), get_col_safe(b_df, 'XWOBA', 0.320)
        b_df['R_Conv'], b_df['RBI_Conv'], b_df['SB_Conv'] = np.where((h + bb) > 0, r / (h + bb), 0), np.where(
            (h - hr) > 0, (rbi - hr) / (h - hr), 0), np.where((b_df['1B_CALC'] + bb) > 0, sb / (b_df['1B_CALC'] + bb),
                                                              0)
        b_df['Archetype'] = b_df.apply(lambda row: categorize_batter_archetype(row['K_Rate'], row['HR_Rate']), axis=1)
        batters = b_df.set_index('NAME_NORM')[
            ['1B_Rate', '2B_Rate', '3B_Rate', 'HR_Rate', 'BB_Rate', 'K_Rate', 'R_Conv', 'RBI_Conv', 'SB_Conv',
             'Barrel_Rate', 'xwOBA', 'Archetype']].to_dict('index')

        p_ip, p_so, p_bb, p_h, p_hr, p_gs = get_col_safe(p_df, 'IP', 1).replace(0, 1), get_col_safe(p_df, 'SO',
                                                                                                    0), get_col_safe(
            p_df, 'BB', 0), get_col_safe(p_df, 'H', 0), get_col_safe(p_df, 'HR', 0), get_col_safe(p_df, 'GS',
                                                                                                  1).replace(0, 1)
        p_bf = get_col_safe(p_df, 'TBF', 1) if 'TBF' in p_df.columns else get_col_safe(p_df, 'BFP',
                                                                                       1) if 'BFP' in p_df.columns else get_col_safe(
            p_df, 'BF', 1) if 'BF' in p_df.columns else (p_ip * 3) + p_h + p_bb
        p_bf = p_bf.replace(0, 1)
        p_df['CALC_HR9'], p_df['K_Rate'], p_df['BB_Rate'], p_df['H_Rate'] = p_hr / (
                    p_ip / 9), p_so / p_bf, p_bb / p_bf, p_h / p_bf
        p_df['BF_per_Start'] = np.where((p_bf / p_gs) < 15, 22, p_bf / p_gs)
        pitchers = p_df.set_index('NAME_NORM')[['CALC_HR9', 'K_Rate', 'BB_Rate', 'H_Rate', 'BF_per_Start']].to_dict(
            'index')

        print(f"      -> SUCCESS: Loaded {len(batters)} Batters and {len(pitchers)} Pitchers into Memory.")
        return batters, pitchers
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        return {}, {}


def generate_pitcher_profile(p_hr9):
    suppression = p_hr9 / 1.25
    if p_hr9 <= 1.05:
        return 'Spin', {'Fastball': {'base_usage': 0.40, 'hit_mod': 1.0 * suppression, 'hr_mod': 1.05 * suppression},
                        'Breaking': {'base_usage': 0.45, 'hit_mod': 0.80 * suppression, 'hr_mod': 0.75 * suppression},
                        'Offspeed': {'base_usage': 0.15, 'hit_mod': 0.95 * suppression, 'hr_mod': 0.90 * suppression}}
    elif p_hr9 >= 1.20:
        return 'Power', {'Fastball': {'base_usage': 0.65, 'hit_mod': 1.1 * suppression, 'hr_mod': 1.25 * suppression},
                         'Breaking': {'base_usage': 0.25, 'hit_mod': 0.95 * suppression, 'hr_mod': 0.90 * suppression},
                         'Offspeed': {'base_usage': 0.10, 'hit_mod': 1.0 * suppression, 'hr_mod': 1.0 * suppression}}
    else:
        return 'Balanced', {'Fastball': {'base_usage': 0.50, 'hit_mod': 1.0 * suppression, 'hr_mod': 1.0 * suppression},
                            'Breaking': {'base_usage': 0.30, 'hit_mod': 0.90 * suppression,
                                         'hr_mod': 0.85 * suppression},
                            'Offspeed': {'base_usage': 0.20, 'hit_mod': 0.95 * suppression,
                                         'hr_mod': 0.95 * suppression}}


def adjust_pitch_mix(arsenal, p_type, b_type):
    fb, br, os = arsenal['Fastball']['base_usage'], arsenal['Breaking']['base_usage'], arsenal['Offspeed']['base_usage']
    if b_type == 'Slugger':
        fb, br, os = max(0.20, fb - 0.15), br + 0.10, os + 0.05
    elif b_type == 'Contact':
        fb, br, os = min(0.75, fb + 0.10), max(0.10, br - 0.05), max(0.10, os - 0.05)
    total = fb + br + os
    return [fb / total, br / total, os / total]


def simulate_pitcher_game(p_stats, p_hand, lineup_b_stats):
    ks, bbs, hits = 0, 0, 0
    bf_target = max(9, int(random.gauss(p_stats['BF_per_Start'], 3.0)))
    for i in range(bf_target):
        b_stats = lineup_b_stats[i % len(lineup_b_stats)]
        has_platoon_adv = (b_stats['Hand'] == 'S') or (p_hand != b_stats['Hand'])
        prob_k = (p_stats['K_Rate'] + (b_stats['K_Rate'] * (0.94 if has_platoon_adv else 1.06))) / 2.0
        prob_bb = (p_stats['BB_Rate'] + b_stats['BB_Rate']) / 2.0
        prob_h = (p_stats['H_Rate'] + (
                    (b_stats['1B_Rate'] + b_stats['2B_Rate'] + b_stats['3B_Rate'] + b_stats['HR_Rate']) * (
                1.05 if has_platoon_adv else 0.96))) / 2.0
        roll = random.random()
        if roll < prob_k:
            ks += 1
        elif roll < prob_k + prob_bb:
            bbs += 1
        elif roll < prob_k + prob_bb + prob_h:
            hits += 1
    return ks, bbs, hits


def simulate_full_game_with_archetypes(b_stats, p_hr9, p_hand, w_boost, park_hr_val, park_avg_val, order_index):
    game_hits, game_tb, game_hr, game_r, game_rbi, game_bb, game_sb = 0, 0, 0, 0, 0, 0, 0
    runs_by_pa = [0] * 8
    avg_pa = [4.65, 4.53, 4.44, 4.35, 4.25, 4.15, 4.03, 3.93, 3.83][min(order_index, 8)]
    plate_appearances = 3
    if random.random() < min(1.0, avg_pa - 3):
        plate_appearances += 1
        if random.random() < max(0.0, avg_pa - 4): plate_appearances += 1

    p_archetype, arsenal = generate_pitcher_profile(p_hr9)
    pitch_types = list(arsenal.keys())
    contextual_usages = adjust_pitch_mix(arsenal, p_archetype, b_stats['Archetype'])
    has_platoon_adv = (b_stats['Hand'] == 'S') or (p_hand != b_stats['Hand'])
    p_h_mod, p_pwr_mod, p_k_mod = (1.05, 1.10, 0.94) if has_platoon_adv else (0.96, 0.92, 1.06)

    for pa_idx in range(plate_appearances):
        pitch = random.choices(pitch_types, weights=contextual_usages)[0]
        p_data = arsenal[pitch]
        mod_1B = b_stats['1B_Rate'] * p_data['hit_mod'] * park_avg_val * p_h_mod
        mod_2B = b_stats['2B_Rate'] * p_data['hit_mod'] * park_avg_val * p_pwr_mod
        mod_3B = b_stats['3B_Rate'] * p_data['hit_mod'] * park_avg_val * p_pwr_mod
        mod_HR = b_stats['HR_Rate'] * p_data['hr_mod'] * w_boost * park_hr_val * p_pwr_mod

        roll, threshold = random.random(), b_stats['BB_Rate']
        if roll < threshold:
            game_bb += 1
            if random.random() < b_stats['SB_Conv']: game_sb += 1
            if random.random() < b_stats['R_Conv']: game_r += 1; runs_by_pa[pa_idx] += 1
            continue
        threshold += b_stats['K_Rate'] * p_k_mod * (1.25 if pitch == 'Breaking' else 0.85)
        if roll < threshold: continue
        threshold += mod_HR
        if roll < threshold:
            game_tb += 4;
            game_hr += 1;
            game_hits += 1;
            game_r += 1;
            runs_by_pa[pa_idx] += 1
            game_rbi += random.choices([1, 2, 3, 4], weights=[0.55, 0.30, 0.10, 0.05])[0]
            continue
        threshold += mod_3B
        if roll < threshold:
            game_tb += 3;
            game_hits += 1
            if random.random() < b_stats['RBI_Conv']: game_rbi += 1
            if random.random() < b_stats['R_Conv']: game_r += 1; runs_by_pa[pa_idx] += 1
            continue
        threshold += mod_2B
        if roll < threshold:
            game_tb += 2;
            game_hits += 1
            if random.random() < b_stats['RBI_Conv']: game_rbi += 1
            if random.random() < b_stats['R_Conv']: game_r += 1; runs_by_pa[pa_idx] += 1
            continue
        threshold += mod_1B
        if roll < threshold:
            game_tb += 1;
            game_hits += 1
            if random.random() < b_stats['SB_Conv']: game_sb += 1
            if random.random() < b_stats['RBI_Conv']: game_rbi += 1
            if random.random() < b_stats['R_Conv']: game_r += 1; runs_by_pa[pa_idx] += 1
            continue
    return game_hr, game_hits, game_tb, game_r, game_rbi, game_hits + game_r + game_rbi, game_bb, game_sb, runs_by_pa


def get_target_odds_range(probability):
    if probability <= 0.001 or probability >= 0.999: return "N/A"
    dec = lambda d: f"+{int((d - 1) * 100)}" if d >= 2.0 else str(int(-100 / (d - 1)))
    return f"{dec(1.02 / probability)} to {dec(1.10 / probability)}"


def format_odds(probability):
    if probability <= 0.001: return "+9999"
    if probability >= 0.999: return "-9999"
    return str(
        int((probability / (1 - probability)) * -100)) if probability > 0.50 else f"+{int((100 / probability) - 100)}"


def calculate_sharp_ou_line(sim_array):
    mean_val = np.mean(sim_array)
    line = round(mean_val * 2) / 2
    if line % 1 == 0: line += 0.5 if mean_val >= line else -0.5
    return line, sum(1 for x in sim_array if x > line) / len(sim_array), sum(1 for x in sim_array if x < line) / len(
        sim_array)


def apply_xgboost_filter(raw_prob, market, features):
    """Feeds the raw simulation through the XGBoost Brain to get the Smart Probability."""
    if not AI_ACTIVE: return raw_prob
    row_data = {col: 0 for col in AI_COLS}

    for f in ['Temp', 'Wind_Speed', 'Lineup_Spot', 'Batter_xwOBA', 'Pitcher_HR9', 'Platoon_Adv']:
        if f in row_data: row_data[f] = features[f]
    row_data['Prob'] = raw_prob

    b_arch_col = f"Batter_Archetype_{features['Batter_Archetype']}"
    if b_arch_col in row_data: row_data[b_arch_col] = 1

    p_arch_col = f"Pitcher_Archetype_{features['Pitcher_Archetype']}"
    if p_arch_col in row_data: row_data[p_arch_col] = 1

    market_col = f"Market_{market}"
    if market_col in row_data: row_data[market_col] = 1

    df_pred = pd.DataFrame([row_data])[AI_COLS]
    return AI_BRAIN.predict_proba(df_pred)[0][1]


def run_prop_market_simulation():
    batters_db, pitchers_db = get_prop_matrices()
    batter_keys = list(batters_db.keys())

    if AI_ACTIVE:
        print("\n[🧠] XGBoost AI Brain Loaded! Predictions will be mathematically filtered.")
    else:
        print("\n[!] No AI Brain found. Using Raw Monte Carlo probabilities.")

    print("\n[2/3] Scouting Today's Live Matchups & Weather...")
    all_matchups = get_todays_matchups()
    if not all_matchups:
        print("\n[!] No games scheduled for today.")
        return

    games_dict = {}
    for m in all_matchups:
        games_dict.setdefault(m['home_stadium'], []).append(m)
    game_list = list(games_dict.values())

    if "--auto" in sys.argv:
        selected_matchups = game_list
    else:
        print("\n=============================================")
        print("           MLB GAMES AVAILABLE               ")
        print("=============================================")
        for i, gt in enumerate(game_list): print(
            f" {i + 1}. {gt[0]['team']} @ {gt[1]['team'] if len(gt) > 1 else gt[0]['home_stadium']}")
        print("-" * 45)
        print(f" {len(game_list) + 1}. Run All Games\n=============================================")
        try:
            choice_idx = int(input(f"\nSelect a game to simulate (1-{len(game_list) + 1}): ")) - 1
            selected_matchups = game_list if choice_idx == len(game_list) else [game_list[choice_idx]]
        except:
            selected_matchups = game_list

    league_avg_batter = {'1B_Rate': 0.145, '2B_Rate': 0.045, '3B_Rate': 0.004, 'HR_Rate': 0.030, 'BB_Rate': 0.085,
                         'K_Rate': 0.225, 'R_Conv': 0.310, 'RBI_Conv': 0.150, 'SB_Conv': 0.050, 'Barrel_Rate': 0.08,
                         'xwOBA': 0.320, 'Archetype': 'Balanced'}
    league_avg_pitcher = {'CALC_HR9': 1.25, 'K_Rate': 0.22, 'BB_Rate': 0.08, 'H_Rate': 0.24, 'BF_per_Start': 22}

    today_str = datetime.now().strftime('%Y-%m-%d')
    report = [f"=== MLB XGBOOST DASHBOARD ({SIM_GAMES} Games Simulated) ===", f"Date: {today_str}\n"]
    ledger_rows = []

    for game_teams in selected_matchups:
        if len(game_teams) == 2:
            away_m, home_m = game_teams[0], game_teams[1]
            stadium, w = away_m['home_stadium'], away_m['weather']
            w_boost = 1 + ((w['temp'] - 70) * 0.01) + ((w['wind_speed'] / 5) * 0.05 if w['wind_dir'] == 'out' else 0)
            park_hr, park_avg = PARK_FACTORS.get(stadium, [1.0, 1.0])

            p_stats_home = pitchers_db.get(match_player_name(away_m['opposing_pitcher'], list(pitchers_db.keys())),
                                           league_avg_pitcher)
            p_stats_away = pitchers_db.get(match_player_name(home_m['opposing_pitcher'], list(pitchers_db.keys())),
                                           league_avg_pitcher)

            away_lineup = [{**batters_db.get(match_player_name(b['name'], batter_keys), league_avg_batter).copy(),
                            'Hand': b['hand'], 'Name': b['name']} for b in away_m['lineup']]
            home_lineup = [{**batters_db.get(match_player_name(b['name'], batter_keys), league_avg_batter).copy(),
                            'Hand': b['hand'], 'Name': b['name']} for b in home_m['lineup']]

            tracker = {'away_wins': 0, 'home_wins': 0, 'away_minus_15': 0, 'home_minus_15': 0, 'f5_away': 0,
                       'f5_home': 0, 'f5_tie': 0, 'nrfi': 0, 'yrfi': 0, 'totals': [], 'f5_totals': [], 'away_tt': [],
                       'home_tt': []}

            for _ in range(SIM_GAMES):
                away_r, away_f5, away_1st = 0, 0, 0
                for idx, b in enumerate(away_lineup):
                    *_, runs_pa = simulate_full_game_with_archetypes(b, p_stats_home['CALC_HR9'],
                                                                     away_m.get('opposing_pitcher_hand', 'R'), w_boost,
                                                                     park_hr, park_avg, idx)
                    away_r += sum(runs_pa);
                    away_f5 += sum(runs_pa[:3])
                    if idx < 4: away_1st += runs_pa[0]

                home_r, home_f5, home_1st = 0, 0, 0
                for idx, b in enumerate(home_lineup):
                    *_, runs_pa = simulate_full_game_with_archetypes(b, p_stats_away['CALC_HR9'],
                                                                     home_m.get('opposing_pitcher_hand', 'R'), w_boost,
                                                                     park_hr, park_avg, idx)
                    home_r += sum(runs_pa);
                    home_f5 += sum(runs_pa[:3])
                    if idx < 4: home_1st += runs_pa[0]

                if away_r == home_r: away_r, home_r = (away_r, home_r + 1) if random.random() < 0.53 else (away_r + 1,
                                                                                                           home_r)
                if away_r > home_r:
                    tracker['away_wins'] += 1
                else:
                    tracker['home_wins'] += 1
                if away_r - home_r >= 1.5: tracker['away_minus_15'] += 1
                if home_r - away_r >= 1.5: tracker['home_minus_15'] += 1
                if away_f5 > home_f5:
                    tracker['f5_away'] += 1
                elif home_f5 > away_f5:
                    tracker['f5_home'] += 1
                else:
                    tracker['f5_tie'] += 1
                if away_1st == 0 and home_1st == 0:
                    tracker['nrfi'] += 1
                else:
                    tracker['yrfi'] += 1
                tracker['totals'].append(away_r + home_r);
                tracker['f5_totals'].append(away_f5 + home_f5)
                tracker['away_tt'].append(away_r);
                tracker['home_tt'].append(home_r)

            away_ml_prob, home_ml_prob = tracker['away_wins'] / SIM_GAMES, tracker['home_wins'] / SIM_GAMES
            gt_line, gt_over, gt_under = calculate_sharp_ou_line(tracker['totals'])

            game_ml_features = {
                'Date': today_str, 'Stadium': stadium, 'Temp': w['temp'], 'Wind_Speed': w['wind_speed'],
                'Wind_Dir': w['wind_dir'],
                'Away_Team': away_m['team'], 'Home_Team': home_m['team']
            }
            ledger_rows.extend([
                {**game_ml_features, 'Player': 'GAME_TOTAL', 'Market': f'Over_{gt_line}', 'Prob': gt_over},
                {**game_ml_features, 'Player': 'GAME_TOTAL', 'Market': 'NRFI', 'Prob': tracker['nrfi'] / SIM_GAMES}
            ])

            report.append(f"==========================================================================")
            report.append(f"[GAME OUTCOMES] {away_m['team'].upper()} @ {home_m['team'].upper()}")
            report.append(f"==========================================================================")
            report.append(f"> FULL GAME MARKETS")
            report.append(
                f"  Moneyline : Away {away_ml_prob * 100:.1f}% ({format_odds(away_ml_prob)}) | Home {home_ml_prob * 100:.1f}% ({format_odds(home_ml_prob)})")
            report.append(
                f"  Game Total ({gt_line}): Over {gt_over * 100:.1f}% ({format_odds(gt_over)}) | Under {gt_under * 100:.1f}% ({format_odds(gt_under)})")
            report.append(f"\n> 1ST INNING (NRFI / YRFI)")
            report.append(
                f"  NRFI (No Run)  : {(tracker['nrfi'] / SIM_GAMES) * 100:.1f}% ({format_odds(tracker['nrfi'] / SIM_GAMES)})\n")

        for m in game_teams:
            park_hr_val, park_avg_val = PARK_FACTORS.get(m['home_stadium'], [1.0, 1.0])
            raw_p_name, p_hand = m['opposing_pitcher'], m.get('opposing_pitcher_hand', 'R')
            p_stats = pitchers_db.get(match_player_name(raw_p_name, list(pitchers_db.keys())), league_avg_pitcher)
            p_hr9 = p_stats['CALC_HR9']
            p_archetype, _ = generate_pitcher_profile(p_hr9)
            w_boost = 1 + ((m['weather']['temp'] - 70) * 0.01) + (
                (m['weather']['wind_speed'] / 5) * 0.05 if m['weather']['wind_dir'] == 'out' else 0)

            report.append(f"==========================================================================")
            report.append(f"PLAYER PROPS: {m['team']} vs {raw_p_name} ({p_hand}HP - {p_archetype} Pitcher)")
            report.append(
                f"ENV: {m['home_stadium']} | {m['weather']['temp']}F | Wind: {m['weather']['wind_speed']}mph {m['weather']['wind_dir']}")
            report.append(f"==========================================================================")

            lineup_b_stats = [{**batters_db.get(match_player_name(b['name'], batter_keys), league_avg_batter).copy(),
                               'Hand': b['hand'], 'Name': b['name']} for b in m['lineup']]

            for order_index, b in enumerate(lineup_b_stats):
                has_platoon = (b['Hand'] == 'S') or (p_hand != b['Hand'])
                tracker = {'HR_1': 0, 'H_1': 0, 'TB_2': 0, 'R_1': 0, 'RBI_1': 0, 'HRR_2': 0}
                for _ in range(SIM_GAMES):
                    hr, hits, tb, runs, rbis, hrr, bb, sb, _ = simulate_full_game_with_archetypes(b, p_hr9, p_hand,
                                                                                                  w_boost, park_hr_val,
                                                                                                  park_avg_val,
                                                                                                  order_index)
                    if hr >= 1: tracker['HR_1'] += 1
                    if hits >= 1: tracker['H_1'] += 1
                    if tb >= 2: tracker['TB_2'] += 1
                    if runs >= 1: tracker['R_1'] += 1
                    if rbis >= 1: tracker['RBI_1'] += 1
                    if hrr >= 2: tracker['HRR_2'] += 1

                raw_hr, raw_h1, raw_tb2, raw_r1, raw_rbi1 = tracker['HR_1'] / SIM_GAMES, tracker['H_1'] / SIM_GAMES, \
                                                            tracker['TB_2'] / SIM_GAMES, tracker['R_1'] / SIM_GAMES, \
                                                            tracker['RBI_1'] / SIM_GAMES
                raw_hrr2 = tracker['HRR_2'] / SIM_GAMES

                ml_features = {
                    'Date': today_str, 'Stadium': m['home_stadium'], 'Temp': m['weather']['temp'],
                    'Wind_Speed': m['weather']['wind_speed'], 'Lineup_Spot': order_index + 1,
                    'Batter': b['Name'], 'Batter_Hand': b['Hand'], 'Batter_Archetype': b['Archetype'],
                    'Batter_xwOBA': b['xwOBA'],
                    'Pitcher': raw_p_name, 'Pitcher_Hand': p_hand, 'Pitcher_Archetype': p_archetype,
                    'Pitcher_HR9': p_hr9,
                    'Platoon_Adv': 1 if has_platoon else 0
                }

                # Apply the XGBoost Brain
                p_hr = apply_xgboost_filter(raw_hr, 'HR', ml_features)
                p_h1 = apply_xgboost_filter(raw_h1, 'Hit', ml_features)
                p_tb2 = apply_xgboost_filter(raw_tb2, 'TB', ml_features)
                p_r1 = apply_xgboost_filter(raw_r1, 'Run', ml_features)
                p_rbi1 = apply_xgboost_filter(raw_rbi1, 'RBI', ml_features)
                p_hrr2 = raw_hrr2  # Keep raw probability since AI wasn't trained on HRR

                ledger_rows.extend([
                    {**ml_features, 'Player': b['Name'], 'Market': 'HR', 'Prob': p_hr},
                    {**ml_features, 'Player': b['Name'], 'Market': 'Hit', 'Prob': p_h1},
                    {**ml_features, 'Player': b['Name'], 'Market': 'TB', 'Prob': p_tb2},
                    {**ml_features, 'Player': b['Name'], 'Market': 'Run', 'Prob': p_r1},
                    {**ml_features, 'Player': b['Name'], 'Market': 'RBI', 'Prob': p_rbi1}
                ])

                report.append(
                    f"\n> {order_index + 1}. {b['Name'].upper()} ({b['Hand']} | {b['Archetype']} | {'🔥 PLATOON ADV' if has_platoon else '❄️ NO ADV'})")
                report.append(f"  MARKET          | TRUE PROB | FAIR ODDS | TARGET RANGE (2% to 10% Edge)")
                report.append(f"  -------------------------------------------------------------------------")
                report.append(
                    f"  To Hit a HR     | {p_hr * 100:>8.1f}% | {format_odds(p_hr):>9} | {get_target_odds_range(p_hr):>16}")
                report.append(
                    f"  To Record 1+ Hit| {p_h1 * 100:>8.1f}% | {format_odds(p_h1):>9} | {get_target_odds_range(p_h1):>16}")
                report.append(
                    f"  To Record 2+ TB | {p_tb2 * 100:>8.1f}% | {format_odds(p_tb2):>9} | {get_target_odds_range(p_tb2):>16}")

    final_text = "\n".join(report)
    print(final_text)
    with open("betting_dashboard_report.txt", "w") as f:
        f.write(final_text)

    if ledger_rows:
        df_ledger = pd.DataFrame(ledger_rows)
        ledger_path = "prediction_ledger.csv"
        if os.path.exists(ledger_path):
            df_ledger.to_csv(ledger_path, mode='a', header=False, index=False)
        else:
            df_ledger.to_csv(ledger_path, index=False)

    print("\n[SUCCESS] XGBoost Dashboard exported to: betting_dashboard_report.txt")


if __name__ == "__main__":
    run_prop_market_simulation()
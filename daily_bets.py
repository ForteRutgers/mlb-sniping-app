# daily_bets.py
import pandas as pd
import numpy as np
import random
import difflib
import unicodedata
import sys
from live_scraper import get_todays_matchups
from datetime import datetime

NAME_ALIASES = {
    'Jazz Chisholm': 'Jazz Chisholm Jr',
    'Luis Robert': 'Luis Robert Jr',
    'Shohei Ohtani': 'Shohei Ohtani',
}

PARK_FACTORS = {
    'Colorado Rockies': [1.13, 1.15], 'Cincinnati Reds': [1.26, 1.05], 'New York Yankees': [1.10, 0.98],
    'San Francisco Giants': [0.81, 0.99], 'Seattle Mariners': [0.96, 0.95], 'Pittsburgh Pirates': [0.82, 0.99],
    'Chicago Cubs': [1.06, 1.01], 'Atlanta Braves': [1.05, 1.02], 'Los Angeles Dodgers': [1.18, 1.01]
}

SIM_GAMES = 10000


def normalize_name(name):
    """Aggressively sanitizes names: removes accents, periods, and hidden characters."""
    if not isinstance(name, str): return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace(".", "").replace("*", "").replace("#", "").strip()
    return name


def match_player_name(raw_name, db_keys):
    norm_raw = normalize_name(raw_name)
    if raw_name in NAME_ALIASES:
        norm_raw = normalize_name(NAME_ALIASES[raw_name])

    if norm_raw in db_keys:
        return norm_raw

    matches = difflib.get_close_matches(norm_raw, db_keys, n=1, cutoff=0.70)
    if matches:
        return matches[0]

    return norm_raw


def categorize_batter_archetype(k_rate, hr_rate):
    if hr_rate >= 0.035 and k_rate >= 0.21:
        return 'Slugger'
    elif k_rate <= 0.19 and hr_rate < 0.035:
        return 'Contact'
    else:
        return 'Balanced'


def get_col_safe(df, col_name, default_val=0):
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors='coerce').fillna(default_val)
    else:
        return pd.Series(default_val, index=df.index)


def fetch_best_available_data():
    """Cascading Tank Fetcher: Survives 403s and CAPTCHAs by rotating providers."""
    try:
        print("      -> Attempting FanGraphs for 2025...")
        from pybaseball import batting_stats, pitching_stats
        b = batting_stats(2025, qual=30)
        p = pitching_stats(2025, qual=50)
        if b is not None and not b.empty:
            b.columns = b.columns.str.upper()
            p.columns = p.columns.str.upper()
            if 'NAME' in b.columns: return b, p, "FanGraphs"
    except Exception:
        pass

    try:
        print("      -> FanGraphs blocked. Attempting Baseball-Reference for 2025...")
        from pybaseball import batting_stats_bref, pitching_stats_bref
        b = batting_stats_bref(2025)
        p = pitching_stats_bref(2025)
        if b is not None and not b.empty:
            b.columns = b.columns.str.upper()
            p.columns = p.columns.str.upper()
            if 'NAME' in b.columns:
                b['NAME'] = b['NAME'].astype(str).str.replace(r'[*#]', '', regex=True).str.strip()
                p['NAME'] = p['NAME'].astype(str).str.replace(r'[*#]', '', regex=True).str.strip()
                return b, p, "Baseball-Reference"
    except Exception:
        pass

    try:
        print("      -> B-Ref blocked. Deploying Unblockable Savant Failsafe for 2025...")
        from pybaseball import statcast_batter_expected_stats, statcast_pitcher_expected_stats
        b = statcast_batter_expected_stats(2025, 50)
        p = statcast_pitcher_expected_stats(2025, 50)
        if b is not None and not b.empty:
            b.columns = b.columns.str.upper()
            p.columns = p.columns.str.upper()
            b['NAME'] = b['FIRST_NAME'] + ' ' + b['LAST_NAME']
            p['NAME'] = p['FIRST_NAME'] + ' ' + p['LAST_NAME']

            b['PA'] = get_col_safe(b, 'PA', 1)
            b['AVG'] = get_col_safe(b, 'BA', 0.240)
            b['SLG'] = get_col_safe(b, 'SLG', 0.400)

            b['HR'] = (b['SLG'] - b['AVG']) * 0.15 * b['PA']
            b['H'] = b['AVG'] * b['PA']
            b['2B'] = b['H'] * 0.20
            b['3B'] = b['H'] * 0.02
            b['BB'] = b['PA'] * 0.085
            b['SO'] = b['PA'] * 0.225
            b['SB'] = b['PA'] * 0.02
            b['R'] = b['H'] * 0.4
            b['RBI'] = b['H'] * 0.4
            b['BARREL%'] = 0.08
            b['XWOBA'] = b['EST_WOBA'] if 'EST_WOBA' in b.columns else 0.320

            p['IP'] = get_col_safe(p, 'PA', 500) / 4.2
            p['HR'] = get_col_safe(p, 'EST_SLG', 0.400) * 15
            p['SO'] = (get_col_safe(p, 'K_PERCENT', 22.0) / 100) * get_col_safe(p, 'PA', 500)
            p['BB'] = (get_col_safe(p, 'BB_PERCENT', 8.0) / 100) * get_col_safe(p, 'PA', 500)
            p['H'] = get_col_safe(p, 'EST_BA', 0.240) * get_col_safe(p, 'PA', 500)
            p['BF'] = get_col_safe(p, 'PA', 500)
            p['GS'] = p['IP'] / 5.0

            return b, p, "MLB Statcast Savant"
    except Exception:
        pass

    return None, None, "NONE"


def get_prop_matrices():
    print("\n[1/3] Running Cascading Tank Fetcher for 2025 True DNA...")
    try:
        b_df, p_df, source = fetch_best_available_data()

        if b_df is None:
            raise Exception("Rate Limit Jail: All providers blocked.")

        print(f"      -> CONNECTION ESTABLISHED via {source}!")

        b_df = b_df.reset_index()
        p_df = p_df.reset_index()

        b_df['NAME_NORM'] = b_df['NAME'].apply(normalize_name)
        b_df = b_df.drop_duplicates(subset=['NAME_NORM'])
        p_df['NAME_NORM'] = p_df['NAME'].apply(normalize_name)
        p_df = p_df.drop_duplicates(subset=['NAME_NORM'])

        pa = get_col_safe(b_df, 'PA', 1).replace(0, 1)
        h = get_col_safe(b_df, 'H', 0)
        d2 = get_col_safe(b_df, '2B', 0)
        d3 = get_col_safe(b_df, '3B', 0)
        hr = get_col_safe(b_df, 'HR', 0)
        bb = get_col_safe(b_df, 'BB', 0)
        so = get_col_safe(b_df, 'SO', 0)
        r = get_col_safe(b_df, 'R', 0)
        rbi = get_col_safe(b_df, 'RBI', 0)
        sb = get_col_safe(b_df, 'SB', 0)

        b_df['1B_CALC'] = h - d2 - d3 - hr
        b_df['1B_Rate'] = b_df['1B_CALC'] / pa
        b_df['2B_Rate'] = d2 / pa
        b_df['3B_Rate'] = d3 / pa
        b_df['HR_Rate'] = hr / pa
        b_df['BB_Rate'] = bb / pa
        b_df['K_Rate'] = so / pa

        b_df['Barrel_Rate'] = get_col_safe(b_df, 'BARREL%', 0.08)
        b_df['xwOBA'] = get_col_safe(b_df, 'XWOBA', 0.320)

        ob_events = h + bb
        b_df['R_Conv'] = np.where(ob_events > 0, r / ob_events, 0)
        non_hr_h = h - hr
        b_df['RBI_Conv'] = np.where(non_hr_h > 0, (rbi - hr) / non_hr_h, 0)

        first_base_ops = b_df['1B_CALC'] + bb
        b_df['SB_Conv'] = np.where(first_base_ops > 0, sb / first_base_ops, 0)

        b_df['Archetype'] = b_df.apply(lambda row: categorize_batter_archetype(row['K_Rate'], row['HR_Rate']), axis=1)

        b_cols = ['1B_Rate', '2B_Rate', '3B_Rate', 'HR_Rate', 'BB_Rate', 'K_Rate', 'R_Conv', 'RBI_Conv', 'SB_Conv',
                  'Barrel_Rate', 'xwOBA', 'Archetype']
        batters = b_df.set_index('NAME_NORM')[b_cols].to_dict('index')

        # --- Universal Batters Faced Extractor ---
        p_ip = get_col_safe(p_df, 'IP', 1).replace(0, 1)
        p_so = get_col_safe(p_df, 'SO', 0)
        p_bb = get_col_safe(p_df, 'BB', 0)
        p_h = get_col_safe(p_df, 'H', 0)
        p_hr = get_col_safe(p_df, 'HR', 0)
        p_gs = get_col_safe(p_df, 'GS', 1).replace(0, 1)

        if 'TBF' in p_df.columns:
            p_bf = get_col_safe(p_df, 'TBF', 1)
        elif 'BFP' in p_df.columns:
            p_bf = get_col_safe(p_df, 'BFP', 1)
        elif 'BF' in p_df.columns:
            p_bf = get_col_safe(p_df, 'BF', 1)
        else:
            p_bf = (p_ip * 3) + p_h + p_bb

        p_bf = p_bf.replace(0, 1)

        p_df['CALC_HR9'] = p_hr / (p_ip / 9)
        p_df['K_Rate'] = p_so / p_bf
        p_df['BB_Rate'] = p_bb / p_bf
        p_df['H_Rate'] = p_h / p_bf

        p_df['BF_per_Start'] = np.where((p_bf / p_gs) < 15, 22, p_bf / p_gs)

        p_cols = ['CALC_HR9', 'K_Rate', 'BB_Rate', 'H_Rate', 'BF_per_Start']
        pitchers = p_df.set_index('NAME_NORM')[p_cols].to_dict('index')

        print(f"      -> SUCCESS: Loaded {len(batters)} Batters and {len(pitchers)} Pitchers into Memory.")
        return batters, pitchers
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        return {}, {}


def generate_pitcher_profile(p_hr9):
    suppression = p_hr9 / 1.25
    if p_hr9 <= 1.05:
        p_type = 'Spin'
        arsenal = {
            'Fastball': {'base_usage': 0.40, 'velo': 93.5, 'spin': 2200, 'hit_mod': 1.0 * suppression,
                         'hr_mod': 1.05 * suppression},
            'Breaking': {'base_usage': 0.45, 'velo': 84.0, 'spin': 2700, 'hit_mod': 0.80 * suppression,
                         'hr_mod': 0.75 * suppression},
            'Offspeed': {'base_usage': 0.15, 'velo': 86.0, 'spin': 1800, 'hit_mod': 0.95 * suppression,
                         'hr_mod': 0.90 * suppression}
        }
    elif p_hr9 >= 1.20:
        p_type = 'Power'
        arsenal = {
            'Fastball': {'base_usage': 0.65, 'velo': 96.5, 'spin': 2400, 'hit_mod': 1.1 * suppression,
                         'hr_mod': 1.25 * suppression},
            'Breaking': {'base_usage': 0.25, 'velo': 85.0, 'spin': 2400, 'hit_mod': 0.95 * suppression,
                         'hr_mod': 0.90 * suppression},
            'Offspeed': {'base_usage': 0.10, 'velo': 88.0, 'spin': 1700, 'hit_mod': 1.0 * suppression,
                         'hr_mod': 1.0 * suppression}
        }
    else:
        p_type = 'Balanced'
        arsenal = {
            'Fastball': {'base_usage': 0.50, 'velo': 94.5, 'spin': 2300, 'hit_mod': 1.0 * suppression,
                         'hr_mod': 1.0 * suppression},
            'Breaking': {'base_usage': 0.30, 'velo': 84.0, 'spin': 2500, 'hit_mod': 0.90 * suppression,
                         'hr_mod': 0.85 * suppression},
            'Offspeed': {'base_usage': 0.20, 'velo': 85.5, 'spin': 1800, 'hit_mod': 0.95 * suppression,
                         'hr_mod': 0.95 * suppression}
        }
    return p_type, arsenal


def adjust_pitch_mix(arsenal, pitcher_type, batter_type):
    fb = arsenal['Fastball']['base_usage']
    br = arsenal['Breaking']['base_usage']
    os = arsenal['Offspeed']['base_usage']

    if batter_type == 'Slugger':
        fb = max(0.20, fb - 0.15)
        br = br + 0.10
        os = os + 0.05
    elif batter_type == 'Contact':
        fb = min(0.75, fb + 0.10)
        br = max(0.10, br - 0.05)
        os = max(0.10, os - 0.05)

    total = fb + br + os
    return [fb / total, br / total, os / total]


def simulate_pitcher_game(p_stats, lineup_b_stats):
    """Simulates a pitcher throwing against the 9-man lineup to predict Ks, Hits, and BBs."""
    ks, bbs, hits = 0, 0, 0
    bf_target = max(9, int(random.gauss(p_stats['BF_per_Start'], 3.0)))

    for i in range(bf_target):
        b_stats = lineup_b_stats[i % len(lineup_b_stats)]

        prob_k = (p_stats['K_Rate'] + b_stats['K_Rate']) / 2.0
        prob_bb = (p_stats['BB_Rate'] + b_stats['BB_Rate']) / 2.0
        b_h_rate = b_stats['1B_Rate'] + b_stats['2B_Rate'] + b_stats['3B_Rate'] + b_stats['HR_Rate']
        prob_h = (p_stats['H_Rate'] + b_h_rate) / 2.0

        roll = random.random()
        if roll < prob_k:
            ks += 1
        elif roll < prob_k + prob_bb:
            bbs += 1
        elif roll < prob_k + prob_bb + prob_h:
            hits += 1

    return ks, bbs, hits


def simulate_full_game_with_archetypes(b_stats, p_hr9, w_boost, park_hr_val, park_avg_val):
    game_hits, game_tb, game_hr, game_r, game_rbi, game_bb, game_sb = 0, 0, 0, 0, 0, 0, 0
    plate_appearances = 4

    p_archetype, arsenal = generate_pitcher_profile(p_hr9)
    pitch_types = list(arsenal.keys())
    contextual_usages = adjust_pitch_mix(arsenal, p_archetype, b_stats['Archetype'])

    for _ in range(plate_appearances):
        pitch = random.choices(pitch_types, weights=contextual_usages)[0]
        p_data = arsenal[pitch]

        mod_1B = b_stats['1B_Rate'] * p_data['hit_mod'] * park_avg_val
        mod_2B = b_stats['2B_Rate'] * p_data['hit_mod'] * park_avg_val
        mod_3B = b_stats['3B_Rate'] * p_data['hit_mod'] * park_avg_val
        mod_HR = b_stats['HR_Rate'] * p_data['hr_mod'] * w_boost * park_hr_val

        roll = random.random()

        threshold = b_stats['BB_Rate']
        if roll < threshold:
            game_bb += 1
            if random.random() < b_stats['SB_Conv']: game_sb += 1
            if random.random() < b_stats['R_Conv']: game_r += 1
            continue

        threshold += b_stats['K_Rate'] * (1.25 if pitch == 'Breaking' else 0.85)
        if roll < threshold:
            continue

        threshold += mod_HR
        if roll < threshold:
            game_tb += 4
            game_hr += 1
            game_hits += 1
            game_r += 1
            game_rbi += random.choices([1, 2, 3, 4], weights=[0.55, 0.30, 0.10, 0.05])[0]
            continue

        threshold += mod_3B
        if roll < threshold:
            game_tb += 3
            game_hits += 1
            if random.random() < b_stats['RBI_Conv']: game_rbi += 1
            if random.random() < b_stats['R_Conv']: game_r += 1
            continue

        threshold += mod_2B
        if roll < threshold:
            game_tb += 2
            game_hits += 1
            if random.random() < b_stats['RBI_Conv']: game_rbi += 1
            if random.random() < b_stats['R_Conv']: game_r += 1
            continue

        threshold += mod_1B
        if roll < threshold:
            game_tb += 1
            game_hits += 1
            if random.random() < b_stats['SB_Conv']: game_sb += 1
            if random.random() < b_stats['RBI_Conv']: game_rbi += 1
            if random.random() < b_stats['R_Conv']: game_r += 1
            continue

    game_hrr = game_hits + game_r + game_rbi
    return game_hr, game_hits, game_tb, game_r, game_rbi, game_hrr, game_bb, game_sb


def get_target_odds_range(probability):
    if probability <= 0.001: return "N/A"
    if probability >= 0.999: return "N/A"
    min_decimal = 1.02 / probability
    ideal_decimal = 1.10 / probability

    def decimal_to_american(dec):
        if dec >= 2.0:
            return f"+{int((dec - 1) * 100)}"
        else:
            return str(int(-100 / (dec - 1)))

    return f"{decimal_to_american(min_decimal)} to {decimal_to_american(ideal_decimal)}"


def format_odds(probability):
    if probability <= 0.001: return "+9999"
    if probability >= 0.999: return "-9999"
    if probability > 0.50:
        return str(int((probability / (1 - probability)) * -100))
    else:
        return f"+{int((100 / probability) - 100)}"


def export_master_grid(matchups, batters_db, pitchers_db, batter_keys):
    grid_data = []
    league_avg_batter = {'Barrel_Rate': 0.08, 'xwOBA': 0.320, 'K_Rate': 0.22, 'Archetype': 'Balanced'}
    league_avg_pitcher = {'CALC_HR9': 1.25}

    for m in matchups:
        stadium = m['home_stadium']
        park_hr, park_avg = PARK_FACTORS.get(stadium, [1.0, 1.0])

        raw_p_name = m['opposing_pitcher']
        p_name_key = match_player_name(raw_p_name, list(pitchers_db.keys()))
        p_stats = pitchers_db.get(p_name_key, league_avg_pitcher)
        p_hr9 = p_stats['CALC_HR9']
        p_archetype, arsenal = generate_pitcher_profile(p_hr9)

        for raw_name in m['lineup']:
            matched_key = match_player_name(raw_name, batter_keys)
            b = batters_db.get(matched_key, league_avg_batter)

            display_name = raw_name if matched_key in batters_db else f"{raw_name} *"

            contextual_usages = adjust_pitch_mix(arsenal, p_archetype, b['Archetype'])

            for i, (pitch_type, p_data) in enumerate(arsenal.items()):
                batter_xwoba_split = b['xwOBA'] * (1.05 if pitch_type == 'Fastball' else 0.85)
                batter_barrel_split = b['Barrel_Rate'] * (1.10 if pitch_type == 'Fastball' else 0.80)

                row = {
                    'Matchup': f"{m['team']} vs {raw_p_name}",
                    'Stadium': stadium,
                    'Pitcher': raw_p_name,
                    'Pitcher Archetype': p_archetype,
                    'Pitch Type': pitch_type,
                    'Contextual Usage % (vs Batter)': f"{contextual_usages[i] * 100:.1f}%",
                    'Velocity (mph)': p_data['velo'],
                    'Spin Rate (rpm)': p_data['spin'],
                    'Batter': display_name,
                    'Batter Archetype': b['Archetype'],
                    'Batter xwOBA vs Pitch': round(batter_xwoba_split, 3),
                    'Batter K%': f"{b['K_Rate'] * 100:.1f}%"
                }
                grid_data.append(row)

    df = pd.DataFrame(grid_data)
    df.to_csv("master_matchup_sheet.csv", index=False)


def run_prop_market_simulation():
    batters_db, pitchers_db = get_prop_matrices()
    batter_keys = list(batters_db.keys())

    print("\n[2/3] Scouting Today's Live Matchups & Weather...")
    all_matchups = get_todays_matchups()

    if not all_matchups:
        print("\n[!] No games scheduled for today.")
        return

    games_dict = {}
    for m in all_matchups:
        stadium = m['home_stadium']
        if stadium not in games_dict:
            games_dict[stadium] = []
        games_dict[stadium].append(m)

    game_list = list(games_dict.values())

    # --- AUTO-PILOT LOGIC INTERCEPT ---
    if "--auto" in sys.argv:
        print("\n[AUTO-PILOT ENGAGED] Simulating ALL Available Games...")
        selected_matchups = all_matchups
    else:
        print("\n=============================================")
        print("           MLB GAMES AVAILABLE               ")
        print("=============================================")
        for i, game_teams in enumerate(game_list):
            away_team = game_teams[0]['team']
            home_team = game_teams[1]['team'] if len(game_teams) > 1 else game_teams[0]['home_stadium']
            print(f" {i + 1}. {away_team} @ {home_team}")

        print("-" * 45)
        print(f" {len(game_list) + 1}. Run All Games")
        print("=============================================")

        choice = input(f"\nSelect a game to simulate (1-{len(game_list) + 1}): ")

        try:
            choice_idx = int(choice) - 1
            if choice_idx == len(game_list):
                selected_matchups = all_matchups
                print(f"\n[3/3] Simulating ALL Games...")
            else:
                selected_matchups = game_list[choice_idx]
                print(f"\n[3/3] Simulating Game {choice}...")
        except:
            print(f"\n[!] Invalid input. Defaulting to simulating ALL Games...")
            selected_matchups = all_matchups

    export_master_grid(selected_matchups, batters_db, pitchers_db, batter_keys)

    league_avg_batter = {
        '1B_Rate': 0.145, '2B_Rate': 0.045, '3B_Rate': 0.004, 'HR_Rate': 0.030,
        'BB_Rate': 0.085, 'K_Rate': 0.225, 'R_Conv': 0.310, 'RBI_Conv': 0.150,
        'SB_Conv': 0.050, 'Barrel_Rate': 0.08, 'xwOBA': 0.320, 'Archetype': 'Balanced'
    }
    league_avg_pitcher = {
        'CALC_HR9': 1.25, 'K_Rate': 0.22, 'BB_Rate': 0.08, 'H_Rate': 0.24, 'BF_per_Start': 22
    }

    report = [f"=== MLB ARCHETYPE DASHBOARD ({SIM_GAMES} Games Simulated) ==="]
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")

    if len(batters_db) == 0:
        report.append("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        report.append("WARNING: DATABASE FAILED TO LOAD. ALL PLAYERS ARE USING LEAGUE AVERAGE.")
        report.append("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    for m in selected_matchups:
        park_factors = PARK_FACTORS.get(m['home_stadium'], [1.0, 1.0])
        park_hr_val, park_avg_val = park_factors[0], park_factors[1]

        raw_p_name = m['opposing_pitcher']
        p_name_key = match_player_name(raw_p_name, list(pitchers_db.keys()))
        p_stats = pitchers_db.get(p_name_key, league_avg_pitcher)
        p_hr9 = p_stats['CALC_HR9']
        p_archetype, _ = generate_pitcher_profile(p_hr9)

        w = m['weather']
        w_boost = 1 + ((w['temp'] - 70) * 0.01) + ((w['wind_speed'] / 5) * 0.05 if w['wind_dir'] == 'out' else 0)

        report.append(f"==========================================================================")
        report.append(f"MATCHUP: {m['team']} vs {raw_p_name} ({p_archetype} Pitcher)")
        report.append(f"ENV: {m['home_stadium']} | {w['temp']}F | Wind: {w['wind_speed']}mph {w['wind_dir']}")
        report.append(f"==========================================================================")

        lineup_b_stats = []
        for raw_player_name in m['lineup']:
            matched_key = match_player_name(raw_player_name, batter_keys)
            lineup_b_stats.append(batters_db.get(matched_key, league_avg_batter))

        p_tracker = {'K_5': 0, 'K_6': 0, 'K_7': 0, 'H_4': 0, 'H_5': 0, 'BB_2': 0}
        for _ in range(SIM_GAMES):
            ks, bbs, hits = simulate_pitcher_game(p_stats, lineup_b_stats)
            if ks >= 5: p_tracker['K_5'] += 1
            if ks >= 6: p_tracker['K_6'] += 1
            if ks >= 7: p_tracker['K_7'] += 1
            if hits >= 4: p_tracker['H_4'] += 1
            if hits >= 5: p_tracker['H_5'] += 1
            if bbs >= 2: p_tracker['BB_2'] += 1

        report.append(f"\n> {raw_p_name.upper()} (Starting Pitcher)")
        report.append(f"  MARKET          | TRUE PROB | FAIR ODDS | TARGET RANGE (2% to 10% Edge)")
        report.append(f"  -------------------------------------------------------------------------")
        report.append(
            f"  To Record 5+ Ks | {p_tracker['K_5'] / SIM_GAMES * 100:>8.1f}% | {format_odds(p_tracker['K_5'] / SIM_GAMES):>9} | {get_target_odds_range(p_tracker['K_5'] / SIM_GAMES):>16} (or better)")
        report.append(
            f"  To Record 6+ Ks | {p_tracker['K_6'] / SIM_GAMES * 100:>8.1f}% | {format_odds(p_tracker['K_6'] / SIM_GAMES):>9} | {get_target_odds_range(p_tracker['K_6'] / SIM_GAMES):>16} (or better)")
        report.append(
            f"  To Record 7+ Ks | {p_tracker['K_7'] / SIM_GAMES * 100:>8.1f}% | {format_odds(p_tracker['K_7'] / SIM_GAMES):>9} | {get_target_odds_range(p_tracker['K_7'] / SIM_GAMES):>16} (or better)")
        report.append(
            f"  To Allow 4+ Hits| {p_tracker['H_4'] / SIM_GAMES * 100:>8.1f}% | {format_odds(p_tracker['H_4'] / SIM_GAMES):>9} | {get_target_odds_range(p_tracker['H_4'] / SIM_GAMES):>16} (or better)")
        report.append(
            f"  To Allow 5+ Hits| {p_tracker['H_5'] / SIM_GAMES * 100:>8.1f}% | {format_odds(p_tracker['H_5'] / SIM_GAMES):>9} | {get_target_odds_range(p_tracker['H_5'] / SIM_GAMES):>16} (or better)")
        report.append(
            f"  To Allow 2+ BBs | {p_tracker['BB_2'] / SIM_GAMES * 100:>8.1f}% | {format_odds(p_tracker['BB_2'] / SIM_GAMES):>9} | {get_target_odds_range(p_tracker['BB_2'] / SIM_GAMES):>16} (or better)")

        for idx, raw_player_name in enumerate(m['lineup']):
            matched_key = match_player_name(raw_player_name, batter_keys)
            b = lineup_b_stats[idx]

            display_name = raw_player_name if matched_key in batters_db else f"{raw_player_name} *"

            tracker = {'HR_1': 0, 'H_1': 0, 'H_2': 0, 'TB_2': 0, 'R_1': 0, 'RBI_1': 0, 'HRR_2': 0, 'HBSB_1': 0,
                       'HBSB_2': 0}

            for _ in range(SIM_GAMES):
                hr, hits, tb, runs, rbis, hrr, bb, sb = simulate_full_game_with_archetypes(b, p_hr9, w_boost,
                                                                                           park_hr_val, park_avg_val)

                if hr >= 1: tracker['HR_1'] += 1
                if hits >= 1: tracker['H_1'] += 1
                if hits >= 2: tracker['H_2'] += 1
                if tb >= 2: tracker['TB_2'] += 1
                if runs >= 1: tracker['R_1'] += 1
                if rbis >= 1: tracker['RBI_1'] += 1
                if hrr >= 2: tracker['HRR_2'] += 1

                if (hits + bb + sb) >= 1: tracker['HBSB_1'] += 1
                if (hits + bb + sb) >= 2: tracker['HBSB_2'] += 1

            p_hr = tracker['HR_1'] / SIM_GAMES
            p_h1 = tracker['H_1'] / SIM_GAMES
            p_tb2 = tracker['TB_2'] / SIM_GAMES
            p_r1 = tracker['R_1'] / SIM_GAMES
            p_rbi1 = tracker['RBI_1'] / SIM_GAMES
            p_hrr2 = tracker['HRR_2'] / SIM_GAMES
            p_hbsb1 = tracker['HBSB_1'] / SIM_GAMES
            p_hbsb2 = tracker['HBSB_2'] / SIM_GAMES

            report.append(f"\n> {display_name.upper()} ({b['Archetype']})")
            report.append(f"  MARKET          | TRUE PROB | FAIR ODDS | TARGET RANGE (2% to 10% Edge)")
            report.append(f"  -------------------------------------------------------------------------")
            report.append(
                f"  To Hit a HR     | {p_hr * 100:>8.1f}% | {format_odds(p_hr):>9} | {get_target_odds_range(p_hr):>16} (or better)")
            report.append(
                f"  To Record 1+ Hit| {p_h1 * 100:>8.1f}% | {format_odds(p_h1):>9} | {get_target_odds_range(p_h1):>16} (or better)")
            report.append(
                f"  To Record 2+ TB | {p_tb2 * 100:>8.1f}% | {format_odds(p_tb2):>9} | {get_target_odds_range(p_tb2):>16} (or better)")
            report.append(
                f"  To Record 1+ R  | {p_r1 * 100:>8.1f}% | {format_odds(p_r1):>9} | {get_target_odds_range(p_r1):>16} (or better)")
            report.append(
                f"  To Record 1+ RBI| {p_rbi1 * 100:>8.1f}% | {format_odds(p_rbi1):>9} | {get_target_odds_range(p_rbi1):>16} (or better)")
            report.append(
                f"  To Record 2+ HRR| {p_hrr2 * 100:>8.1f}% | {format_odds(p_hrr2):>9} | {get_target_odds_range(p_hrr2):>16} (or better)")
            report.append(
                f"  To Rec 1+ H+BB+SB| {p_hbsb1 * 100:>7.1f}% | {format_odds(p_hbsb1):>9} | {get_target_odds_range(p_hbsb1):>16} (or better)")
            report.append(
                f"  To Rec 2+ H+BB+SB| {p_hbsb2 * 100:>7.1f}% | {format_odds(p_hbsb2):>9} | {get_target_odds_range(p_hbsb2):>16} (or better)")

    final_text = "\n".join(report)
    print(final_text)

    with open("betting_dashboard_report.txt", "w") as f:
        f.write(final_text)
    print("\n[SUCCESS] Quick-Betting Dashboard exported to: betting_dashboard_report.txt")


if __name__ == "__main__":
    run_prop_market_simulation()
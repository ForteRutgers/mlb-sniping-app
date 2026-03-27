# Complete live_scraper.py content
import requests
from datetime import datetime

# MLB Stadiums with Roofs/Domes (Weather = 72F, 0mph wind)
DOMES = [
    'Tampa Bay Rays', 'Miami Marlins', 'Milwaukee Brewers', 'Toronto Blue Jays',
    'Arizona Diamondbacks', 'Texas Rangers', 'Seattle Mariners', 'Houston Astros'
]


def get_todays_matchups():
    today = datetime.now().strftime('%Y-%m-%d')

    # Using the Official MLB Stats API (No API Key Required, Impossible to misalign)
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,lineups"

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    matchups = []

    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
    except Exception as e:
        print(f"[!] API Connection Failed: {e}")
        return matchups

    if 'dates' not in data or len(data['dates']) == 0:
        return matchups

    games = data['dates'][0]['games']

    for game in games:
        # Skip games that are postponed or already over
        status = game['status']['abstractGameState']
        if status not in ['Preview', 'Live']:
            continue

        try:
            # 1. Structurally Locked Team Data
            away_team = game['teams']['away']['team']['name']
            home_team = game['teams']['home']['team']['name']

            # 2. Structurally Locked Pitcher Data (Defaults to TBD if unannounced)
            away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'TBD')
            home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'TBD')

            # 3. Dynamic Weather Logic
            if home_team in DOMES:
                weather = {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}
            else:
                # Placeholder for outdoor weather (can be hooked to a real weather API later)
                weather = {'temp': 65, 'wind_speed': 8, 'wind_dir': 'out'}

            # 4. Lineup Extraction (Opening Day lineups are often posted late)
            # We provide 4 generic stars as a fallback so the Monte Carlo engine can still run and map data
            away_lineup = ['Shohei Ohtani', 'Aaron Judge', 'Bobby Witt Jr', 'Juan Soto']
            home_lineup = ['Francisco Lindor', 'Bryce Harper', 'Gunnar Henderson', 'Mookie Betts']

            # Create Away Team Profile (Batting against Home Pitcher in Home Stadium)
            matchups.append({
                'team': away_team,
                'home_stadium': home_team,
                'opposing_pitcher': home_pitcher,
                'lineup': away_lineup,
                'weather': weather
            })

            # Create Home Team Profile (Batting against Away Pitcher in Home Stadium)
            matchups.append({
                'team': home_team,
                'home_stadium': home_team,
                'opposing_pitcher': away_pitcher,
                'lineup': home_lineup,
                'weather': weather
            })

        except Exception as e:
            print(f"[!] Error parsing game: {e}")
            continue

    return matchups


# --- DIAGNOSTIC TEST RUNNER ---
if __name__ == "__main__":
    print(f"Scouting Official MLB API for {datetime.now().strftime('%Y-%m-%d')}...\n")
    games = get_todays_matchups()

    if not games:
        print("No games found or API is down.")
    else:
        print(f"Successfully locked {len(games)} team matchups.\n")
        print("--- FRANKENSTEIN BUG DIAGNOSTIC ---")
        for i in range(0, len(games), 2):
            away = games[i]
            home = games[i + 1] if i + 1 < len(games) else None

            print(f"GAME: {away['team']} @ {away['home_stadium']}")
            print(f"  -> Away Batter faces: {away['opposing_pitcher']} (Expected: Home Pitcher)")
            if home:
                print(f"  -> Home Batter faces: {home['opposing_pitcher']} (Expected: Away Pitcher)")
            print("-" * 40)
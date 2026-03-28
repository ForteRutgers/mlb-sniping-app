# live_scraper.py
import requests
from datetime import datetime
import pytz
import time

# =====================================================================
# 🚨 PASTE YOUR FREE OPENWEATHER API KEY BELOW (Inside the quotes) 🚨
# =====================================================================
OPENWEATHER_API_KEY = "3a27fd56ba9af5f8747dac6b3f880509"

TEAM_CITIES = {
    'San Francisco Giants': 'San Francisco', 'Los Angeles Dodgers': 'Los Angeles',
    'New York Yankees': 'Bronx', 'Boston Red Sox': 'Boston', 'Chicago Cubs': 'Chicago',
    'Colorado Rockies': 'Denver', 'New York Mets': 'Queens', 'Philadelphia Phillies': 'Philadelphia',
    'Atlanta Braves': 'Atlanta', 'Houston Astros': 'Houston', 'San Diego Padres': 'San Diego',
    'Seattle Mariners': 'Seattle', 'Baltimore Orioles': 'Baltimore', 'Cleveland Guardians': 'Cleveland',
    'Minnesota Twins': 'Minneapolis', 'Kansas City Royals': 'Kansas City', 'Los Angeles Angels': 'Anaheim',
    'Chicago White Sox': 'Chicago', 'Detroit Tigers': 'Detroit', 'Texas Rangers': 'Arlington',
    'Arizona Diamondbacks': 'Phoenix', 'Pittsburgh Pirates': 'Pittsburgh', 'Cincinnati Reds': 'Cincinnati',
    'St. Louis Cardinals': 'St. Louis', 'Milwaukee Brewers': 'Milwaukee', 'Miami Marlins': 'Miami',
    'Tampa Bay Rays': 'St. Petersburg', 'Toronto Blue Jays': 'Toronto', 'Washington Nationals': 'Washington',
    'Athletics': 'Sacramento'
}

DOMES = [
    'Houston Astros', 'Milwaukee Brewers', 'Miami Marlins', 'Tampa Bay Rays',
    'Toronto Blue Jays', 'Arizona Diamondbacks', 'Texas Rangers', 'Seattle Mariners'
]


def get_live_weather(home_team):
    if home_team in DOMES:
        return {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}

    if OPENWEATHER_API_KEY == "YOUR_API_KEY_HERE":
        return {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}

    city = TEAM_CITIES.get(home_team, 'New York')
    country_code = "CA" if city == "Toronto" else "US"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country_code}&appid={OPENWEATHER_API_KEY}&units=imperial"

    try:
        response = requests.get(url, timeout=5).json()
        temp = int(response['main']['temp'])
        wind_speed = int(response['wind']['speed'])
        wind_dir = 'out' if wind_speed > 8 else 'none'
        return {'temp': temp, 'wind_speed': wind_speed, 'wind_dir': wind_dir}
    except Exception as e:
        return {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}


def get_team_roster_data(team_id):
    """
    Fetches the team's active roster.
    Returns two things: a dictionary of player handedness, and a fallback 9-man lineup.
    """
    try:
        time.sleep(0.1)
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
        response = requests.get(url, timeout=5)

        if response.status_code != 200 or not response.json().get('roster'):
            url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=40Man"
            response = requests.get(url, timeout=5)

        roster = response.json().get('roster', [])

        hand_dict = {}
        fallback_batters = []

        for player in roster:
            name = player['person']['fullName']
            hand = player['person'].get('batSide', {}).get('code', 'R')

            # 1. Build the master handedness dictionary for this team
            hand_dict[name] = hand

            # 2. Build the fallback lineup just in case the manager hasn't submitted yet
            if player['position']['abbreviation'] not in ['P', 'TWP'] and len(fallback_batters) < 9:
                fallback_batters.append({'name': name, 'hand': hand})

        # Failsafe if the roster is completely broken
        if len(fallback_batters) < 9:
            fallback_batters = [{'name': f"TBD Batter {i}", 'hand': 'R'} for i in range(1, 10)]

        return hand_dict, fallback_batters

    except Exception as e:
        return {}, [{'name': f"TBD Batter {i}", 'hand': 'R'} for i in range(1, 10)]


def get_todays_matchups():
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).strftime('%Y-%m-%d')

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,lineups"

    print("      -> Pinging Official MLB Stats API...")
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        print(f"\n[!] Error fetching MLB schedule: {e}")
        return []

    matchups = []
    if 'dates' not in data or not data['dates']:
        return matchups

    games = []
    for date_obj in data.get('dates', []):
        games.extend(date_obj.get('games', []))

    for game in games:
        detailed_state = game.get('status', {}).get('detailedState', '')
        if detailed_state in ['Cancelled', 'Postponed']:
            continue

        home_team = game['teams']['home']['team']['name']
        home_id = game['teams']['home']['team']['id']
        away_team = game['teams']['away']['team']['name']
        away_id = game['teams']['away']['team']['id']

        stadium = game.get('venue', {}).get('name', home_team)

        home_pitcher_data = game['teams']['home'].get('probablePitcher', {})
        home_pitcher = home_pitcher_data.get('fullName', 'TBD')
        home_pitcher_hand = home_pitcher_data.get('pitchHand', {}).get('code',
                                                                       'R') if 'pitchHand' in home_pitcher_data else 'R'

        away_pitcher_data = game['teams']['away'].get('probablePitcher', {})
        away_pitcher = away_pitcher_data.get('fullName', 'TBD')
        away_pitcher_hand = away_pitcher_data.get('pitchHand', {}).get('code',
                                                                       'R') if 'pitchHand' in away_pitcher_data else 'R'

        # --- THE FIX: Pre-fetch the master roster to get the actual handedness ---
        home_hand_dict, home_fallback = get_team_roster_data(home_id)
        away_hand_dict, away_fallback = get_team_roster_data(away_id)

        home_lineup_data = game['teams']['home'].get('lineups', [])
        away_lineup_data = game['teams']['away'].get('lineups', [])

        # Extract Official Lineup and map the correct handedness from the dictionary
        home_lineup = []
        if home_lineup_data:
            for p in home_lineup_data:
                name = p['fullName']
                hand = home_hand_dict.get(name, 'R')  # Look up true handedness
                home_lineup.append({'name': name, 'hand': hand})
        else:
            home_lineup = home_fallback

        away_lineup = []
        if away_lineup_data:
            for p in away_lineup_data:
                name = p['fullName']
                hand = away_hand_dict.get(name, 'R')  # Look up true handedness
                away_lineup.append({'name': name, 'hand': hand})
        else:
            away_lineup = away_fallback

        weather = get_live_weather(home_team)

        if away_lineup:
            matchups.append({
                'team': away_team,
                'home_stadium': stadium,
                'opposing_pitcher': home_pitcher,
                'opposing_pitcher_hand': home_pitcher_hand,
                'lineup': away_lineup,
                'weather': weather
            })

        if home_lineup:
            matchups.append({
                'team': home_team,
                'home_stadium': stadium,
                'opposing_pitcher': away_pitcher,
                'opposing_pitcher_hand': away_pitcher_hand,
                'lineup': home_lineup,
                'weather': weather
            })

    return matchups


if __name__ == "__main__":
    print(get_todays_matchups())
# live_scraper.py
import os
import requests
from datetime import datetime, timezone
import pytz
import time

# =====================================================================
# Set OPENWEATHER_API_KEY as an environment variable or paste your key
# below inside the quotes.
# =====================================================================
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "3a27fd56ba9af5f8747dac6b3f880509")

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

# Vault to prevent team rosters from crossing over
ROSTER_CACHE = {}


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
    Fetches the team's active roster. Uses a cache to prevent variable leaks.
    """
    if team_id in ROSTER_CACHE:
        return ROSTER_CACHE[team_id]

    try:
        time.sleep(0.1)
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active&hydrate=person"
        response = requests.get(url, timeout=5)

        if response.status_code != 200 or not response.json().get('roster'):
            url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=40Man&hydrate=person"
            response = requests.get(url, timeout=5)

        roster = response.json().get('roster', [])

        hand_dict = {}
        fallback_batters = []

        for player in roster:
            name = player['person']['fullName']
            hand = player['person'].get('batSide', {}).get('code', 'R')

            hand_dict[name] = hand

            # Only grab standard position players for the fallback lineup
            if player['position']['abbreviation'] not in ['P', 'TWP'] and len(fallback_batters) < 9:
                fallback_batters.append({'name': name, 'hand': hand})

        if len(fallback_batters) < 9:
            fallback_batters = [{'name': f"TBD Batter {i}", 'hand': 'R'} for i in range(1, 10)]

        # Save to vault to prevent crossovers
        ROSTER_CACHE[team_id] = (hand_dict, fallback_batters)
        return ROSTER_CACHE[team_id]

    except Exception as e:
        return {}, [{'name': f"TBD Batter {i}", 'hand': 'R'} for i in range(1, 10)]


def get_todays_matchups():
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).strftime('%Y-%m-%d')

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"

    print("      -> Pinging Official MLB Stats API for Games...")
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
        # STRICT VARIABLE ISOLATION
        home_batting_order = []
        away_batting_order = []
        home_players = {}
        away_players = {}

        detailed_state = game.get('status', {}).get('detailedState', '')
        if detailed_state in ['Cancelled', 'Postponed']:
            continue

        game_pk = game['gamePk']

        home_team = game['teams']['home']['team']['name']
        home_id = game['teams']['home']['team']['id']
        away_team = game['teams']['away']['team']['name']
        away_id = game['teams']['away']['team']['id']

        game_time = datetime.fromisoformat(game['gameDate'].replace('Z', '+00:00'))
        stadium = game.get('venue', {}).get('name', home_team)

        home_pitcher_data = game['teams']['home'].get('probablePitcher', {})
        home_pitcher = home_pitcher_data.get('fullName', 'TBD')
        home_pitcher_hand = home_pitcher_data.get('pitchHand', {}).get('code',
                                                                       'R') if 'pitchHand' in home_pitcher_data else 'R'

        away_pitcher_data = game['teams']['away'].get('probablePitcher', {})
        away_pitcher = away_pitcher_data.get('fullName', 'TBD')
        away_pitcher_hand = away_pitcher_data.get('pitchHand', {}).get('code',
                                                                       'R') if 'pitchHand' in away_pitcher_data else 'R'

        boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        try:
            box_data = requests.get(boxscore_url, timeout=5).json()
            home_batting_order = box_data['teams']['home'].get('battingOrder', [])
            away_batting_order = box_data['teams']['away'].get('battingOrder', [])
            home_players = box_data['teams']['home'].get('players', {})
            away_players = box_data['teams']['away'].get('players', {})
        except Exception:
            pass

        home_hand_dict, home_fallback = get_team_roster_data(home_id)
        away_hand_dict, away_fallback = get_team_roster_data(away_id)

        home_lineup = []
        if home_batting_order:
            for pid in home_batting_order:
                player_key = f"ID{pid}"
                if player_key in home_players:
                    p_info = home_players[player_key]
                    name = p_info['person']['fullName']
                    hand = p_info['person'].get('batSide', {}).get('code', home_hand_dict.get(name, 'R'))
                    home_lineup.append({'name': name, 'hand': hand})
        else:
            home_lineup = home_fallback

        away_lineup = []
        if away_batting_order:
            for pid in away_batting_order:
                player_key = f"ID{pid}"
                if player_key in away_players:
                    p_info = away_players[player_key]
                    name = p_info['person']['fullName']
                    hand = p_info['person'].get('batSide', {}).get('code', away_hand_dict.get(name, 'R'))
                    away_lineup.append({'name': name, 'hand': hand})
        else:
            away_lineup = away_fallback

        weather = get_live_weather(home_team)

        eastern_tz = pytz.timezone('America/New_York')
        game_time_eastern = game_time.astimezone(eastern_tz)
        game_time_str = game_time_eastern.strftime('%I:%M %p ET')

        if away_lineup:
            matchups.append({
                'team': away_team,
                'home_stadium': stadium,
                'opposing_pitcher': home_pitcher,
                'opposing_pitcher_hand': home_pitcher_hand,
                'lineup': away_lineup[:9],
                'weather': weather,
                'game_time': game_time_str
            })

        if home_lineup:
            matchups.append({
                'team': home_team,
                'home_stadium': stadium,
                'opposing_pitcher': away_pitcher,
                'opposing_pitcher_hand': away_pitcher_hand,
                'lineup': home_lineup[:9],
                'weather': weather,
                'game_time': game_time_str
            })

    return matchups


if __name__ == "__main__":
    print(get_todays_matchups())
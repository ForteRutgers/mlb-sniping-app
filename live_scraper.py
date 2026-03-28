# live_scraper.py
import requests
from datetime import datetime
import pytz
import time

# =====================================================================
# 🚨 PASTE YOUR FREE OPENWEATHER API KEY BELOW (Inside the quotes) 🚨
# =====================================================================
OPENWEATHER_API_KEY = "3a27fd56ba9af5f8747dac6b3f880509"

# Mapping teams to their cities for the weather API
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

# Stadiums with roofs - lock weather to 72 degrees and no wind
DOMES = [
    'Houston Astros', 'Milwaukee Brewers', 'Miami Marlins', 'Tampa Bay Rays',
    'Toronto Blue Jays', 'Arizona Diamondbacks', 'Texas Rangers', 'Seattle Mariners'
]


def get_live_weather(home_team):
    """Fetches real-time temperature and wind data, accounting for dome stadiums."""
    if home_team in DOMES:
        return {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}

    if OPENWEATHER_API_KEY == "YOUR_API_KEY_HERE":
        return {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}

    city = TEAM_CITIES.get(home_team, 'New York')
    # Handle the Canadian team explicitly
    country_code = "CA" if city == "Toronto" else "US"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country_code}&appid={OPENWEATHER_API_KEY}&units=imperial"

    try:
        response = requests.get(url, timeout=5).json()
        temp = int(response['main']['temp'])
        wind_speed = int(response['wind']['speed'])

        # simplified prevailing wind direction
        wind_dir = 'out' if wind_speed > 8 else 'none'

        return {'temp': temp, 'wind_speed': wind_speed, 'wind_dir': wind_dir}
    except Exception as e:
        print(f"    [!] Weather fetch failed for {city}, defaulting to 72F. Error: {e}")
        return {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}


def get_team_roster_fallback(team_id):
    """Fetches active roster to use as a placeholder if official lineup isn't posted."""
    try:
        time.sleep(0.1)
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
        response = requests.get(url, timeout=5)

        if response.status_code != 200 or not response.json().get('roster'):
            url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=40Man"
            response = requests.get(url, timeout=5)

        data = response.json()
        roster = data.get('roster', [])

        batters = []
        for player in roster:
            if player['position']['abbreviation'] not in ['P', 'TWP']:
                name = player['person']['fullName']
                hand = 'R'
                if 'batSide' in player['person']:
                    hand = player['person']['batSide'].get('code', 'R')
                batters.append({'name': name, 'hand': hand})

            if len(batters) >= 9:
                break

        if len(batters) == 9:
            return batters
        else:
            return [{'name': f"TBD Batter {i}", 'hand': 'R'} for i in range(1, 10)]

    except Exception as e:
        return [{'name': f"TBD Batter {i}", 'hand': 'R'} for i in range(1, 10)]


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

        home_lineup_data = game['teams']['home'].get('lineups', [])
        away_lineup_data = game['teams']['away'].get('lineups', [])

        home_lineup = []
        for p in home_lineup_data:
            hand = p.get('batSide', {}).get('code', 'R') if 'batSide' in p else 'R'
            home_lineup.append({'name': p['fullName'], 'hand': hand})

        away_lineup = []
        for p in away_lineup_data:
            hand = p.get('batSide', {}).get('code', 'R') if 'batSide' in p else 'R'
            away_lineup.append({'name': p['fullName'], 'hand': hand})

        if not home_lineup:
            home_lineup = get_team_roster_fallback(home_id)
        if not away_lineup:
            away_lineup = get_team_roster_fallback(away_id)

        # PING THE LIVE WEATHER HERE
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
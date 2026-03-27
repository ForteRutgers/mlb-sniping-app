# live_scraper.py
import requests
from datetime import datetime
import pytz
import time


def get_team_roster_fallback(team_id):
    """Fetches active roster to use as a placeholder if official lineup isn't posted."""
    try:
        # Prevent MLB API Rate Limiting by pausing for a fraction of a second
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
                batters.append(player['person']['fullName'])
            if len(batters) >= 9:
                break

        if len(batters) == 9:
            return batters
        else:
            return [f"TBD Batter {i}" for i in range(1, 10)]

    except Exception as e:
        return [f"TBD Batter {i}" for i in range(1, 10)]


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

    # Grab all games across all possible date arrays the MLB API might return
    games = []
    for date_obj in data.get('dates', []):
        games.extend(date_obj.get('games', []))

    for game in games:
        # CRITICAL FIX: Look at detailedState instead of abbreviations to prevent skipping "Pre-Game" matches
        detailed_state = game.get('status', {}).get('detailedState', '')
        if detailed_state in ['Cancelled', 'Postponed']:
            continue

        home_team = game['teams']['home']['team']['name']
        home_id = game['teams']['home']['team']['id']
        away_team = game['teams']['away']['team']['name']
        away_id = game['teams']['away']['team']['id']

        stadium = game.get('venue', {}).get('name', home_team)

        home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'TBD')
        away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'TBD')

        home_lineup_data = game['teams']['home'].get('lineups', [])
        away_lineup_data = game['teams']['away'].get('lineups', [])

        home_lineup = [player['fullName'] for player in home_lineup_data]
        away_lineup = [player['fullName'] for player in away_lineup_data]

        if not home_lineup:
            home_lineup = get_team_roster_fallback(home_id)
        if not away_lineup:
            away_lineup = get_team_roster_fallback(away_id)

        weather = {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}

        if away_lineup:
            matchups.append({
                'team': away_team,
                'home_stadium': stadium,
                'opposing_pitcher': home_pitcher,
                'lineup': away_lineup,
                'weather': weather
            })

        if home_lineup:
            matchups.append({
                'team': home_team,
                'home_stadium': stadium,
                'opposing_pitcher': away_pitcher,
                'lineup': home_lineup,
                'weather': weather
            })

    return matchups


if __name__ == "__main__":
    print(get_todays_matchups())
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

        # Try to get the active roster first
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
        response = requests.get(url, timeout=5)

        # If the active roster is empty (common right before Opening Day), try the 40-Man
        if response.status_code != 200 or not response.json().get('roster'):
            url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=40Man"
            response = requests.get(url, timeout=5)

        data = response.json()
        roster = data.get('roster', [])

        batters = []
        for player in roster:
            # Filter out Pitchers (P) and Two-Way Players (TWP) if they are pitching
            if player['position']['abbreviation'] not in ['P', 'TWP']:
                batters.append(player['person']['fullName'])
            # Stop once we have 9 position players
            if len(batters) >= 9:
                break

        # If we successfully found 9 players, return them
        if len(batters) == 9:
            return batters
        else:
            # Failsafe: Return generic names so the game stays on the menu
            return [f"TBD Batter {i}" for i in range(1, 10)]

    except Exception as e:
        print(f"    [!] Roster fallback failed for team {team_id}: {e}")
        return [f"TBD Batter {i}" for i in range(1, 10)]


def get_todays_matchups():
    """
    Pings the official MLB API for today's schedule, probable pitchers, and official lineups.
    """
    # Force US/Eastern time so it doesn't break based on your server's local clock
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).strftime('%Y-%m-%d')

    # Hydrate pulls everything in one single, safe request
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,lineups"

    print("      -> Pinging Official MLB Stats API...")
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        print(f"\n[!] Error fetching MLB schedule: {e}")
        return []

    matchups = []

    # If there are no games today
    if 'dates' not in data or not data['dates']:
        return matchups

    games = data['dates'][0].get('games', [])

    for game in games:
        # Skip games that are Postponed (P) or Cancelled (C)
        if game['status']['statusCode'] in ['P', 'C']:
            continue

        home_team = game['teams']['home']['team']['name']
        home_id = game['teams']['home']['team']['id']
        away_team = game['teams']['away']['team']['name']
        away_id = game['teams']['away']['team']['id']

        stadium = game.get('venue', {}).get('name', home_team)

        home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'TBD')
        away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'TBD')

        # Extract Official Lineups natively from the schedule hydrate
        home_lineup_data = game['teams']['home'].get('lineups', [])
        away_lineup_data = game['teams']['away'].get('lineups', [])

        home_lineup = [player['fullName'] for player in home_lineup_data]
        away_lineup = [player['fullName'] for player in away_lineup_data]

        # If the manager hasn't submitted yet, use the robust API-friendly fallback
        if not home_lineup:
            home_lineup = get_team_roster_fallback(home_id)
        if not away_lineup:
            away_lineup = get_team_roster_fallback(away_id)

        weather = {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}

        # Append Away Batters vs Home Pitcher
        matchups.append({
            'team': away_team,
            'home_stadium': stadium,
            'opposing_pitcher': home_pitcher,
            'lineup': away_lineup,
            'weather': weather
        })

        # Append Home Batters vs Away Pitcher
        matchups.append({
            'team': home_team,
            'home_stadium': stadium,
            'opposing_pitcher': away_pitcher,
            'lineup': home_lineup,
            'weather': weather
        })

    return matchups


if __name__ == "__main__":
    # Quick test if you run this file directly
    print(get_todays_matchups())
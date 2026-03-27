# live_scraper.py
import requests
from datetime import datetime
import pytz


def get_official_lineup(game_pk, team_type):
    """Hits the live boxscore endpoint to get the locked-in 1-9 batting order."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        data = requests.get(url).json()
        team_data = data['teams'][team_type]

        # battingOrder contains the official 1-9 player IDs submitted by the manager
        batting_order = team_data.get('battingOrder', [])
        players_dict = team_data.get('players', {})

        lineup = []
        for player_id in batting_order:
            player_key = f"ID{player_id}"
            if player_key in players_dict:
                lineup.append(players_dict[player_key]['person']['fullName'])

        return lineup
    except Exception as e:
        return []


def get_team_roster_fallback(team_id):
    """
    Failsafe: If official lineups aren't posted yet, grab 9 position players
    from the actual team's active roster so the simulation doesn't use fake data.
    """
    try:
        # CRITICAL FIX: Added ?rosterType=active. Without this, the MLB API returns empty data!
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
        data = requests.get(url).json()
        roster = data.get('roster', [])

        batters = []
        for player in roster:
            # Filter out Pitchers (P) and Two-Way Players (TWP) if they are pitching
            if player['position']['abbreviation'] not in ['P', 'TWP']:
                batters.append(player['person']['fullName'])
            if len(batters) >= 9:
                break
        return batters
    except Exception as e:
        print(f"    [!] Roster fallback failed for team {team_id}: {e}")
        return []


def get_todays_matchups():
    """
    Pings the official MLB API for today's schedule, probable pitchers, and official lineups.
    """
    # Force US/Eastern time so it doesn't break based on your server's local clock
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).strftime('%Y-%m-%d')

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"

    print("      -> Pinging Official MLB Stats API...")
    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print(f"\n[!] Error fetching MLB schedule: {e}")
        return []

    matchups = []

    # If there are no games today (off-season or rainouts)
    if 'dates' not in data or not data['dates']:
        return matchups

    games = data['dates'][0].get('games', [])

    for game in games:
        # Skip games that are Postponed (P) or Cancelled (C)
        if game['status']['statusCode'] in ['P', 'C']:
            continue

        game_pk = game['gamePk']
        home_team = game['teams']['home']['team']['name']
        home_id = game['teams']['home']['team']['id']
        away_team = game['teams']['away']['team']['name']
        away_id = game['teams']['away']['team']['id']

        # Get the actual stadium the game is being played in
        stadium = game.get('venue', {}).get('name', home_team)

        # Extract Probable Pitchers
        home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'TBD')
        away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'TBD')

        # 1. Try to get the official manager-submitted lineups from the Boxscore
        home_lineup = get_official_lineup(game_pk, 'home')
        away_lineup = get_official_lineup(game_pk, 'away')

        # 2. If the manager hasn't submitted yet (empty list), use the active roster fallback
        if not home_lineup:
            home_lineup = get_team_roster_fallback(home_id)
        if not away_lineup:
            away_lineup = get_team_roster_fallback(away_id)

        # Default weather placeholder (can be linked to a Weather API later)
        weather = {'temp': 72, 'wind_speed': 0, 'wind_dir': 'none'}

        # Construct Away Batters vs Home Pitcher
        if away_lineup:
            matchups.append({
                'team': away_team,
                'home_stadium': stadium,
                'opposing_pitcher': home_pitcher,
                'lineup': away_lineup,
                'weather': weather
            })

        # Construct Home Batters vs Away Pitcher
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
    # Quick test if you run this file directly
    print(get_todays_matchups())
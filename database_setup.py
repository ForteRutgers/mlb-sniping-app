# database_setup.py
import sqlite3
import os

def initialize_database():
    db_path = 'mlb_predictions.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f" -> Initializing SQLite Database at: {os.path.abspath(db_path)}")

    # 1. Create Table for Game Markets (ML, Over/Under, NRFI)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            game_date TEXT,
            away_team TEXT,
            home_team TEXT,
            stadium TEXT,
            market TEXT,
            probability REAL,
            fair_odds TEXT,
            processed INTEGER DEFAULT 0
        )
    ''')

    # 2. Create Table for Player Props (Hits, HRs, etc.)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            market TEXT,
            probability REAL,
            fair_odds TEXT,
            lineup_spot INTEGER,
            is_platoon_adv INTEGER,
            processed INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()
    print(" [SUCCESS] Database tables created and ready for data.")

if __name__ == "__main__":
    initialize_database()
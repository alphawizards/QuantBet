"""
Seed Test Data into Test Database.

Loads fixture data from tests/fixtures/ into the test database.
"""

import json
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def seed_test_data():
    """Load all fixture data into test database."""
    
    # Get database URL
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    print(f"🌱 Seeding test database...")
    
    engine = create_engine(db_url)
    
    # Load fixtures
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    
    # 1. Load sample games
    games_file = fixtures_dir / "games_2024_sample.json"
    if games_file.exists():
        with open(games_file, 'r') as f:
            games_data = json.load(f)
        
        print(f"   Loading {len(games_data['games'])} sample games...")
        
        with engine.begin() as conn:
            for game in games_data['games']:
                # Insert into games table (adjust schema as needed)
                conn.execute(text("""
                    INSERT INTO games (
                        game_id, home_team, away_team, game_date,
                        home_score, away_score, outcome
                    ) VALUES (
                        :game_id, :home_team, :away_team, :game_date,
                        :home_score, :away_score, :outcome
                    )
                    ON CONFLICT (game_id) DO NOTHING
                """), game)
        
        print(f"   ✅ Loaded {len(games_data['games'])} games")
    
    # 2. Load calibration test data (for prediction_logs table)
    cal_file = fixtures_dir / "calibration_perfect.json"
    if cal_file.exists():
        with open(cal_file, 'r') as f:
            cal_data = json.load(f)
        
        print(f"   Loading {len(cal_data['predictions'])} calibration predictions...")
        # This would go into prediction_logs table when that's created
        print(f"   ⚠️  Skipping calibration data (table not yet created)")
    
    # 3. Load sample bets (for tracked_bets table - Phase 1)
    bets_file = fixtures_dir / "user_bets_sample.json"
    if bets_file.exists():
        with open(bets_file, 'r') as f:
            bets_data = json.load(f)
        
        print(f"   Loading {len(bets_data['bets'])} sample bets...")
        # This would go into tracked_bets table in Phase 1
        print(f"   ⚠️  Skipping bet data (table not yet created)")
    
    print("✅ Test data seeding complete!")
    
    # Verify
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM games"))
        count = result.fetchone()[0]
        print(f"   Total games in database: {count}")

if __name__ == "__main__":
    seed_test_data()

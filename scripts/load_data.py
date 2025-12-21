"""
Data loader for NBL historical data from AusSportsBetting xlsx.

This script loads the nbl.xlsx file into the PostgreSQL database,
mapping the Excel columns to our schema.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://quantbet:quantbet_secret@localhost:5432/quantbet"
)


def get_engine():
    """Create SQLAlchemy engine."""
    return create_engine(DATABASE_URL)


def load_nbl_data(
    xlsx_path: str = "data/nbl.xlsx",
    sheet_name: str = "Data",
    header_row: int = 1
) -> pd.DataFrame:
    """
    Load NBL data from Excel file.
    
    Args:
        xlsx_path: Path to the xlsx file
        sheet_name: Name of the sheet to read
        header_row: Row number containing headers (0-indexed)
    
    Returns:
        DataFrame with game and odds data
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_row)
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add season column (NBL season runs Oct-Mar)
    df['Season'] = df['Date'].apply(lambda d: 
        f"{d.year-1}-{str(d.year)[2:]}" if d.month < 7 
        else f"{d.year}-{str(d.year+1)[2:]}"
    )
    
    return df


def insert_games_and_odds(
    df: pd.DataFrame,
    engine,
    batch_size: int = 100
) -> dict:
    """
    Insert games and odds into PostgreSQL.
    
    Uses raw SQL for simplicity with our existing schema.
    Maps Excel data to games + odds_history tables.
    
    Returns:
        Dict with counts of inserted records
    """
    # Create a simple games table if the complex one doesn't work
    create_simple_tables = """
    -- Drop and recreate simpler tables for the xlsx data
    DROP TABLE IF EXISTS nbl_odds CASCADE;
    DROP TABLE IF EXISTS nbl_games CASCADE;
    
    CREATE TABLE IF NOT EXISTS nbl_games (
        game_id SERIAL PRIMARY KEY,
        game_date DATE NOT NULL,
        season VARCHAR(10) NOT NULL,
        home_team VARCHAR(100) NOT NULL,
        away_team VARCHAR(100) NOT NULL,
        home_score INTEGER,
        away_score INTEGER,
        is_playoff BOOLEAN DEFAULT FALSE,
        is_overtime BOOLEAN DEFAULT FALSE,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS nbl_odds (
        odds_id SERIAL PRIMARY KEY,
        game_id INTEGER REFERENCES nbl_games(game_id),
        
        -- Moneyline odds
        home_odds DECIMAL(6,3),
        away_odds DECIMAL(6,3),
        home_odds_open DECIMAL(6,3),
        home_odds_close DECIMAL(6,3),
        away_odds_open DECIMAL(6,3),
        away_odds_close DECIMAL(6,3),
        
        -- Spread
        home_line_open DECIMAL(5,2),
        home_line_close DECIMAL(5,2),
        home_line_odds_open DECIMAL(6,3),
        home_line_odds_close DECIMAL(6,3),
        
        -- Total
        total_open DECIMAL(5,1),
        total_close DECIMAL(5,1),
        total_over_odds_open DECIMAL(6,3),
        total_over_odds_close DECIMAL(6,3),
        total_under_odds_open DECIMAL(6,3),
        total_under_odds_close DECIMAL(6,3),
        
        bookmakers_surveyed INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_nbl_games_date ON nbl_games(game_date);
    CREATE INDEX IF NOT EXISTS idx_nbl_games_season ON nbl_games(season);
    CREATE INDEX IF NOT EXISTS idx_nbl_games_teams ON nbl_games(home_team, away_team);
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_simple_tables))
        conn.commit()
    
    games_inserted = 0
    odds_inserted = 0
    
    with engine.connect() as conn:
        for _, row in df.iterrows():
            # Insert game
            game_sql = """
            INSERT INTO nbl_games (game_date, season, home_team, away_team, 
                                   home_score, away_score, is_playoff, is_overtime, notes)
            VALUES (:game_date, :season, :home_team, :away_team, 
                    :home_score, :away_score, :is_playoff, :is_overtime, :notes)
            RETURNING game_id
            """
            
            is_playoff = str(row.get('Play Off Game?', '')).upper() == 'Y'
            is_overtime = str(row.get('Over Time?', '')).upper() == 'Y'
            
            result = conn.execute(text(game_sql), {
                'game_date': row['Date'].date(),
                'season': row['Season'],
                'home_team': row['Home Team'],
                'away_team': row['Away Team'],
                'home_score': int(row['Home Score']) if pd.notna(row['Home Score']) else None,
                'away_score': int(row['Away Score']) if pd.notna(row['Away Score']) else None,
                'is_playoff': is_playoff,
                'is_overtime': is_overtime,
                'notes': row.get('Notes') if pd.notna(row.get('Notes')) else None,
            })
            
            game_id = result.fetchone()[0]
            games_inserted += 1
            
            # Insert odds
            odds_sql = """
            INSERT INTO nbl_odds (game_id, home_odds, away_odds,
                                  home_odds_open, home_odds_close,
                                  away_odds_open, away_odds_close,
                                  home_line_open, home_line_close,
                                  home_line_odds_open, home_line_odds_close,
                                  total_open, total_close,
                                  total_over_odds_open, total_over_odds_close,
                                  total_under_odds_open, total_under_odds_close,
                                  bookmakers_surveyed)
            VALUES (:game_id, :home_odds, :away_odds,
                    :home_odds_open, :home_odds_close,
                    :away_odds_open, :away_odds_close,
                    :home_line_open, :home_line_close,
                    :home_line_odds_open, :home_line_odds_close,
                    :total_open, :total_close,
                    :total_over_odds_open, :total_over_odds_close,
                    :total_under_odds_open, :total_under_odds_close,
                    :bookmakers_surveyed)
            """
            
            conn.execute(text(odds_sql), {
                'game_id': game_id,
                'home_odds': row.get('Home Odds') if pd.notna(row.get('Home Odds')) else None,
                'away_odds': row.get('Away Odds') if pd.notna(row.get('Away Odds')) else None,
                'home_odds_open': row.get('Home Odds Open') if pd.notna(row.get('Home Odds Open')) else None,
                'home_odds_close': row.get('Home Odds Close') if pd.notna(row.get('Home Odds Close')) else None,
                'away_odds_open': row.get('Away Odds Open') if pd.notna(row.get('Away Odds Open')) else None,
                'away_odds_close': row.get('Away Odds Close') if pd.notna(row.get('Away Odds Close')) else None,
                'home_line_open': row.get('Home Line Open') if pd.notna(row.get('Home Line Open')) else None,
                'home_line_close': row.get('Home Line Close') if pd.notna(row.get('Home Line Close')) else None,
                'home_line_odds_open': row.get('Home Line Odds Open') if pd.notna(row.get('Home Line Odds Open')) else None,
                'home_line_odds_close': row.get('Home Line Odds Close') if pd.notna(row.get('Home Line Odds Close')) else None,
                'total_open': row.get('Total Score Open') if pd.notna(row.get('Total Score Open')) else None,
                'total_close': row.get('Total Score Close') if pd.notna(row.get('Total Score Close')) else None,
                'total_over_odds_open': row.get('Total Score Over Open') if pd.notna(row.get('Total Score Over Open')) else None,
                'total_over_odds_close': row.get('Total Score Over Close') if pd.notna(row.get('Total Score Over Close')) else None,
                'total_under_odds_open': row.get('Total Score Under Open') if pd.notna(row.get('Total Score Under Open')) else None,
                'total_under_odds_close': row.get('Total Score Under Close') if pd.notna(row.get('Total Score Under Close')) else None,
                'bookmakers_surveyed': int(row.get('Bookmakers Surveyed', 0)) if pd.notna(row.get('Bookmakers Surveyed')) else None,
            })
            odds_inserted += 1
            
            if games_inserted % 100 == 0:
                print(f"Inserted {games_inserted} games...")
                conn.commit()
        
        conn.commit()
    
    return {
        'games_inserted': games_inserted,
        'odds_inserted': odds_inserted
    }


def main():
    """Main entry point for data loading."""
    print("=" * 60)
    print("QuantBet NBL Data Loader")
    print("=" * 60)
    
    # Check for xlsx file
    xlsx_path = Path("data/nbl.xlsx")
    if not xlsx_path.exists():
        print(f"ERROR: {xlsx_path} not found!")
        print("Please place your nbl.xlsx file in the data/ folder.")
        return
    
    print(f"\n1. Loading data from {xlsx_path}...")
    df = load_nbl_data(str(xlsx_path))
    print(f"   Loaded {len(df)} games from {df['Season'].min()} to {df['Season'].max()}")
    
    print("\n2. Connecting to PostgreSQL...")
    engine = get_engine()
    
    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("   Connection successful!")
    except Exception as e:
        print(f"   ERROR: Could not connect to database: {e}")
        print("   Make sure Docker is running: docker-compose up -d")
        return
    
    print("\n3. Inserting data into database...")
    stats = insert_games_and_odds(df, engine)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"   Games inserted: {stats['games_inserted']}")
    print(f"   Odds inserted:  {stats['odds_inserted']}")
    print("=" * 60)
    
    # Show sample query
    print("\nSample data (last 5 games):")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT g.game_date, g.home_team, g.away_team, 
                   g.home_score, g.away_score,
                   o.home_odds_close, o.away_odds_close
            FROM nbl_games g
            JOIN nbl_odds o ON g.game_id = o.game_id
            ORDER BY g.game_date DESC
            LIMIT 5
        """))
        for row in result:
            print(f"  {row}")


if __name__ == "__main__":
    main()

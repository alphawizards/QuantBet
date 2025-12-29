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
from sqlalchemy import create_engine, text, select
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from src.core.database.models import Base, Game, OddsHistory, Team, LeagueType, GameStatus, BetType, BetOutcome
from src.collectors.integration import TEAM_NAME_MAP

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


def get_or_create_team(session: Session, team_code: str, team_name: str) -> Team:
    """Get existing team or create new one."""
    team = session.scalar(select(Team).where(Team.team_code == team_code))
    if not team:
        team = Team(
            team_code=team_code,
            team_name=team_name,
            league=LeagueType.NBL,
            city="Unknown" # Would need a map for cities
        )
        session.add(team)
        session.flush() # flush to get ID
    return team

def normalize_team_name(name: str) -> str:
    """Normalize team name using shared map."""
    return TEAM_NAME_MAP.get(name, "UNKNOWN")


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
    Insert games and odds into PostgreSQL using SQLAlchemy ORM.
    
    Populates the 'games' and 'odds_history' tables.
    
    Returns:
        Dict with counts of inserted records
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    games_inserted = 0
    odds_inserted = 0
    
    try:
        # Cache teams to avoid repeated lookups
        team_cache = {}

        for _, row in df.iterrows():
            # Resolve Teams
            home_name_raw = row['Home Team']
            away_name_raw = row['Away Team']
            
            home_code = normalize_team_name(home_name_raw)
            away_code = normalize_team_name(away_name_raw)
            
            if home_code == "UNKNOWN" or away_code == "UNKNOWN":
                print(f"Skipping game due to unknown team: {home_name_raw} vs {away_name_raw}")
                continue

            if home_code not in team_cache:
                team_cache[home_code] = get_or_create_team(session, home_code, home_name_raw)
            if away_code not in team_cache:
                team_cache[away_code] = get_or_create_team(session, away_code, away_name_raw)

            home_team = team_cache[home_code]
            away_team = team_cache[away_code]
            
            # Create Game
            game_date = row['Date']
            
            # Check for existing game (simple check by date + teams)
            existing_game = session.scalar(
                select(Game).where(
                    Game.scheduled_datetime == game_date,
                    Game.home_team_id == home_team.team_id,
                    Game.away_team_id == away_team.team_id
                )
            )
            
            if existing_game:
                game = existing_game
            else:
                game = Game(
                    league=LeagueType.NBL,
                    season=row['Season'],
                    scheduled_datetime=game_date,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=int(row['Home Score']) if pd.notna(row['Home Score']) else None,
                    away_score=int(row['Away Score']) if pd.notna(row['Away Score']) else None,
                    status=GameStatus.COMPLETED if pd.notna(row['Home Score']) else GameStatus.SCHEDULED,
                    venue_name=home_team.venue_name, # Default to home venue
                    notes=row.get('Notes') if pd.notna(row.get('Notes')) else None
                )
                session.add(game)
                session.flush() # Get ID
                games_inserted += 1

            # Create OddsHistory entries
            # 1. Home/Away Closing Odds
            if pd.notna(row.get('Home Odds Close')):
                odds_home = OddsHistory(
                    game=game,
                    sportsbook="Consensus",
                    bet_type=BetType.MONEYLINE,
                    selection=home_team.team_name,
                    decimal_odds=row['Home Odds Close'],
                    is_closing_line=True
                )
                session.add(odds_home)
                odds_inserted += 1

            if pd.notna(row.get('Away Odds Close')):
                odds_away = OddsHistory(
                    game=game,
                    sportsbook="Consensus",
                    bet_type=BetType.MONEYLINE,
                    selection=away_team.team_name,
                    decimal_odds=row['Away Odds Close'],
                    is_closing_line=True
                )
                session.add(odds_away)
                odds_inserted += 1
            
            if games_inserted % 100 == 0:
                session.commit()
                print(f"Processed {games_inserted} games...")

        session.commit()
        
    except Exception as e:
        session.rollback()
        print(f"Error inserting data: {e}")
        raise
    finally:
        session.close()
    
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
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        games = session.scalars(
            select(Game)
            .order_by(Game.scheduled_datetime.desc())
            .limit(5)
        ).all()

        for g in games:
            print(f"  {g.scheduled_datetime.date()}: {g.home_team.team_code} vs {g.away_team.team_code} "
                  f"({g.home_score}-{g.away_score})")
    finally:
        session.close()


if __name__ == "__main__":
    main()

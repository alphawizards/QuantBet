"""
NBL Data Integration Module.

Combines scraped nblR data with xlsx historical odds data
to create a unified dataset for the betting system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from fuzzywuzzy import fuzz

# Configure logging
logger = logging.getLogger(__name__)


# Team name mapping: nblR names -> standardized codes
TEAM_NAME_MAP = {
    # Current NBL Teams
    "Melbourne United": "MEL",
    "Sydney Kings": "SYD",
    "Perth Wildcats": "PER",
    "Brisbane Bullets": "BRI",
    "Adelaide 36ers": "ADL",
    "New Zealand Breakers": "NZB",
    "Illawarra Hawks": "ILL",
    "Cairns Taipans": "CAI",
    "Tasmania JackJumpers": "TAS",
    "South East Melbourne Phoenix": "SEM",
    
    # Historical team names
    "Melbourne Tigers": "MEL",
    "Melbourne": "MEL",
    "Sydney": "SYD",
    "Perth": "PER",
    "Brisbane": "BRI",
    "Adelaide": "ADL",
    "New Zealand": "NZB",
    "Illawarra": "ILL",
    "Cairns": "CAI",
    "Tasmania": "TAS",
    
    # Nicknames
    "United": "MEL",
    "Kings": "SYD",
    "Wildcats": "PER",
    "Bullets": "BRI",
    "36ers": "ADL",
    "Breakers": "NZB",
    "Hawks": "ILL",
    "Taipans": "CAI",
    "JackJumpers": "TAS",
    "Phoenix": "SEM",
}


class NBLDataIntegrator:
    """
    Integrates scraped NBL data with xlsx historical data.
    
    The integration strategy:
        - nblR provides match results, team stats, and player stats
        - xlsx provides betting odds data
        - Merge by (date, home_team, away_team)
        - xlsx odds enrich nblR match data
    
    Attributes:
        fuzzy_threshold: Minimum score for fuzzy team name matching (0-100)
    """
    
    def __init__(self, fuzzy_threshold: int = 80):
        """
        Initialize the integrator.
        
        Args:
            fuzzy_threshold: Minimum fuzzy match score for team names
        """
        self.fuzzy_threshold = fuzzy_threshold
    
    def normalize_team_name(self, name: str) -> str:
        """
        Normalize a team name to a standard team code.
        
        Args:
            name: Raw team name from data source
            
        Returns:
            Standardized team code (e.g., 'MEL', 'SYD')
        """
        if pd.isna(name):
            return "UNKNOWN"
        
        name = str(name).strip()
        
        # Direct lookup first
        if name in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[name]
        
        # Try case-insensitive lookup
        name_lower = name.lower()
        for key, code in TEAM_NAME_MAP.items():
            if key.lower() == name_lower:
                return code
        
        # Fuzzy match as fallback
        best_score = 0
        best_code = "UNKNOWN"
        
        for key, code in TEAM_NAME_MAP.items():
            score = fuzz.ratio(name_lower, key.lower())
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_code = code
        
        if best_code == "UNKNOWN":
            logger.warning(f"Could not match team name: {name}")
        
        return best_code
    
    def prepare_scraped_data(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare scraped match results for merging.
        
        Args:
            results_df: Match results from NBLDataScraper.get_match_results()
            
        Returns:
            DataFrame with standardized columns for merging
        """
        df = results_df.copy()
        
        # Parse match time
        if 'match_time' in df.columns:
            df['game_date'] = pd.to_datetime(df['match_time']).dt.date
        elif 'match_time_utc' in df.columns:
            df['game_date'] = pd.to_datetime(df['match_time_utc']).dt.date
        
        # Normalize team names
        df['home_team'] = df['home_team_name'].apply(self.normalize_team_name)
        df['away_team'] = df['away_team_name'].apply(self.normalize_team_name)
        
        # Parse scores
        df['home_score'] = pd.to_numeric(df['home_score_string'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score_string'], errors='coerce')
        
        # Select and rename columns
        columns = {
            'match_id': 'external_game_id',
            'game_date': 'game_date',
            'season': 'season',
            'home_team': 'home_team',
            'away_team': 'away_team',
            'home_score': 'home_score',
            'away_score': 'away_score',
            'venue_name': 'venue_name',
            'attendance': 'attendance',
            'round_number': 'round_number',
            'match_type': 'match_type',
            'extra_periods_used': 'overtime_periods',
        }
        
        result = pd.DataFrame()
        for src, dst in columns.items():
            if src in df.columns:
                result[dst] = df[src]
        
        # Add metadata
        result['data_source'] = 'nblr'
        
        return result
    
    def prepare_xlsx_data(
        self,
        xlsx_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare xlsx odds data for merging.
        
        Args:
            xlsx_df: Data from load_nbl_data() in scripts/load_data.py
            
        Returns:
            DataFrame with standardized columns for merging
        """
        df = xlsx_df.copy()
        
        # Normalize date
        df['game_date'] = pd.to_datetime(df['Date']).dt.date
        
        # Normalize team names
        df['home_team'] = df['Home Team'].apply(self.normalize_team_name)
        df['away_team'] = df['Away Team'].apply(self.normalize_team_name)
        
        # Select odds columns
        odds_columns = {
            'game_date': 'game_date',
            'home_team': 'home_team',
            'away_team': 'away_team',
            'Home Score': 'home_score',
            'Away Score': 'away_score',
            'Home Odds': 'home_odds',
            'Away Odds': 'away_odds',
            'Home Odds Open': 'home_odds_open',
            'Home Odds Close': 'home_odds_close',
            'Away Odds Open': 'away_odds_open',
            'Away Odds Close': 'away_odds_close',
            'Home Line Open': 'home_line_open',
            'Home Line Close': 'home_line_close',
            'Total Score Open': 'total_open',
            'Total Score Close': 'total_close',
            'Bookmakers Surveyed': 'bookmakers_surveyed',
        }
        
        result = pd.DataFrame()
        for src, dst in odds_columns.items():
            if src in df.columns:
                result[dst] = df[src]
        
        result['data_source'] = 'xlsx'
        
        return result
    
    def merge_data_sources(
        self,
        scraped_df: pd.DataFrame,
        xlsx_df: pd.DataFrame,
        prefer_source: str = "nblr"
    ) -> pd.DataFrame:
        """
        Merge scraped and xlsx data sources.
        
        Strategy:
            1. nblR is primary source for match results (more complete)
            2. xlsx enriches with odds data (unique to xlsx)
            3. Merge on (game_date, home_team, away_team)
            4. For conflicts, prefer specified source
        
        Args:
            scraped_df: Prepared scraped data
            xlsx_df: Prepared xlsx data
            prefer_source: Which source to prefer for conflicts ('nblr' or 'xlsx')
            
        Returns:
            Merged DataFrame with all available data
        """
        # Prepare merge keys
        scraped = self.prepare_scraped_data(scraped_df)
        xlsx = self.prepare_xlsx_data(xlsx_df)
        
        merge_keys = ['game_date', 'home_team', 'away_team']
        
        # Identify common and unique columns
        scraped_only = [c for c in scraped.columns 
                       if c not in xlsx.columns and c not in merge_keys]
        xlsx_only = [c for c in xlsx.columns 
                    if c not in scraped.columns and c not in merge_keys]
        common = [c for c in scraped.columns 
                 if c in xlsx.columns and c not in merge_keys]
        
        logger.info(f"Scraped-only columns: {scraped_only}")
        logger.info(f"XLSX-only columns: {xlsx_only}")
        logger.info(f"Common columns: {common}")
        
        # Perform outer merge
        merged = pd.merge(
            scraped,
            xlsx[merge_keys + xlsx_only],  # Only unique xlsx columns
            on=merge_keys,
            how='outer',
            suffixes=('_nblr', '_xlsx')
        )
        
        # For common columns with conflicts, use preferred source
        for col in common:
            if f'{col}_nblr' in merged.columns:
                if prefer_source == 'nblr':
                    merged[col] = merged[f'{col}_nblr'].fillna(merged.get(f'{col}_xlsx'))
                else:
                    merged[col] = merged[f'{col}_xlsx'].fillna(merged.get(f'{col}_nblr'))
                
                merged.drop([f'{col}_nblr', f'{col}_xlsx'], axis=1, errors='ignore', inplace=True)
        
        # Sort by date
        merged.sort_values('game_date', ascending=False, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        logger.info(f"Merged dataset: {len(merged)} total games")
        logger.info(f"  - From nblR only: {(merged['data_source'] == 'nblr').sum()}")
        logger.info(f"  - From xlsx only: {(merged['data_source'] == 'xlsx').sum()}")
        
        return merged
    
    def get_merge_stats(
        self,
        scraped_df: pd.DataFrame,
        xlsx_df: pd.DataFrame
    ) -> dict:
        """
        Get statistics about the merge without actually merging.
        
        Args:
            scraped_df: Raw scraped data
            xlsx_df: Raw xlsx data
            
        Returns:
            Dictionary with merge statistics
        """
        scraped = self.prepare_scraped_data(scraped_df)
        xlsx = self.prepare_xlsx_data(xlsx_df)
        
        merge_keys = ['game_date', 'home_team', 'away_team']
        
        # Find overlapping games
        scraped_keys = set(scraped[merge_keys].apply(tuple, axis=1))
        xlsx_keys = set(xlsx[merge_keys].apply(tuple, axis=1))
        
        overlap = scraped_keys & xlsx_keys
        scraped_only = scraped_keys - xlsx_keys
        xlsx_only = xlsx_keys - scraped_keys
        
        return {
            'total_scraped': len(scraped_keys),
            'total_xlsx': len(xlsx_keys),
            'overlap': len(overlap),
            'scraped_only': len(scraped_only),
            'xlsx_only': len(xlsx_only),
            'total_merged': len(scraped_keys | xlsx_keys),
            'overlap_percentage': len(overlap) / max(len(xlsx_keys), 1) * 100,
        }

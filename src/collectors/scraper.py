"""
NBL Data Scraper Module.

Downloads basketball data from the nblR GitHub releases (RDS files)
and converts them to pandas DataFrames for analysis.

Data Sources:
    - Match Results (1979+): Wide or long format game results
    - Team Box Scores (2015-2016+): Team-level statistics
    - Player Box Scores (2015-2016+): Individual player statistics
    - Play-by-Play (2015-2016+): Event-level game data
    - Shots (2015-2016+): Shot location and outcome data

Example:
    >>> scraper = NBLDataScraper()
    >>> results = scraper.get_match_results(format="wide")
    >>> print(f"Downloaded {len(results)} matches")
"""

import hashlib
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import pyreadr
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)


class NBLDataScraper:
    """
    Scraper for NBL data from nblR GitHub releases.
    
    The nblR project (https://github.com/JaseZiv/nblR) maintains
    historical NBL data in RDS format, updated regularly.
    
    This scraper downloads the RDS files and converts them to
    pandas DataFrames for use in the QuantBet prediction system.
    
    Attributes:
        cache_dir: Directory for caching downloaded files
        use_cache: Whether to use local cache
        cache_ttl_hours: Hours before cache expires (default: 24)
    
    Example:
        >>> scraper = NBLDataScraper(cache_dir="data/cache")
        >>> results = scraper.get_match_results()
        >>> print(results.head())
    """
    
    # Base URL for nblR data releases
    BASE_URL = "https://github.com/JaseZiv/nblr_data/releases/download"
    
    # Endpoints for different data types
    ENDPOINTS = {
        "results_wide": f"{BASE_URL}/match_results/results_wide.rds",
        "results_long": f"{BASE_URL}/match_results/results_long.rds",
        "box_team": f"{BASE_URL}/box_team/box_team.rds",
        "box_player": f"{BASE_URL}/box_player/box_player.rds",
        "pbp": f"{BASE_URL}/pbp/pbp.rds",
        "shots": f"{BASE_URL}/shots/shots.rds",
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize the NBL data scraper.
        
        Args:
            cache_dir: Directory for caching downloaded files.
                      If None, uses 'data/cache' in the project root.
            use_cache: Whether to use local file caching.
            cache_ttl_hours: Hours before cached files are considered stale.
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Correct path with new structure: src/collectors -> src -> .. -> data
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        
        self.use_cache = use_cache
        self.cache_ttl_hours = cache_ttl_hours
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, endpoint: str) -> Path:
        """Get the cache file path for an endpoint."""
        url_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
        return self.cache_dir / f"{endpoint}_{url_hash}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not stale."""
        if not cache_path.exists():
            return False
        
        # Check file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        
        return age_hours < self.cache_ttl_hours
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _download_rds(self, url: str) -> pd.DataFrame:
        """
        Download RDS file and convert to DataFrame.
        
        Uses tenacity retry logic for transient network errors.
        
        Args:
            url: URL to download RDS file from
            
        Returns:
            DataFrame parsed from RDS file
            
        Raises:
            requests.HTTPError: If download fails after retries
            ValueError: If RDS parsing fails
        """
        logger.info(f"Downloading: {url}")
        
        # Download to temporary file
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
        
        try:
            # Read RDS file using pyreadr
            result = pyreadr.read_r(tmp_path)
            
            # RDS files contain a single object, get the first (and only) value
            if result is None or len(result) == 0:
                raise ValueError(f"Empty RDS file from {url}")
            
            df = list(result.values())[0]
            logger.info(f"Downloaded {len(df)} rows from {url}")
            
            return df
            
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)
    
    def _fetch_data(self, endpoint_key: str) -> pd.DataFrame:
        """
        Fetch data for an endpoint, using cache if available.
        
        Args:
            endpoint_key: Key from ENDPOINTS dictionary
            
        Returns:
            DataFrame with the requested data
        """
        if endpoint_key not in self.ENDPOINTS:
            raise ValueError(f"Unknown endpoint: {endpoint_key}")
        
        url = self.ENDPOINTS[endpoint_key]
        cache_path = self._get_cache_path(endpoint_key)
        
        # Check cache first
        if self.use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Loading from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        
        # Download fresh data
        df = self._download_rds(url)
        
        # Save to cache
        if self.use_cache:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached to: {cache_path}")
        
        return df
    
    def get_match_results(
        self,
        format: Literal["wide", "long"] = "wide"
    ) -> pd.DataFrame:
        """
        Get NBL match results since 1979.
        
        Args:
            format: 
                - "wide": One row per match (home/away columns)
                - "long": Two rows per match (one per team)
        
        Returns:
            DataFrame with columns including:
                - match_id, season, venue_name
                - home_team_name, home_score_string
                - away_team_name, away_score_string
                - match_time, attendance
        
        Example:
            >>> results = scraper.get_match_results(format="wide")
            >>> print(results[['season', 'home_team_name', 'away_team_name', 'home_score_string']])
        """
        endpoint = f"results_{format}"
        return self._fetch_data(endpoint)
    
    def get_team_box_scores(self) -> pd.DataFrame:
        """
        Get team-level box score statistics (2015-2016+).
        
        Returns:
            DataFrame with 67 columns including:
                - match_id, season, home_away, name, score
                - field_goals_made/attempted, three_pointers_made/attempted
                - rebounds_total, assists, turnovers, steals, blocks
                - points_from_turnovers, points_second_chance, bench_points
        
        Example:
            >>> team_box = scraper.get_team_box_scores()
            >>> mel = team_box[team_box['code'] == 'MEL']
            >>> print(mel[['season', 'score', 'field_goals_percentage']])
        """
        return self._fetch_data("box_team")
    
    def get_player_box_scores(self) -> pd.DataFrame:
        """
        Get player-level box score statistics (2015-2016+).
        
        Returns:
            DataFrame with 47 columns including:
                - match_id, season, team_name, first_name, family_name
                - starter, minutes, points
                - field_goals_made/attempted, three_pointers_made/attempted
                - rebounds_total, assists, turnovers, steals, blocks
        
        Example:
            >>> player_box = scraper.get_player_box_scores()
            >>> print(player_box.groupby('family_name')['points'].mean().sort_values(ascending=False))
        """
        return self._fetch_data("box_player")
    
    def get_play_by_play(self) -> pd.DataFrame:
        """
        Get play-by-play event data (2015-2016+).
        
        Returns:
            DataFrame with 25 columns including:
                - match_id, season, team_name, period, gt (game time)
                - action_type, sub_type, success, scoring
                - first_name, family_name
        
        Note:
            This is a large dataset (~600k+ rows). Use with caution.
        
        Example:
            >>> pbp = scraper.get_play_by_play()
            >>> scoring_plays = pbp[pbp['scoring'] == 1]
        """
        return self._fetch_data("pbp")
    
    def get_shots(self) -> pd.DataFrame:
        """
        Get shot location and outcome data (2015-2016+).
        
        Returns:
            DataFrame with 20 columns including:
                - match_id, season, team_name, home_away
                - x, y (court coordinates), r (result: 0=miss, 1=make)
                - action_type (2pt/3pt), sub_type (jumpshot, layup, etc.)
                - first_name, family_name
        
        Example:
            >>> shots = scraper.get_shots()
            >>> three_pt = shots[shots['action_type'] == '3pt']
            >>> print(f"3PT%: {three_pt['r'].mean():.1%}")
        """
        return self._fetch_data("shots")
    
    def get_all_data(self) -> dict:
        """
        Download all available data sources.
        
        Returns:
            Dictionary with keys:
                - results: Match results (wide format)
                - team_box: Team box scores
                - player_box: Player box scores
                - pbp: Play-by-play data
                - shots: Shot data
        
        Warning:
            This downloads a significant amount of data.
            Consider using individual methods if you don't need everything.
        """
        return {
            "results": self.get_match_results(format="wide"),
            "team_box": self.get_team_box_scores(),
            "player_box": self.get_player_box_scores(),
            "pbp": self.get_play_by_play(),
            "shots": self.get_shots(),
        }
    
    def clear_cache(self) -> int:
        """
        Clear all cached files.
        
        Returns:
            Number of files deleted
        """
        deleted = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
            deleted += 1
        logger.info(f"Cleared {deleted} cached files")
        return deleted
    
    # =========================================================================
    # Four Factors Data Extraction Methods
    # =========================================================================
    
    def get_four_factors_data(
        self,
        season: Optional[str] = None,
        min_season: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get prepared Four Factors dataset with team and opponent stats merged.
        
        Returns game-level data with all stats needed to compute Dean Oliver's
        Four Factors for both home and away teams.
        
        Args:
            season: Filter to specific season (e.g., "2023-2024")
            min_season: Only include seasons >= this (e.g., "2020-2021")
        
        Returns:
            DataFrame with columns:
                - match_id, season, game_date
                - home_team, away_team
                - home_score, away_score
                - home_fgm, home_fga, home_fg3m, home_fta, home_ftm
                - home_turnovers, home_orb, home_drb
                - away_fgm, away_fga, away_fg3m, away_fta, away_ftm
                - away_turnovers, away_orb, away_drb
                - home_possessions, away_possessions, pace
        
        Example:
            >>> ff_data = scraper.get_four_factors_data(min_season="2020-2021")
            >>> print(ff_data[['home_team', 'away_team', 'pace']])
        """
        # Get team box scores
        team_box = self.get_team_box_scores()
        
        # Get match results for dates
        results = self.get_match_results(format="wide")
        
        # Ensure we have required columns
        required_cols = [
            'match_id', 'season', 'home_away', 'name', 'code', 'score',
            'field_goals_made', 'field_goals_attempted',
            'three_pointers_made', 'free_throws_made', 'free_throws_attempted',
            'turnovers', 'rebounds_offensive', 'rebounds_defensive'
        ]
        
        missing = set(required_cols) - set(team_box.columns)
        if missing:
            logger.warning(f"Missing columns in team box scores: {missing}")
            # Try alternate column names
            col_remap = {
                'offensive_rebounds': 'rebounds_offensive',
                'defensive_rebounds': 'rebounds_defensive',
            }
            for old, new in col_remap.items():
                if old in team_box.columns and new not in team_box.columns:
                    team_box[new] = team_box[old]
        
        # Split into home and away teams
        home_box = team_box[team_box['home_away'] == 'home'].copy()
        away_box = team_box[team_box['home_away'] == 'away'].copy()
        
        # Rename columns for home/away perspective
        home_cols = {
            'name': 'home_team_name',
            'code': 'home_team',
            'score': 'home_score',
            'field_goals_made': 'home_fgm',
            'field_goals_attempted': 'home_fga',
            'three_pointers_made': 'home_fg3m',
            'free_throws_made': 'home_ftm',
            'free_throws_attempted': 'home_fta',
            'turnovers': 'home_turnovers',
            'rebounds_offensive': 'home_orb',
            'rebounds_defensive': 'home_drb',
        }
        
        away_cols = {
            'name': 'away_team_name',
            'code': 'away_team',
            'score': 'away_score',
            'field_goals_made': 'away_fgm',
            'field_goals_attempted': 'away_fga',
            'three_pointers_made': 'away_fg3m',
            'free_throws_made': 'away_ftm',
            'free_throws_attempted': 'away_fta',
            'turnovers': 'away_turnovers',
            'rebounds_offensive': 'away_orb',
            'rebounds_defensive': 'away_drb',
        }
        
        home_box = home_box.rename(columns=home_cols)
        away_box = away_box.rename(columns=away_cols)
        
        # Merge home and away on match_id
        home_keep = ['match_id', 'season'] + list(home_cols.values())
        away_keep = ['match_id'] + list(away_cols.values())
        
        home_keep = [c for c in home_keep if c in home_box.columns]
        away_keep = [c for c in away_keep if c in away_box.columns]
        
        merged = pd.merge(
            home_box[home_keep],
            away_box[away_keep],
            on='match_id',
            how='inner'
        )
        
        # Add game date from results
        if 'match_time' in results.columns:
            date_df = results[['match_id', 'match_time']].copy()
            date_df['game_date'] = pd.to_datetime(date_df['match_time']).dt.date
            merged = pd.merge(merged, date_df[['match_id', 'game_date']], 
                              on='match_id', how='left')
        
        # Calculate possessions and pace
        merged = self._add_possessions_and_pace(merged)
        
        # Filter by season
        if season:
            merged = merged[merged['season'] == season]
        if min_season:
            merged = merged[merged['season'] >= min_season]
        
        # Sort by date
        if 'game_date' in merged.columns:
            merged = merged.sort_values('game_date').reset_index(drop=True)
        
        logger.info(f"Four Factors data: {len(merged)} games")
        return merged
    
    def _add_possessions_and_pace(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add estimated possessions and pace to DataFrame.
        
        Uses the standard possession formula:
            Poss = FGA - OREB + TO + 0.44 * FTA
        
        Pace is normalized to NBA 48-minute standard from FIBA 40-minute games:
            Pace = (48/40) * (home_poss + away_poss) / 2
        """
        df = df.copy()
        
        # Home possessions
        if all(c in df.columns for c in ['home_fga', 'home_orb', 'home_turnovers', 'home_fta']):
            df['home_possessions'] = (
                df['home_fga'] 
                - df['home_orb'] 
                + df['home_turnovers'] 
                + 0.44 * df['home_fta']
            )
        
        # Away possessions
        if all(c in df.columns for c in ['away_fga', 'away_orb', 'away_turnovers', 'away_fta']):
            df['away_possessions'] = (
                df['away_fga'] 
                - df['away_orb'] 
                + df['away_turnovers'] 
                + 0.44 * df['away_fta']
            )
        
        # Game pace (normalized to 48-min NBA standard)
        if 'home_possessions' in df.columns and 'away_possessions' in df.columns:
            # Average of both teams' possessions, scaled from 40 to 48 minutes
            df['pace'] = (48 / 40) * (df['home_possessions'] + df['away_possessions']) / 2
        
        return df
    
    def normalize_to_per_100(
        self,
        df: pd.DataFrame,
        stat_columns: list,
        poss_column: str = 'home_possessions'
    ) -> pd.DataFrame:
        """
        Normalize counting stats to per-100-possessions rate.
        
        This accounts for NBL's higher pace compared to other leagues
        and allows fair comparison across games/seasons.
        
        Args:
            df: DataFrame with counting stats
            stat_columns: List of column names to normalize
            poss_column: Column containing possession count
            
        Returns:
            DataFrame with new columns named {original}_per100
            
        Example:
            >>> normalized = scraper.normalize_to_per_100(
            ...     df, 
            ...     ['home_score', 'home_turnovers'],
            ...     'home_possessions'
            ... )
        """
        df = df.copy()
        
        if poss_column not in df.columns:
            logger.warning(f"Possession column '{poss_column}' not found")
            return df
        
        for col in stat_columns:
            if col in df.columns:
                # Per 100 = (stat / possessions) * 100
                df[f'{col}_per100'] = (df[col] / df[poss_column]) * 100
            else:
                logger.warning(f"Stat column '{col}' not found")
        
        return df

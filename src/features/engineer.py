"""
NBL/WNBL Feature Engineering Module.

This module implements the NBLFeatureEngineer class which generates
domain-specific features for Australian basketball betting models.

Features generated:
    1. Rolling Efficiency (Last 5 games Offensive/Defensive Rating)
    2. Travel Fatigue Score (Distance-based with Perth-NZ weighting)
    3. Import Availability (Boolean for key import players)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ============================================================================
# Constants & Configuration
# ============================================================================

# NBL/WNBL team locations (latitude, longitude)
# Used for travel distance calculations
TEAM_LOCATIONS: Dict[str, Tuple[float, float]] = {
    # NBL Teams
    "MEL": (-37.8136, 144.9631),   # Melbourne United
    "SYD": (-33.8688, 151.2093),   # Sydney Kings
    "PER": (-31.9505, 115.8605),   # Perth Wildcats
    "BRI": (-27.4698, 153.0251),   # Brisbane Bullets
    "ADL": (-34.9285, 138.6007),   # Adelaide 36ers
    "NZB": (-36.8485, 174.7633),   # New Zealand Breakers (Auckland)
    "ILL": (-34.4278, 150.8931),   # Illawarra Hawks
    "CAI": (-16.9186, 145.7781),   # Cairns Taipans
    "TAS": (-42.8821, 147.3272),   # Tasmania JackJumpers
    "SEM": (-37.8136, 144.9631),   # South East Melbourne Phoenix
    
    # WNBL Teams (some overlap with NBL)
    "CAN": (-35.2809, 149.1300),   # Canberra Capitals
    "TOW": (-19.2590, 146.8169),   # Townsville Fire
    "BEN": (-36.7570, 144.2794),   # Bendigo Spirit
    "PER_W": (-31.9505, 115.8605), # Perth Lynx
    "SYD_W": (-33.8688, 151.2093), # Sydney Flames
    "MEL_W": (-37.8136, 144.9631), # Melbourne Boomers
    "ADL_W": (-34.9285, 138.6007), # Adelaide Lightning
}

# Travel fatigue multipliers
class TravelMultiplier(Enum):
    """Multipliers for travel fatigue based on route type."""
    INTRA_CITY = 1.0      # Same city (no fatigue)
    INTERSTATE = 1.0      # Standard interstate
    PERTH_EAST = 1.5      # Perth to East Coast (long haul)
    PERTH_NZ = 2.0        # Perth to New Zealand (cross-Tasman + time zones)
    NZ_INTERSTATE = 1.3   # NZ to any Australian city


@dataclass
class TeamEfficiency:
    """Container for team offensive/defensive efficiency metrics."""
    offensive_rating: float  # Points per 100 possessions
    defensive_rating: float  # Opponent points per 100 possessions
    net_rating: float       # Offensive - Defensive
    pace: float             # Possessions per game
    
    @property
    def efficiency_differential(self) -> float:
        """Alias for net rating."""
        return self.net_rating


@dataclass
class TravelFatigueResult:
    """Container for travel fatigue calculation results."""
    score: int              # 0-5 fatigue score
    distance_km: float      # Total distance traveled
    route_type: str         # Description of route
    back_to_back: bool      # If game is on consecutive days
    days_rest: int          # Days since last game


# ============================================================================
# NBL Feature Engineer Class
# ============================================================================

class NBLFeatureEngineer:
    """
    Feature engineering for NBL/WNBL game predictions.
    
    Generates three categories of features:
        1. Rolling Efficiency: Last N games Offensive/Defensive Rating
        2. Travel Fatigue Score: Distance-based with route multipliers
        3. Import Availability: Boolean flags for key import players
    
    Mathematical foundations:
    
    **Offensive Rating (ORtg):**
    
    .. math::
        ORtg = 100 \\times \\frac{Points}{Possessions}
    
    **Defensive Rating (DRtg):**
    
    .. math::
        DRtg = 100 \\times \\frac{Opponent\\ Points}{Possessions}
    
    **Possessions Estimate:**
    
    .. math::
        Poss \\approx FGA - OREB + TO + 0.44 \\times FTA
    
    Example:
        >>> engineer = NBLFeatureEngineer()
        >>> features = engineer.generate_game_features(
        ...     home_team="MEL",
        ...     away_team="PER",
        ...     game_date=datetime(2024, 1, 15),
        ...     historical_data=games_df
        ... )
    
    Attributes:
        rolling_window: Number of games for rolling calculations (default: 5)
        team_locations: Dictionary mapping team codes to (lat, lon) tuples
    """
    
    def __init__(
        self,
        rolling_window: int = 5,
        team_locations: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize the feature engineer.
        
        Args:
            rolling_window: Number of games for rolling efficiency calculations.
                            Default is 5 (roughly 2 weeks of NBL games).
            team_locations: Custom team location coordinates. If None, uses
                           the default NBL/WNBL team locations.
        """
        self.rolling_window = rolling_window
        self.team_locations = team_locations or TEAM_LOCATIONS
    
    # ========================================================================
    # Rolling Efficiency Features
    # ========================================================================
    
    def calculate_possessions(
        self,
        fga: int,
        oreb: int,
        turnovers: int,
        fta: int
    ) -> float:
        """
        Estimate possessions using the standard formula.
        
        The formula used is:
        
        .. math::
            Possessions \\approx FGA - OREB + TO + 0.44 \\times FTA
        
        The 0.44 coefficient accounts for the fact that not all free throws
        end possessions (and-ones, technical fouls, etc.).
        
        Args:
            fga: Field goals attempted
            oreb: Offensive rebounds
            turnovers: Turnovers committed
            fta: Free throws attempted
        
        Returns:
            Estimated number of possessions
        """
        return fga - oreb + turnovers + (0.44 * fta)
    
    def calculate_team_ratings(
        self,
        team_stats: pd.DataFrame
    ) -> TeamEfficiency:
        """
        Calculate offensive and defensive efficiency ratings.
        
        Args:
            team_stats: DataFrame with columns:
                - points: Team points scored
                - opp_points: Opponent points scored
                - fga, oreb, turnovers, fta: For possession calculation
                - opp_fga, opp_oreb, opp_turnovers, opp_fta: Opponent stats
        
        Returns:
            TeamEfficiency object with calculated ratings
        
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = [
            'points', 'opp_points', 'fga', 'oreb', 'turnovers', 'fta',
            'opp_fga', 'opp_oreb', 'opp_turnovers', 'opp_fta'
        ]
        missing = set(required_cols) - set(team_stats.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Calculate possessions for both teams
        team_poss = self.calculate_possessions(
            team_stats['fga'].sum(),
            team_stats['oreb'].sum(),
            team_stats['turnovers'].sum(),
            team_stats['fta'].sum()
        )
        
        opp_poss = self.calculate_possessions(
            team_stats['opp_fga'].sum(),
            team_stats['opp_oreb'].sum(),
            team_stats['opp_turnovers'].sum(),
            team_stats['opp_fta'].sum()
        )
        
        # Average possessions (should be roughly equal)
        avg_poss = (team_poss + opp_poss) / 2
        num_games = len(team_stats)
        
        if avg_poss == 0:
            return TeamEfficiency(0.0, 0.0, 0.0, 0.0)
        
        # Calculate ratings per 100 possessions
        total_points = team_stats['points'].sum()
        total_opp_points = team_stats['opp_points'].sum()
        
        ortg = 100 * (total_points / avg_poss)
        drtg = 100 * (total_opp_points / avg_poss)
        
        return TeamEfficiency(
            offensive_rating=round(ortg, 1),
            defensive_rating=round(drtg, 1),
            net_rating=round(ortg - drtg, 1),
            pace=round(avg_poss / num_games, 1)
        )
    
    def get_rolling_efficiency(
        self,
        team_code: str,
        game_date: datetime,
        historical_data: pd.DataFrame,
        n_games: Optional[int] = None
    ) -> Optional[TeamEfficiency]:
        """
        Calculate rolling efficiency for the last N games before a given date.
        
        Args:
            team_code: Team identifier (e.g., 'MEL', 'PER')
            game_date: Date of the game to predict (features from before this)
            historical_data: DataFrame with game-level stats
            n_games: Number of games to include (default: self.rolling_window)
        
        Returns:
            TeamEfficiency object or None if insufficient data
        """
        n = n_games or self.rolling_window
        
        # Filter to team's games before the target date
        team_games = historical_data[
            ((historical_data['home_team'] == team_code) |
             (historical_data['away_team'] == team_code)) &
            (historical_data['game_date'] < game_date)
        ].sort_values('game_date', ascending=False).head(n)
        
        if len(team_games) < n:
            return None  # Insufficient data
        
        # Normalize perspective (team is always "us")
        normalized = self._normalize_team_perspective(team_games, team_code)
        
        return self.calculate_team_ratings(normalized)
    
    def _normalize_team_perspective(
        self,
        games: pd.DataFrame,
        team_code: str
    ) -> pd.DataFrame:
        """
        Normalize game data so team_code is always the 'home' team perspective.
        
        This allows consistent calculation of offensive/defensive stats
        regardless of whether the team was home or away in each game.
        """
        normalized_rows = []
        
        for _, game in games.iterrows():
            if game['home_team'] == team_code:
                # Team was home, use stats as-is
                normalized_rows.append({
                    'points': game['home_score'],
                    'opp_points': game['away_score'],
                    'fga': game.get('home_fga', 0),
                    'oreb': game.get('home_oreb', 0),
                    'turnovers': game.get('home_turnovers', 0),
                    'fta': game.get('home_fta', 0),
                    'opp_fga': game.get('away_fga', 0),
                    'opp_oreb': game.get('away_oreb', 0),
                    'opp_turnovers': game.get('away_turnovers', 0),
                    'opp_fta': game.get('away_fta', 0),
                })
            else:
                # Team was away, flip perspective
                normalized_rows.append({
                    'points': game['away_score'],
                    'opp_points': game['home_score'],
                    'fga': game.get('away_fga', 0),
                    'oreb': game.get('away_oreb', 0),
                    'turnovers': game.get('away_turnovers', 0),
                    'fta': game.get('away_fta', 0),
                    'opp_fga': game.get('home_fga', 0),
                    'opp_oreb': game.get('home_oreb', 0),
                    'opp_turnovers': game.get('home_turnovers', 0),
                    'opp_fta': game.get('home_fta', 0),
                })
        
        return pd.DataFrame(normalized_rows)
    
    # ========================================================================
    # Travel Fatigue Features
    # ========================================================================
    
    @staticmethod
    def haversine_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate the great-circle distance between two points on Earth.
        
        Uses the Haversine formula:
        
        .. math::
            a = \\sin^2(\\Delta\\phi/2) + \\cos(\\phi_1) \\cos(\\phi_2) \\sin^2(\\Delta\\lambda/2)
        
        .. math::
            c = 2 \\arctan2(\\sqrt{a}, \\sqrt{1-a})
        
        .. math::
            d = R \\times c
        
        Where R = 6371 km (Earth's radius).
        
        Args:
            lat1, lon1: Coordinates of first point (degrees)
            lat2, lon2: Coordinates of second point (degrees)
        
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def get_travel_distance(
        self,
        from_team: str,
        to_team: str
    ) -> float:
        """
        Calculate travel distance between two team cities.
        
        Args:
            from_team: Origin team code
            to_team: Destination team code
        
        Returns:
            Distance in kilometers
        
        Raises:
            ValueError: If team code not found in locations
        """
        if from_team not in self.team_locations:
            raise ValueError(f"Unknown team code: {from_team}")
        if to_team not in self.team_locations:
            raise ValueError(f"Unknown team code: {to_team}")
        
        lat1, lon1 = self.team_locations[from_team]
        lat2, lon2 = self.team_locations[to_team]
        
        return self.haversine_distance(lat1, lon1, lat2, lon2)
    
    def _get_route_multiplier(
        self,
        from_team: str,
        to_team: str,
        distance: float
    ) -> Tuple[float, str]:
        """
        Determine travel fatigue multiplier based on route characteristics.
        
        Perth and New Zealand routes get higher multipliers due to:
        - Long flight times
        - Time zone changes
        - Limited recovery time
        
        Returns:
            Tuple of (multiplier, route_description)
        """
        # Check for Perth routes
        perth_teams = {'PER', 'PER_W'}
        nz_teams = {'NZB'}
        
        from_perth = from_team in perth_teams
        to_perth = to_team in perth_teams
        from_nz = from_team in nz_teams
        to_nz = to_team in nz_teams
        
        # Perth to/from NZ (worst case)
        if (from_perth and to_nz) or (from_nz and to_perth):
            return TravelMultiplier.PERTH_NZ.value, "Perth-NZ Cross-Tasman"
        
        # Perth to/from East Coast
        if (from_perth or to_perth) and distance > 2000:
            return TravelMultiplier.PERTH_EAST.value, "Perth Long Haul"
        
        # NZ to/from Australia
        if from_nz or to_nz:
            return TravelMultiplier.NZ_INTERSTATE.value, "Trans-Tasman"
        
        # Standard interstate
        if distance < 100:
            return TravelMultiplier.INTRA_CITY.value, "Local"
        
        return TravelMultiplier.INTERSTATE.value, "Interstate"
    
    def calculate_travel_fatigue(
        self,
        team_code: str,
        game_date: datetime,
        schedule: pd.DataFrame
    ) -> TravelFatigueResult:
        """
        Calculate travel fatigue score for a team before a game.
        
        The score is 0-5 based on:
        - Distance traveled from last game venue
        - Route type (Perth/NZ legs weighted higher)
        - Days of rest
        - Back-to-back game status
        
        Scoring:
            - Base score from distance (0-3)
            - Route multiplier applied
            - +1 if back-to-back
            - -1 per extra day of rest (minimum 0)
        
        Args:
            team_code: Team identifier
            game_date: Date of upcoming game
            schedule: DataFrame with 'team', 'game_date', 'venue_team' columns
        
        Returns:
            TravelFatigueResult with score and metadata
        """
        # Find team's previous game
        team_games = schedule[
            ((schedule['home_team'] == team_code) |
             (schedule['away_team'] == team_code)) &
            (schedule['game_date'] < game_date)
        ].sort_values('game_date', ascending=False)
        
        if team_games.empty:
            # Season opener or first game we have data for
            return TravelFatigueResult(
                score=0,
                distance_km=0.0,
                route_type="First Game",
                back_to_back=False,
                days_rest=7  # Assume well-rested
            )
        
        last_game = team_games.iloc[0]
        
        # Determine where last game was played
        if last_game['home_team'] == team_code:
            last_venue = last_game['home_team']  # Home game
        else:
            last_venue = last_game['away_team']  # Away game location
        
        # Current game venue
        current_game = schedule[
            schedule['game_date'] == game_date
        ].iloc[0]
        
        if current_game['home_team'] == team_code:
            current_venue = team_code  # Playing at home
        else:
            current_venue = current_game['home_team']  # Playing away
        
        # Calculate distance
        distance = self.get_travel_distance(last_venue, current_venue)
        
        # Get route multiplier
        multiplier, route_type = self._get_route_multiplier(
            last_venue, current_venue, distance
        )
        
        # Calculate days rest
        days_rest = (game_date.date() - last_game['game_date'].date()).days
        back_to_back = days_rest <= 1
        
        # Calculate base score from distance (0-3)
        if distance < 200:
            base_score = 0
        elif distance < 1000:
            base_score = 1
        elif distance < 2500:
            base_score = 2
        else:
            base_score = 3
        
        # Apply multiplier and adjustments
        raw_score = base_score * multiplier
        
        if back_to_back:
            raw_score += 1
        else:
            # Reduce fatigue for extra rest (cap at -2)
            rest_reduction = min(2, max(0, days_rest - 2))
            raw_score -= rest_reduction
        
        # Bound to 0-5
        final_score = int(min(5, max(0, round(raw_score))))
        
        return TravelFatigueResult(
            score=final_score,
            distance_km=round(distance, 1),
            route_type=route_type,
            back_to_back=back_to_back,
            days_rest=days_rest
        )
    
    # ========================================================================
    # Import Availability Features
    # ========================================================================
    
    def check_import_availability(
        self,
        team_code: str,
        game_date: datetime,
        player_availability: pd.DataFrame
    ) -> Dict[str, bool]:
        """
        Check availability of import players for a team.
        
        NBL teams typically have 2-3 import players who have outsized
        impact on team performance. Their availability is a key feature.
        
        Args:
            team_code: Team identifier
            game_date: Date of the game
            player_availability: DataFrame with columns:
                - player_id: Unique player identifier
                - team: Team code
                - is_import: Boolean flag
                - available_from: Date player is available from
                - unavailable_until: Date player is unavailable until (if any)
        
        Returns:
            Dictionary with:
                - all_imports_available: Boolean
                - import_count_available: Number of available imports
                - import_count_total: Total imports on roster
                - missing_imports: List of unavailable import player IDs
        """
        # Filter to team's import players
        team_imports = player_availability[
            (player_availability['team'] == team_code) &
            (player_availability['is_import'] == True)
        ]
        
        if team_imports.empty:
            return {
                'all_imports_available': True,
                'import_count_available': 0,
                'import_count_total': 0,
                'missing_imports': []
            }
        
        available = []
        missing = []
        
        for _, player in team_imports.iterrows():
            is_available = True
            
            # Check if player has availability window
            if pd.notna(player.get('available_from')):
                if game_date.date() < player['available_from'].date():
                    is_available = False
            
            if pd.notna(player.get('unavailable_until')):
                if game_date.date() <= player['unavailable_until'].date():
                    is_available = False
            
            if is_available:
                available.append(player['player_id'])
            else:
                missing.append(player['player_id'])
        
        return {
            'all_imports_available': len(missing) == 0,
            'import_count_available': len(available),
            'import_count_total': len(team_imports),
            'missing_imports': missing
        }
    
    # ========================================================================
    # Combined Feature Generator
    # ========================================================================
    
    def generate_game_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        historical_data: pd.DataFrame,
        schedule: pd.DataFrame,
        player_availability: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Generate all features for a single game.
        
        This is the main entry point for feature generation, combining
        all three feature categories into a single feature dictionary.
        
        Args:
            home_team: Home team code
            away_team: Away team code
            game_date: Date of the game
            historical_data: Historical game results and stats
            schedule: Season schedule for travel calculations
            player_availability: Optional player availability data
        
        Returns:
            Dictionary of feature names to values
        """
        features: Dict[str, float] = {}
        
        # ----------------------------------------------------------------
        # Rolling Efficiency Features
        # ----------------------------------------------------------------
        home_efficiency = self.get_rolling_efficiency(
            home_team, game_date, historical_data
        )
        away_efficiency = self.get_rolling_efficiency(
            away_team, game_date, historical_data
        )
        
        if home_efficiency:
            features['home_ortg_l5'] = home_efficiency.offensive_rating
            features['home_drtg_l5'] = home_efficiency.defensive_rating
            features['home_netrtg_l5'] = home_efficiency.net_rating
            features['home_pace_l5'] = home_efficiency.pace
        else:
            features['home_ortg_l5'] = np.nan
            features['home_drtg_l5'] = np.nan
            features['home_netrtg_l5'] = np.nan
            features['home_pace_l5'] = np.nan
        
        if away_efficiency:
            features['away_ortg_l5'] = away_efficiency.offensive_rating
            features['away_drtg_l5'] = away_efficiency.defensive_rating
            features['away_netrtg_l5'] = away_efficiency.net_rating
            features['away_pace_l5'] = away_efficiency.pace
        else:
            features['away_ortg_l5'] = np.nan
            features['away_drtg_l5'] = np.nan
            features['away_netrtg_l5'] = np.nan
            features['away_pace_l5'] = np.nan
        
        # Differential features
        if home_efficiency and away_efficiency:
            features['ortg_diff'] = (
                home_efficiency.offensive_rating - 
                away_efficiency.offensive_rating
            )
            features['drtg_diff'] = (
                home_efficiency.defensive_rating - 
                away_efficiency.defensive_rating
            )
            features['netrtg_diff'] = (
                home_efficiency.net_rating - 
                away_efficiency.net_rating
            )
        else:
            features['ortg_diff'] = np.nan
            features['drtg_diff'] = np.nan
            features['netrtg_diff'] = np.nan
        
        # ----------------------------------------------------------------
        # Travel Fatigue Features
        # ----------------------------------------------------------------
        home_travel = self.calculate_travel_fatigue(
            home_team, game_date, schedule
        )
        away_travel = self.calculate_travel_fatigue(
            away_team, game_date, schedule
        )
        
        features['home_travel_fatigue'] = float(home_travel.score)
        features['home_distance_km'] = home_travel.distance_km
        features['home_back_to_back'] = float(home_travel.back_to_back)
        features['home_days_rest'] = float(home_travel.days_rest)
        
        features['away_travel_fatigue'] = float(away_travel.score)
        features['away_distance_km'] = away_travel.distance_km
        features['away_back_to_back'] = float(away_travel.back_to_back)
        features['away_days_rest'] = float(away_travel.days_rest)
        
        # Travel differential (positive = away team more fatigued)
        features['travel_fatigue_diff'] = (
            away_travel.score - home_travel.score
        )
        features['rest_diff'] = (
            home_travel.days_rest - away_travel.days_rest
        )
        
        # ----------------------------------------------------------------
        # Import Availability Features
        # ----------------------------------------------------------------
        if player_availability is not None:
            home_imports = self.check_import_availability(
                home_team, game_date, player_availability
            )
            away_imports = self.check_import_availability(
                away_team, game_date, player_availability
            )
            
            features['home_imports_available'] = float(
                home_imports['all_imports_available']
            )
            features['home_import_count'] = float(
                home_imports['import_count_available']
            )
            features['away_imports_available'] = float(
                away_imports['all_imports_available']
            )
            features['away_import_count'] = float(
                away_imports['import_count_available']
            )
            
            # Import advantage
            features['import_advantage'] = (
                home_imports['import_count_available'] - 
                away_imports['import_count_available']
            )
        else:
            features['home_imports_available'] = np.nan
            features['home_import_count'] = np.nan
            features['away_imports_available'] = np.nan
            features['away_import_count'] = np.nan
            features['import_advantage'] = np.nan
        
        return features
    
    def generate_dataset(
        self,
        games: pd.DataFrame,
        historical_data: pd.DataFrame,
        schedule: pd.DataFrame,
        player_availability: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate features for multiple games.
        
        Args:
            games: DataFrame with games to generate features for
                   Must have: home_team, away_team, game_date
            historical_data: Historical game results
            schedule: Season schedule
            player_availability: Optional player data
        
        Returns:
            DataFrame with all features for each game
        """
        feature_rows = []
        
        for _, game in games.iterrows():
            features = self.generate_game_features(
                home_team=game['home_team'],
                away_team=game['away_team'],
                game_date=game['game_date'],
                historical_data=historical_data,
                schedule=schedule,
                player_availability=player_availability
            )
            features['game_id'] = game.get('game_id', '')
            features['home_team'] = game['home_team']
            features['away_team'] = game['away_team']
            features['game_date'] = game['game_date']
            
            feature_rows.append(features)
        
        return pd.DataFrame(feature_rows)

"""
NBL/WNBL Feature Engineering Module.

This module implements the NBLFeatureEngineer class which generates
domain-specific features for Australian basketball betting models.

Features generated:
    1. Rolling Efficiency (Last 5 games Offensive/Defensive Rating)
    2. Travel Fatigue Score (Distance-based with Perth-NZ weighting)
    3. Import Availability (Boolean for key import players)
    4. Advanced Metrics (BPM, SOS, Expected Wins)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .advanced_metrics import AdvancedMetricsCalculator


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
        self._advanced_calc = AdvancedMetricsCalculator()
    
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
        matching_games = schedule[
            schedule['game_date'] == game_date
        ]

        if matching_games.empty:
            return TravelFatigueResult(
                score=0.0,
                distance_km=0.0,
                route_type="Unknown",
                back_to_back=False,
                days_rest=7
            )

        current_game = matching_games.iloc[0]
        
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
    # Rest Days Features
    # ========================================================================
    
    def get_rest_days(
        self,
        team_code: str,
        game_date: datetime,
        schedule: pd.DataFrame
    ) -> int:
        """
        Calculate days of rest since team's last game.
        
        Rest is a critical factor in basketball performance:
        - Back-to-back (0-1 days): Significant fatigue
        - Normal rest (2-3 days): Standard
        - Extended rest (4+ days): Well-rested, possible rust
        
        Args:
            team_code: Team identifier
            game_date: Date of upcoming game
            schedule: DataFrame with game schedule
        
        Returns:
            Days since last game (999 if no previous game found)
        """
        team_games = schedule[
            ((schedule['home_team'] == team_code) |
             (schedule['away_team'] == team_code)) &
            (schedule['game_date'] < game_date)
        ].sort_values('game_date', ascending=False)
        
        if team_games.empty:
            return 999  # No previous game, treat as well-rested
        
        last_game_date = team_games.iloc[0]['game_date']
        
        # Handle both datetime and date objects
        if hasattr(last_game_date, 'date'):
            last_date = last_game_date.date()
        else:
            last_date = last_game_date
            
        if hasattr(game_date, 'date'):
            current_date = game_date.date()
        else:
            current_date = game_date
        
        return (current_date - last_date).days
    
    def calculate_rest_advantage(
        self,
        home_rest: int,
        away_rest: int
    ) -> Dict[str, float]:
        """
        Calculate rest-related features from rest days.
        
        Args:
            home_rest: Days rest for home team
            away_rest: Days rest for away team
        
        Returns:
            Dictionary with rest features:
                - rest_diff: home_rest - away_rest
                - home_b2b: 1 if home team on back-to-back
                - away_b2b: 1 if away team on back-to-back
                - rest_advantage_category: -1/0/1 for disadvantage/neutral/advantage
        """
        rest_diff = home_rest - away_rest
        
        # Categorize rest advantage
        if rest_diff >= 2:
            rest_category = 1  # Home team rested advantage
        elif rest_diff <= -2:
            rest_category = -1  # Away team rested advantage
        else:
            rest_category = 0  # Neutral
        
        return {
            'rest_diff': float(rest_diff),
            'home_b2b': float(home_rest <= 1),
            'away_b2b': float(away_rest <= 1),
            'rest_advantage_category': float(rest_category)
        }
    
    # ========================================================================
    # Injury Impact Features
    # ========================================================================
    
    def calculate_injury_impact(
        self,
        team_code: str,
        game_date: datetime,
        injury_data: Optional[pd.DataFrame],
        historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate expected performance impact from injuries/absences.
        
        Uses player minutes and efficiency to estimate impact when 
        key players are unavailable.
        
        Args:
            team_code: Team identifier
            game_date: Date of the game
            injury_data: DataFrame with columns:
                - player_id: Player identifier
                - team: Team code
                - status: 'out', 'doubtful', 'questionable', 'probable'
                - avg_minutes: Season average minutes per game
                - efficiency: Player efficiency rating (optional)
            historical_data: For calculating team averages
        
        Returns:
            Dictionary with:
                - injury_impact_score: 0-1 scale (0 = no impact, 1 = critical)
                - players_out: Number of players listed as out
                - minutes_lost: Expected minutes lost
                - key_players_out: Number of starters/key rotation players out
        """
        if injury_data is None or injury_data.empty:
            return {
                'injury_impact_score': 0.0,
                'players_out': 0,
                'minutes_lost': 0.0,
                'key_players_out': 0
            }
        
        # Filter to team's injured/questionable players for this game
        team_injuries = injury_data[
            (injury_data['team'] == team_code)
        ]
        
        if team_injuries.empty:
            return {
                'injury_impact_score': 0.0,
                'players_out': 0,
                'minutes_lost': 0.0,
                'key_players_out': 0
            }
        
        # Weight by status
        status_weights = {
            'out': 1.0,
            'doubtful': 0.75,
            'questionable': 0.5,
            'probable': 0.1,
            'available': 0.0
        }
        
        total_minutes_lost = 0.0
        weighted_efficiency_lost = 0.0
        players_out = 0
        key_players_out = 0  # Players with 20+ min/game
        
        for _, player in team_injuries.iterrows():
            status = player.get('status', 'questionable').lower()
            weight = status_weights.get(status, 0.5)
            
            avg_minutes = player.get('avg_minutes', 15.0)
            efficiency = player.get('efficiency', 10.0)
            
            minutes_impact = avg_minutes * weight
            total_minutes_lost += minutes_impact
            weighted_efficiency_lost += efficiency * weight
            
            if weight >= 0.75:  # Doubtful or out
                players_out += 1
                if avg_minutes >= 20:
                    key_players_out += 1
        
        # Normalize impact score (max ~240 total minutes in a game)
        # Key players (20+ min) count more
        impact_score = min(1.0, (total_minutes_lost / 100.0) + (key_players_out * 0.15))
        
        return {
            'injury_impact_score': round(impact_score, 3),
            'players_out': players_out,
            'minutes_lost': round(total_minutes_lost, 1),
            'key_players_out': key_players_out
        }
    
    # ========================================================================
    # Streak and Momentum Features
    # ========================================================================
    
    def get_win_streak(
        self,
        team_code: str,
        game_date: datetime,
        historical_data: pd.DataFrame,
        max_lookback: int = 10
    ) -> int:
        """
        Calculate current win/loss streak.
        
        Positive values indicate consecutive wins, negative indicate losses.
        Streak momentum can be a psychological factor in game outcomes.
        
        Args:
            team_code: Team identifier
            game_date: Date of upcoming game
            historical_data: Historical game results
            max_lookback: Maximum games to look back
        
        Returns:
            Current streak (positive = wins, negative = losses)
            0 if mixed recent results
        """
        team_games = historical_data[
            ((historical_data['home_team'] == team_code) |
             (historical_data['away_team'] == team_code)) &
            (historical_data['game_date'] < game_date)
        ].sort_values('game_date', ascending=False).head(max_lookback)
        
        if team_games.empty:
            return 0
        
        streak = 0
        streak_type = None  # 'W' or 'L'
        
        for _, game in team_games.iterrows():
            # Determine if team won
            if game['home_team'] == team_code:
                won = game['home_score'] > game['away_score']
            else:
                won = game['away_score'] > game['home_score']
            
            current_result = 'W' if won else 'L'
            
            if streak_type is None:
                streak_type = current_result
                streak = 1 if won else -1
            elif current_result == streak_type:
                streak += 1 if won else -1
            else:
                break  # Streak ended
        
        return streak
    
    def get_recent_form(
        self,
        team_code: str,
        game_date: datetime,
        historical_data: pd.DataFrame,
        n_games: int = 5,
        home_away_filter: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate recent form metrics over last N games.
        
        Args:
            team_code: Team identifier
            game_date: Date of upcoming game
            historical_data: Historical game results
            n_games: Number of games to consider
            home_away_filter: 'home', 'away', or None for all games
        
        Returns:
            Dictionary with:
                - win_pct: Win percentage in window
                - avg_margin: Average scoring margin
                - games_played: Actual games in window
        """
        # Filter to team's games
        if home_away_filter == 'home':
            team_games = historical_data[
                (historical_data['home_team'] == team_code) &
                (historical_data['game_date'] < game_date)
            ]
        elif home_away_filter == 'away':
            team_games = historical_data[
                (historical_data['away_team'] == team_code) &
                (historical_data['game_date'] < game_date)
            ]
        else:
            team_games = historical_data[
                ((historical_data['home_team'] == team_code) |
                 (historical_data['away_team'] == team_code)) &
                (historical_data['game_date'] < game_date)
            ]
        
        team_games = team_games.sort_values('game_date', ascending=False).head(n_games)
        
        if team_games.empty:
            return {
                'win_pct': 0.5,  # Default to 50%
                'avg_margin': 0.0,
                'games_played': 0
            }
        
        wins = 0
        total_margin = 0.0
        
        for _, game in team_games.iterrows():
            if game['home_team'] == team_code:
                margin = game['home_score'] - game['away_score']
            else:
                margin = game['away_score'] - game['home_score']
            
            total_margin += margin
            if margin > 0:
                wins += 1
        
        games_played = len(team_games)
        
        return {
            'win_pct': round(wins / games_played, 3) if games_played > 0 else 0.5,
            'avg_margin': round(total_margin / games_played, 1) if games_played > 0 else 0.0,
            'games_played': games_played
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
        
        # ----------------------------------------------------------------
        # Rest Days Features (NEW)
        # ----------------------------------------------------------------
        home_rest = self.get_rest_days(home_team, game_date, schedule)
        away_rest = self.get_rest_days(away_team, game_date, schedule)
        
        features['home_rest_days'] = float(min(home_rest, 14))  # Cap at 14
        features['away_rest_days'] = float(min(away_rest, 14))
        
        rest_features = self.calculate_rest_advantage(home_rest, away_rest)
        features['rest_diff'] = rest_features['rest_diff']
        features['home_b2b'] = rest_features['home_b2b']
        features['away_b2b'] = rest_features['away_b2b']
        features['rest_advantage_category'] = rest_features['rest_advantage_category']
        
        # ----------------------------------------------------------------
        # Streak and Momentum Features (NEW)
        # ----------------------------------------------------------------
        home_streak = self.get_win_streak(home_team, game_date, historical_data)
        away_streak = self.get_win_streak(away_team, game_date, historical_data)
        
        features['home_streak'] = float(home_streak)
        features['away_streak'] = float(away_streak)
        features['streak_diff'] = float(home_streak - away_streak)
        
        # Recent form (overall)
        home_form = self.get_recent_form(home_team, game_date, historical_data)
        away_form = self.get_recent_form(away_team, game_date, historical_data)
        
        features['home_win_pct_l5'] = home_form['win_pct']
        features['home_avg_margin_l5'] = home_form['avg_margin']
        features['away_win_pct_l5'] = away_form['win_pct']
        features['away_avg_margin_l5'] = away_form['avg_margin']
        features['form_diff'] = home_form['win_pct'] - away_form['win_pct']
        
        # Home/Away specific form
        home_home_form = self.get_recent_form(
            home_team, game_date, historical_data, home_away_filter='home'
        )
        away_away_form = self.get_recent_form(
            away_team, game_date, historical_data, home_away_filter='away'
        )
        
        features['home_home_win_pct'] = home_home_form['win_pct']
        features['away_away_win_pct'] = away_away_form['win_pct']
        
        # ----------------------------------------------------------------
        # Four Factors Features (Dean Oliver's Framework)
        # ----------------------------------------------------------------
        # Calculate Four Factors differentials from historical rolling data
        # These are the most predictive metrics in basketball analytics
        four_factors = self._calculate_four_factors_features(
            home_team, away_team, game_date, historical_data
        )
        features.update(four_factors)
        
        # ----------------------------------------------------------------
        # Advanced Metrics Features (BPM, SOS)
        # ----------------------------------------------------------------
        # Calculate BPM and SOS-based features for matchup prediction
        advanced_features = self._compute_advanced_features(
            home_team, away_team, game_date, historical_data
        )
        features.update(advanced_features)
        
        return features
    
    def _calculate_four_factors_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate Dean Oliver's Four Factors differential features.
        
        The Four Factors explain ~90% of winning variance:
            1. eFG% - Effective Field Goal % (shooting efficiency)
            2. TOV% - Turnover Percentage (ball security)
            3. ORB% - Offensive Rebound % (second chances)
            4. FTR - Free Throw Rate (getting to the line)
        
        Returns:
            Dictionary of Four Factors features
        """
        features = {}
        
        # Get last N games for each team
        n_games = self.rolling_window
        
        home_games = historical_data[
            ((historical_data['home_team'] == home_team) |
             (historical_data['away_team'] == home_team)) &
            (historical_data['game_date'] < game_date)
        ].sort_values('game_date', ascending=False).head(n_games)
        
        away_games = historical_data[
            ((historical_data['home_team'] == away_team) |
             (historical_data['away_team'] == away_team)) &
            (historical_data['game_date'] < game_date)
        ].sort_values('game_date', ascending=False).head(n_games)
        
        # Calculate Four Factors for each team
        home_ff = self._compute_team_four_factors(home_games, home_team)
        away_ff = self._compute_team_four_factors(away_games, away_team)
        
        if home_ff and away_ff:
            # Differential features (home - away)
            features['delta_efg'] = home_ff['efg_pct'] - away_ff['efg_pct']
            features['delta_tov'] = home_ff['tov_pct'] - away_ff['tov_pct']
            features['delta_orb'] = home_ff['orb_pct'] - away_ff['orb_pct']
            features['delta_ftr'] = home_ff['ft_rate'] - away_ff['ft_rate']
            
            # Raw values for each team
            features['home_efg_pct'] = home_ff['efg_pct']
            features['home_tov_pct'] = home_ff['tov_pct']
            features['home_orb_pct'] = home_ff['orb_pct']
            features['home_ft_rate'] = home_ff['ft_rate']
            
            features['away_efg_pct'] = away_ff['efg_pct']
            features['away_tov_pct'] = away_ff['tov_pct']
            features['away_orb_pct'] = away_ff['orb_pct']
            features['away_ft_rate'] = away_ff['ft_rate']
            
            # Weighted composite score (Oliver's weights)
            # eFG: 40%, TOV: 25%, ORB: 20%, FTR: 15%
            # Note: TOV is inverted (lower is better)
            features['four_factors_score'] = (
                0.40 * features['delta_efg'] * 100 +
                -0.25 * features['delta_tov'] +  # Inverted
                0.20 * features['delta_orb'] * 100 +
                0.15 * features['delta_ftr'] * 100
            )
        else:
            # Insufficient data
            for key in ['delta_efg', 'delta_tov', 'delta_orb', 'delta_ftr',
                       'home_efg_pct', 'home_tov_pct', 'home_orb_pct', 'home_ft_rate',
                       'away_efg_pct', 'away_tov_pct', 'away_orb_pct', 'away_ft_rate',
                       'four_factors_score']:
                features[key] = np.nan
        
        return features
    
    def _compute_team_four_factors(
        self,
        games: pd.DataFrame,
        team_code: str
    ) -> Optional[Dict[str, float]]:
        """
        Compute Four Factors from a team's recent games.
        
        Args:
            games: DataFrame of team's games
            team_code: Team to compute factors for
        
        Returns:
            Dictionary with efg_pct, tov_pct, orb_pct, ft_rate
        """
        if len(games) < 3:
            return None
        
        # Accumulate stats
        totals = {
            'fgm': 0, 'fg3m': 0, 'fga': 0,
            'turnovers': 0, 'fta': 0, 'ftm': 0,
            'orb': 0, 'opp_drb': 0
        }
        
        for _, game in games.iterrows():
            # Determine if team was home or away
            if game['home_team'] == team_code:
                prefix = 'home_'
                opp_prefix = 'away_'
            else:
                prefix = 'away_'
                opp_prefix = 'home_'
            
            # Safely get stats with defaults
            totals['fgm'] += game.get(f'{prefix}fgm', 0) or 0
            totals['fg3m'] += game.get(f'{prefix}fg3m', 0) or 0
            totals['fga'] += game.get(f'{prefix}fga', 0) or 0
            totals['turnovers'] += game.get(f'{prefix}turnovers', 0) or 0
            totals['fta'] += game.get(f'{prefix}fta', 0) or 0
            totals['ftm'] += game.get(f'{prefix}ftm', 0) or 0
            totals['orb'] += game.get(f'{prefix}oreb', game.get(f'{prefix}orb', 0)) or 0
            totals['opp_drb'] += game.get(f'{opp_prefix}dreb', game.get(f'{opp_prefix}drb', 0)) or 0
        
        # Calculate Four Factors
        fga = totals['fga']
        if fga == 0:
            return None
        
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        efg_pct = (totals['fgm'] + 0.5 * totals['fg3m']) / fga
        
        # TOV% = 100 * TO / (FGA + 0.44*FTA + TO)
        plays = fga + 0.44 * totals['fta'] + totals['turnovers']
        tov_pct = 100 * totals['turnovers'] / plays if plays > 0 else 0
        
        # ORB% = ORB / (ORB + Opp_DRB)
        total_reb = totals['orb'] + totals['opp_drb']
        orb_pct = totals['orb'] / total_reb if total_reb > 0 else 0
        
        # FTR = FT / FGA (Oliver's definition)
        ft_rate = totals['ftm'] / fga
        
        return {
            'efg_pct': round(efg_pct, 4),
            'tov_pct': round(tov_pct, 2),
            'orb_pct': round(orb_pct, 4),
            'ft_rate': round(ft_rate, 4)
        }
    
    # ========================================================================
    # Advanced Metrics Features (BPM, SOS)
    # ========================================================================
    
    def _compute_advanced_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        historical_data: pd.DataFrame,
        player_box_scores: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Compute advanced metrics features (BPM, SOS, Expected Wins).
        
        These metrics provide additional signal for game predictions by
        estimating player-level contributions and schedule-adjusted performance.
        
        Args:
            home_team: Home team code
            away_team: Away team code
            game_date: Date of the game
            historical_data: Historical game results
            player_box_scores: Optional player-level box scores for BPM
        
        Returns:
            Dictionary with advanced metric features
        """
        features: Dict[str, float] = {}
        n_games = self.rolling_window
        
        # Get team ratings for SOS calculation
        all_teams = set(
            historical_data['home_team'].unique().tolist() +
            historical_data['away_team'].unique().tolist()
        )
        
        team_ratings = {}
        for team in all_teams:
            team_games = historical_data[
                ((historical_data['home_team'] == team) |
                 (historical_data['away_team'] == team)) &
                (historical_data['game_date'] < game_date)
            ].tail(n_games)
            
            if len(team_games) >= 3:
                # Calculate simple rating from point differential
                margins = []
                for _, g in team_games.iterrows():
                    if g['home_team'] == team:
                        margins.append(g['home_score'] - g['away_score'])
                    else:
                        margins.append(g['away_score'] - g['home_score'])
                team_ratings[team] = np.mean(margins) / 10.0  # Normalize
        
        # Calculate SOS for both teams
        if team_ratings:
            home_sos = self._advanced_calc.calculate_sos(
                historical_data[historical_data['game_date'] < game_date],
                home_team,
                team_ratings
            )
            away_sos = self._advanced_calc.calculate_sos(
                historical_data[historical_data['game_date'] < game_date],
                away_team,
                team_ratings
            )
            features['home_sos'] = home_sos
            features['away_sos'] = away_sos
            features['sos_diff'] = home_sos - away_sos
        else:
            features['home_sos'] = 0.0
            features['away_sos'] = 0.0
            features['sos_diff'] = 0.0
        
        # Calculate team record and SOS-adjusted win %
        for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
            team_games = historical_data[
                ((historical_data['home_team'] == team) |
                 (historical_data['away_team'] == team)) &
                (historical_data['game_date'] < game_date)
            ].tail(n_games)
            
            if len(team_games) >= 3:
                wins = 0
                points_for = 0
                points_against = 0
                
                for _, g in team_games.iterrows():
                    if g['home_team'] == team:
                        won = g['home_score'] > g['away_score']
                        points_for += g['home_score']
                        points_against += g['away_score']
                    else:
                        won = g['away_score'] > g['home_score']
                        points_for += g['away_score']
                        points_against += g['home_score']
                    wins += int(won)
                
                win_pct = wins / len(team_games)
                sos = features.get(f'{prefix}_sos', 0.0)
                
                features[f'{prefix}_sos_adj_win_pct'] = (
                    self._advanced_calc.calculate_sos_adjusted_win_pct(win_pct, sos)
                )
                features[f'{prefix}_expected_wins'] = (
                    self._advanced_calc.calculate_expected_wins(
                        points_for, points_against, len(team_games)
                    )
                )
            else:
                features[f'{prefix}_sos_adj_win_pct'] = 0.5
                features[f'{prefix}_expected_wins'] = 0.0
        
        features['sos_adj_win_pct_diff'] = (
            features['home_sos_adj_win_pct'] - features['away_sos_adj_win_pct']
        )
        
        # Calculate BPM features if player data available
        if player_box_scores is not None and not player_box_scores.empty:
            home_bpm = self._advanced_calc.calculate_team_bpm(
                player_box_scores, home_team, n_games
            )
            away_bpm = self._advanced_calc.calculate_team_bpm(
                player_box_scores, away_team, n_games
            )
            
            features['home_bpm'] = home_bpm
            features['away_bpm'] = away_bpm
            features['bpm_differential'] = home_bpm - away_bpm
        else:
            # BPM not available without player data
            features['home_bpm'] = np.nan
            features['away_bpm'] = np.nan
            features['bpm_differential'] = np.nan
        
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

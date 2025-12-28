"""
Dean Oliver's Four Factors Analysis for NBL Betting.

This module implements comprehensive Four Factors analysis including:
    1. eFG% (Effective Field Goal Percentage) - Shooting efficiency
    2. TOV% (Turnover Percentage) - Ball security
    3. ORB% (Offensive Rebound Percentage) - Second chances
    4. FTR (Free Throw Rate) - Getting to the line

The Four Factors explain ~90% of the variance in NBA/NBL outcomes.
Oliver's original weights: 40% shooting, 25% turnovers, 20% rebounding, 15% FT.

References:
    - Oliver, D. (2004). "Basketball on Paper"
    - Kubatko et al. (2007). "A Starting Point for Analyzing Basketball Statistics"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
import warnings


@dataclass
class FourFactors:
    """
    Container for Dean Oliver's Four Factors.
    
    Attributes:
        efg_pct: Effective Field Goal % = (FGM + 0.5*3PM) / FGA
        tov_pct: Turnover % = 100 * TO / (FGA + 0.44*FTA + TO)
        orb_pct: Offensive Rebound % = ORB / (ORB + Opp_DRB)
        ft_rate: Free Throw Rate = FT / FGA (NOT FTA/FGA, per Oliver)
    """
    efg_pct: float
    tov_pct: float
    orb_pct: float
    ft_rate: float
    
    @property
    def as_dict(self) -> Dict[str, float]:
        """Return as dictionary for DataFrame compatibility."""
        return {
            'efg_pct': self.efg_pct,
            'tov_pct': self.tov_pct,
            'orb_pct': self.orb_pct,
            'ft_rate': self.ft_rate
        }
    
    @property
    def as_array(self) -> np.ndarray:
        """Return as numpy array for model input."""
        return np.array([self.efg_pct, self.tov_pct, self.orb_pct, self.ft_rate])


@dataclass
class FourFactorsDifferential:
    """
    Differential between team and opponent Four Factors.
    
    Positive values favor the team (except TOV% where negative is better).
    """
    delta_efg: float  # Team eFG% - Opp eFG%
    delta_tov: float  # Team TOV% - Opp TOV% (NEGATIVE is better)
    delta_orb: float  # Team ORB% - Opp ORB%
    delta_ftr: float  # Team FTR - Opp FTR
    
    @property
    def weighted_score(self) -> float:
        """
        Calculate weighted differential using Oliver's weights.
        
        Weights: eFG 40%, TOV 25%, ORB 20%, FT 15%
        Note: TOV is inverted since lower is better.
        """
        return (
            0.40 * self.delta_efg * 100 +  # Scale to percentage points
            -0.25 * self.delta_tov +       # Invert: our low TOV is good
            0.20 * self.delta_orb * 100 +
            0.15 * self.delta_ftr * 100
        )


@dataclass 
class CorrelationResult:
    """Result of correlation analysis between a metric and wins."""
    metric_name: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    is_significant: bool  # p < 0.05
    
    def __str__(self) -> str:
        sig_str = "✓" if self.is_significant else "✗"
        return (
            f"{self.metric_name}: r={self.pearson_r:+.3f} (p={self.pearson_p:.3f}) "
            f"ρ={self.spearman_rho:+.3f} {sig_str}"
        )


@dataclass
class DataQualityReport:
    """Data quality assessment for Four Factors analysis."""
    total_games: int
    games_with_complete_data: int
    completeness_pct: float
    missing_fields: Dict[str, int]
    date_range: Tuple[str, str]
    teams_with_data: List[str]
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"Data Quality Report\n"
            f"{'='*50}\n"
            f"Total Games: {self.total_games}\n"
            f"Complete Data: {self.games_with_complete_data} ({self.completeness_pct:.1f}%)\n"
            f"Date Range: {self.date_range[0]} to {self.date_range[1]}\n"
            f"Teams: {len(self.teams_with_data)}\n"
            f"Warnings: {len(self.warnings)}"
        )


# =============================================================================
# NBL Pace Calculator
# =============================================================================

@dataclass
class PaceMetrics:
    """Container for pace-related metrics."""
    
    possessions: float  # Estimated total possessions
    pace: float  # Per-40-minute (FIBA standard)
    pace_48: float  # Normalized to NBA 48-minute standard
    
    # NBL league averages for context
    NBL_AVG_PACE = 75.0  # Approximate NBL average per 40-min
    NBA_AVG_PACE = 100.0  # Approximate NBA average per 48-min
    
    @property
    def pace_relative_to_league(self) -> float:
        """Pace relative to NBL average (1.0 = average)."""
        return self.pace / self.NBL_AVG_PACE if self.NBL_AVG_PACE > 0 else 1.0
    
    @property
    def is_high_pace(self) -> bool:
        """True if above league average pace."""
        return self.pace > self.NBL_AVG_PACE * 1.05  # 5% above average


class PaceCalculator:
    """
    Calculate pace and possession metrics for NBL games.
    
    NBL uses FIBA rules (40-minute games), so pace is naturally lower
    than NBA 48-minute games. This calculator provides both FIBA and
    NBA-normalized pace for comparison.
    
    NBL Pace Context:
        - NBL average: ~75 possessions per 40 min
        - High pace teams: 78+ possessions
        - Slow pace teams: 72 or less
    
    Example:
        >>> calc = PaceCalculator()
        >>> metrics = calc.calculate_game_pace(
        ...     fga=80, oreb=10, turnovers=15, fta=20
        ... )
        >>> print(f"Pace: {metrics.pace:.1f}")
    """
    
    # FIBA game length in minutes
    FIBA_GAME_MINUTES = 40
    NBA_GAME_MINUTES = 48
    
    # Free throw possession coefficient (industry standard)
    FT_COEFFICIENT = 0.44
    
    @staticmethod
    def estimate_possessions(
        fga: float,
        oreb: float,
        turnovers: float,
        fta: float
    ) -> float:
        """
        Estimate possessions using the standard formula.
        
        Poss = FGA - OREB + TO + 0.44 * FTA
        
        The 0.44 FTA coefficient accounts for:
        - And-1 plays (shot + free throws = 1 possession)
        - Technical fouls (not possession-ending)
        - 3-shot fouls (3 FTA per possession)
        """
        return fga - oreb + turnovers + (0.44 * fta)
    
    def calculate_game_pace(
        self,
        fga: float,
        oreb: float,
        turnovers: float,
        fta: float,
        minutes_played: float = 40.0
    ) -> PaceMetrics:
        """
        Calculate pace metrics for a single team's game.
        
        Args:
            fga: Field goals attempted
            oreb: Offensive rebounds
            turnovers: Turnovers committed
            fta: Free throws attempted
            minutes_played: Actual minutes (default 40 for FIBA)
            
        Returns:
            PaceMetrics with possessions and pace calculations
        """
        possessions = self.estimate_possessions(fga, oreb, turnovers, fta)
        
        # Pace per 40 minutes (FIBA standard)
        if minutes_played > 0:
            pace = (possessions / minutes_played) * self.FIBA_GAME_MINUTES
        else:
            pace = possessions
        
        # Pace normalized to 48 minutes (for NBA comparison)
        pace_48 = (pace / self.FIBA_GAME_MINUTES) * self.NBA_GAME_MINUTES
        
        return PaceMetrics(
            possessions=possessions,
            pace=pace,
            pace_48=pace_48
        )
    
    def calculate_game_average_pace(
        self,
        home_fga: float, home_oreb: float, home_to: float, home_fta: float,
        away_fga: float, away_oreb: float, away_to: float, away_fta: float
    ) -> PaceMetrics:
        """
        Calculate game-level pace (average of both teams).
        
        This is the preferred method for game-level analysis since
        both teams share the same number of possessions.
        """
        home_poss = self.estimate_possessions(home_fga, home_oreb, home_to, home_fta)
        away_poss = self.estimate_possessions(away_fga, away_oreb, away_to, away_fta)
        
        # Average possessions (should be roughly equal in a game)
        avg_poss = (home_poss + away_poss) / 2
        
        pace = avg_poss  # For 40-minute game
        pace_48 = (pace / self.FIBA_GAME_MINUTES) * self.NBA_GAME_MINUTES
        
        return PaceMetrics(
            possessions=avg_poss,
            pace=pace,
            pace_48=pace_48
        )
    
    def calculate_nbl_pace_impact(
        self,
        team_pace: float,
        opponent_pace: float
    ) -> Dict[str, float]:
        """
        Analyze pace matchup impact for NBL games.
        
        High-pace vs low-pace matchups can significantly affect
        game outcomes and betting value.
        
        Returns:
            Dictionary with pace analysis metrics
        """
        pace_diff = team_pace - opponent_pace
        avg_pace = (team_pace + opponent_pace) / 2
        expected_pace = avg_pace  # Simple average for expected game pace
        
        return {
            'team_pace': team_pace,
            'opponent_pace': opponent_pace,
            'pace_differential': pace_diff,
            'expected_game_pace': expected_pace,
            'is_pace_advantage': pace_diff > 0,  # Team prefers faster pace
            'pace_mismatch': abs(pace_diff) > 3.0,  # Significant mismatch
        }


# =============================================================================
# Rolling Four Factors Calculator
# =============================================================================

class RollingFourFactors:
    """
    Calculate rolling Four Factors over recent games.
    
    This is essential for prediction - we use a team's recent performance
    (not full-season averages) to predict upcoming games.
    
    Example:
        >>> rolling = RollingFourFactors(window=5)
        >>> features = rolling.calculate_rolling_differentials(
        ...     df, team='MEL', opponent='SYD', game_date='2024-01-15'
        ... )
    """
    
    def __init__(self, window: int = 5):
        """
        Initialize with rolling window size.
        
        Args:
            window: Number of recent games to include (default 5)
        """
        self.window = window
        self.calculator = FourFactorsCalculator()
    
    def calculate_team_rolling_factors(
        self,
        games_df: pd.DataFrame,
        team_code: str,
        as_of_date: pd.Timestamp,
        min_games: int = 3
    ) -> Optional[FourFactors]:
        """
        Calculate rolling average Four Factors for a team.
        
        Only uses games BEFORE as_of_date to prevent look-ahead bias.
        
        Args:
            games_df: DataFrame with game-level Four Factors
            team_code: Team to calculate for (home_team or away_team)
            as_of_date: Calculate factors using games before this date
            min_games: Minimum games required (returns None if fewer)
            
        Returns:
            FourFactors with rolling averages, or None if insufficient data
        """
        # Ensure game_date is datetime
        if 'game_date' in games_df.columns:
            games_df = games_df.copy()
            games_df['game_date'] = pd.to_datetime(games_df['game_date'])
        else:
            return None
        
        # Get games before cutoff date
        prior_games = games_df[games_df['game_date'] < as_of_date]
        
        # Find games where team played (home or away)
        home_mask = prior_games.get('home_team', pd.Series()) == team_code
        away_mask = prior_games.get('away_team', pd.Series()) == team_code
        team_games = prior_games[home_mask | away_mask].copy()
        
        if len(team_games) < min_games:
            return None
        
        # Take last N games
        team_games = team_games.sort_values('game_date').tail(self.window)
        
        # Calculate factors for each game from team's perspective
        efg_list, tov_list, orb_list, ftr_list = [], [], [], []
        
        for _, game in team_games.iterrows():
            is_home = game.get('home_team') == team_code
            prefix = 'home_' if is_home else 'away_'
            opp_prefix = 'away_' if is_home else 'home_'
            
            # Get team's stats
            fgm = game.get(f'{prefix}fgm', 0)
            fg3m = game.get(f'{prefix}fg3m', 0)
            fga = game.get(f'{prefix}fga', 0)
            ftm = game.get(f'{prefix}ftm', 0)
            fta = game.get(f'{prefix}fta', 0)
            turnovers = game.get(f'{prefix}turnovers', 0)
            orb = game.get(f'{prefix}orb', 0)
            opp_drb = game.get(f'{opp_prefix}drb', 0)
            
            if fga > 0:
                efg_list.append(self.calculator.calculate_efg_pct(fgm, fg3m, fga))
                tov_list.append(self.calculator.calculate_tov_pct(turnovers, fga, fta))
                orb_list.append(self.calculator.calculate_orb_pct(orb, opp_drb))
                ftr_list.append(self.calculator.calculate_ft_rate(ftm, fga))
        
        if not efg_list:
            return None
        
        return FourFactors(
            efg_pct=np.mean(efg_list),
            tov_pct=np.mean(tov_list),
            orb_pct=np.mean(orb_list),
            ft_rate=np.mean(ftr_list)
        )
    
    def calculate_rolling_differentials(
        self,
        games_df: pd.DataFrame,
        home_team: str,
        away_team: str,
        game_date: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """
        Calculate Four Factors differential between two teams.
        
        This is the primary feature for win prediction - the relative
        strength in each factor between home and away teams.
        
        Args:
            games_df: Historical game data
            home_team: Home team code
            away_team: Away team code
            game_date: Date of game being predicted
            
        Returns:
            Dictionary with delta_efg, delta_tov, delta_orb, delta_ftr
            and weighted_score, or None if insufficient data
        """
        home_factors = self.calculate_team_rolling_factors(
            games_df, home_team, game_date
        )
        away_factors = self.calculate_team_rolling_factors(
            games_df, away_team, game_date
        )
        
        if home_factors is None or away_factors is None:
            return None
        
        diff = self.calculator.calculate_differential(home_factors, away_factors)
        
        return {
            'delta_efg': diff.delta_efg,
            'delta_tov': diff.delta_tov,
            'delta_orb': diff.delta_orb,
            'delta_ftr': diff.delta_ftr,
            'weighted_score': diff.weighted_score,
            'home_efg': home_factors.efg_pct,
            'home_tov': home_factors.tov_pct,
            'away_efg': away_factors.efg_pct,
            'away_tov': away_factors.tov_pct,
        }
    
    def add_rolling_features_to_df(
        self,
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add rolling Four Factors features to entire DataFrame.
        
        For each game, calculates the rolling differentials using
        only data available before that game (temporal validation).
        
        Args:
            games_df: DataFrame with game data
            
        Returns:
            DataFrame with new rolling feature columns
        """
        df = games_df.copy()
        df = df.sort_values('game_date').reset_index(drop=True)
        
        feature_cols = [
            'roll_delta_efg', 'roll_delta_tov', 'roll_delta_orb', 
            'roll_delta_ftr', 'roll_weighted_score'
        ]
        
        for col in feature_cols:
            df[col] = np.nan
        
        for idx, row in df.iterrows():
            if idx < self.window:  # Skip first N games
                continue
            
            home_team = row.get('home_team')
            away_team = row.get('away_team')
            game_date = pd.to_datetime(row.get('game_date'))
            
            if not home_team or not away_team:
                continue
            
            diffs = self.calculate_rolling_differentials(
                df.iloc[:idx],  # Only use prior games
                home_team, away_team, game_date
            )
            
            if diffs:
                df.loc[idx, 'roll_delta_efg'] = diffs['delta_efg']
                df.loc[idx, 'roll_delta_tov'] = diffs['delta_tov']
                df.loc[idx, 'roll_delta_orb'] = diffs['delta_orb']
                df.loc[idx, 'roll_delta_ftr'] = diffs['delta_ftr']
                df.loc[idx, 'roll_weighted_score'] = diffs['weighted_score']
        
        return df


class FourFactorsCalculator:
    """
    Calculate Dean Oliver's Four Factors from box score data.
    
    Handles FIBA 40-minute game adjustments for NBL/WNBL.
    """
    
    @staticmethod
    def calculate_efg_pct(fgm: float, fg3m: float, fga: float) -> float:
        """
        Calculate Effective Field Goal Percentage.
        
        eFG% = (FGM + 0.5 * 3PM) / FGA
        
        This weights 3-pointers at 1.5x since they're worth 50% more.
        """
        if fga == 0:
            return 0.0
        return (fgm + 0.5 * fg3m) / fga
    
    @staticmethod
    def calculate_tov_pct(
        turnovers: float, 
        fga: float, 
        fta: float
    ) -> float:
        """
        Calculate Turnover Percentage.
        
        TOV% = 100 * TO / (FGA + 0.44*FTA + TO)
        
        Represents turnovers per 100 plays (possessions).
        The 0.44 coefficient adjusts for free throw possession usage.
        """
        plays = fga + 0.44 * fta + turnovers
        if plays == 0:
            return 0.0
        return 100 * turnovers / plays
    
    @staticmethod
    def calculate_orb_pct(
        orb_team: float,
        drb_opponent: float
    ) -> float:
        """
        Calculate Offensive Rebound Percentage.
        
        ORB% = ORB / (ORB + Opp_DRB)
        
        Percentage of available offensive rebounds grabbed by team.
        NBL average is typically around 25-28%.
        """
        total = orb_team + drb_opponent
        if total == 0:
            return 0.0
        return orb_team / total
    
    @staticmethod
    def calculate_ft_rate(ft_made: float, fga: float) -> float:
        """
        Calculate Free Throw Rate.
        
        FTR = FT / FGA (Oliver's definition, NOT FTA/FGA)
        
        Using FT made (not attempted) captures both getting to the
        line AND making them.
        """
        if fga == 0:
            return 0.0
        return ft_made / fga
    
    def calculate_game_factors(
        self,
        fgm: float,
        fg3m: float,
        fga: float,
        turnovers: float,
        fta: float,
        ft_made: float,
        orb: float,
        opp_drb: float
    ) -> FourFactors:
        """
        Calculate all Four Factors from box score stats.
        
        Args:
            fgm: Field goals made
            fg3m: Three-pointers made
            fga: Field goals attempted
            turnovers: Turnovers committed
            fta: Free throws attempted
            ft_made: Free throws made
            orb: Offensive rebounds
            opp_drb: Opponent defensive rebounds
        
        Returns:
            FourFactors object with all metrics
        """
        return FourFactors(
            efg_pct=self.calculate_efg_pct(fgm, fg3m, fga),
            tov_pct=self.calculate_tov_pct(turnovers, fga, fta),
            orb_pct=self.calculate_orb_pct(orb, opp_drb),
            ft_rate=self.calculate_ft_rate(ft_made, fga)
        )
    
    def calculate_differential(
        self,
        team_factors: FourFactors,
        opp_factors: FourFactors
    ) -> FourFactorsDifferential:
        """
        Calculate the differential between team and opponent factors.
        
        Used for predictive modeling - the differential is the
        predictive signal, not the raw factors themselves.
        """
        return FourFactorsDifferential(
            delta_efg=team_factors.efg_pct - opp_factors.efg_pct,
            delta_tov=team_factors.tov_pct - opp_factors.tov_pct,
            delta_orb=team_factors.orb_pct - opp_factors.orb_pct,
            delta_ftr=team_factors.ft_rate - opp_factors.ft_rate
        )


class FourFactorsAnalyzer:
    """
    Correlation and regression analysis for Four Factors.
    
    Determines which factors have the strongest relationship 
    with winning in NBL games.
    """
    
    OLIVER_WEIGHTS = {
        'efg_pct': 0.40,
        'tov_pct': 0.25,
        'orb_pct': 0.20,
        'ft_rate': 0.15
    }
    
    REQUIRED_COLUMNS = [
        'home_fgm', 'home_fg3m', 'home_fga', 'home_turnovers', 
        'home_fta', 'home_ftm', 'home_orb', 'home_drb',
        'away_fgm', 'away_fg3m', 'away_fga', 'away_turnovers',
        'away_fta', 'away_ftm', 'away_orb', 'away_drb',
        'home_score', 'away_score', 'game_date'
    ]
    
    def __init__(self):
        self.calculator = FourFactorsCalculator()
        self.scaler = StandardScaler()
    
    def assess_data_quality(
        self,
        games_df: pd.DataFrame
    ) -> DataQualityReport:
        """
        Generate data quality report for the input data.
        
        Checks for:
        - Missing required columns
        - Null/zero values
        - Date range coverage
        - Team representation
        """
        warnings_list = []
        
        # Check for required columns
        missing = set(self.REQUIRED_COLUMNS) - set(games_df.columns)
        if missing:
            warnings_list.append(f"Missing columns: {missing}")
        
        # Check for complete data rows
        available_cols = [c for c in self.REQUIRED_COLUMNS if c in games_df.columns]
        complete_mask = games_df[available_cols].notna().all(axis=1)
        games_complete = complete_mask.sum()
        
        # Missing field counts
        missing_fields = {}
        for col in available_cols:
            null_count = games_df[col].isna().sum()
            if null_count > 0:
                missing_fields[col] = null_count
        
        # Check for zeros that might indicate missing data
        for col in ['home_fga', 'away_fga']:
            if col in games_df.columns:
                zero_count = (games_df[col] == 0).sum()
                if zero_count > 0:
                    warnings_list.append(f"{col} has {zero_count} zero values")
        
        # Date range
        if 'game_date' in games_df.columns:
            try:
                # Drop NaN values before min/max
                valid_dates = games_df['game_date'].dropna()
                if len(valid_dates) > 0:
                    date_min = str(valid_dates.min())[:10]
                    date_max = str(valid_dates.max())[:10]
                else:
                    date_min = date_max = "Unknown"
            except Exception:
                date_min = date_max = "Unknown"
        else:
            date_min = date_max = "Unknown"
            
        # Teams
        teams = set()
        if 'home_team' in games_df.columns:
            teams.update(games_df['home_team'].unique())
        if 'away_team' in games_df.columns:
            teams.update(games_df['away_team'].unique())
        
        # Sample size warning
        if len(games_df) < 100:
            warnings_list.append(
                f"Small sample size ({len(games_df)} games). "
                "Results may not be statistically robust."
            )
        
        return DataQualityReport(
            total_games=len(games_df),
            games_with_complete_data=int(games_complete),
            completeness_pct=100 * games_complete / max(1, len(games_df)),
            missing_fields=missing_fields,
            date_range=(date_min, date_max),
            teams_with_data=sorted(teams),
            warnings=warnings_list
        )
    
    def compute_four_factors_for_games(
        self,
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute Four Factors for all games in DataFrame.
        
        Returns new DataFrame with Four Factors columns appended.
        """
        calc = self.calculator
        
        result = games_df.copy()
        
        # Home team factors
        result['home_efg_pct'] = games_df.apply(
            lambda r: calc.calculate_efg_pct(
                r['home_fgm'], r['home_fg3m'], r['home_fga']
            ), axis=1
        )
        result['home_tov_pct'] = games_df.apply(
            lambda r: calc.calculate_tov_pct(
                r['home_turnovers'], r['home_fga'], r['home_fta']
            ), axis=1
        )
        result['home_orb_pct'] = games_df.apply(
            lambda r: calc.calculate_orb_pct(
                r['home_orb'], r['away_drb']
            ), axis=1
        )
        result['home_ft_rate'] = games_df.apply(
            lambda r: calc.calculate_ft_rate(
                r['home_ftm'], r['home_fga']
            ), axis=1
        )
        
        # Away team factors
        result['away_efg_pct'] = games_df.apply(
            lambda r: calc.calculate_efg_pct(
                r['away_fgm'], r['away_fg3m'], r['away_fga']
            ), axis=1
        )
        result['away_tov_pct'] = games_df.apply(
            lambda r: calc.calculate_tov_pct(
                r['away_turnovers'], r['away_fga'], r['away_fta']
            ), axis=1
        )
        result['away_orb_pct'] = games_df.apply(
            lambda r: calc.calculate_orb_pct(
                r['away_orb'], r['home_drb']
            ), axis=1
        )
        result['away_ft_rate'] = games_df.apply(
            lambda r: calc.calculate_ft_rate(
                r['away_ftm'], r['away_fga']
            ), axis=1
        )
        
        # Differentials (home - away)
        result['delta_efg'] = result['home_efg_pct'] - result['away_efg_pct']
        result['delta_tov'] = result['home_tov_pct'] - result['away_tov_pct']
        result['delta_orb'] = result['home_orb_pct'] - result['away_orb_pct']
        result['delta_ftr'] = result['home_ft_rate'] - result['away_ft_rate']
        
        # Outcome
        result['home_win'] = (result['home_score'] > result['away_score']).astype(int)
        result['margin'] = result['home_score'] - result['away_score']
        
        return result
    
    def calculate_correlation_with_wins(
        self,
        games_df: pd.DataFrame
    ) -> List[CorrelationResult]:
        """
        Calculate correlation of each Four Factor with winning.
        
        Tests both Pearson (linear) and Spearman (rank) correlations.
        """
        # Ensure Four Factors are calculated
        if 'delta_efg' not in games_df.columns:
            games_df = self.compute_four_factors_for_games(games_df)
        
        results = []
        metrics = [
            ('delta_efg', 'eFG% Differential'),
            ('delta_tov', 'TOV% Differential'),
            ('delta_orb', 'ORB% Differential'),
            ('delta_ftr', 'FT Rate Differential'),
        ]
        
        for col, name in metrics:
            if col not in games_df.columns:
                continue
                
            x = games_df[col].values
            y = games_df['home_win'].values
            
            # Remove NaN pairs
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 10:
                continue
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
            
            # Spearman rank correlation (more robust)
            spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
            
            results.append(CorrelationResult(
                metric_name=name,
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                spearman_rho=spearman_rho,
                spearman_p=spearman_p,
                is_significant=min(pearson_p, spearman_p) < 0.05
            ))
        
        # Sort by absolute correlation strength
        results.sort(key=lambda x: abs(x.pearson_r), reverse=True)
        
        return results
    
    def calculate_spread_correlation(
        self,
        games_df: pd.DataFrame,
        spread_column: str = 'spread'
    ) -> List[CorrelationResult]:
        """
        Calculate correlation of Four Factors with spread coverage.
        
        Spread coverage = actual margin - expected margin (spread).
        Positive = team covered, negative = team failed to cover.
        """
        if spread_column not in games_df.columns:
            warnings.warn(f"Spread column '{spread_column}' not found")
            return []
        
        df = games_df.copy()
        df['margin_vs_spread'] = df['margin'] - df[spread_column]
        df['covered'] = (df['margin_vs_spread'] > 0).astype(int)
        
        results = []
        metrics = [
            ('delta_efg', 'eFG% vs Spread'),
            ('delta_tov', 'TOV% vs Spread'),
            ('delta_orb', 'ORB% vs Spread'),
            ('delta_ftr', 'FTR vs Spread'),
        ]
        
        for col, name in metrics:
            if col not in df.columns:
                continue
                
            x = df[col].values
            y = df['margin_vs_spread'].values
            
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 10:
                continue
            
            pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
            spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
            
            results.append(CorrelationResult(
                metric_name=name,
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                spearman_rho=spearman_rho,
                spearman_p=spearman_p,
                is_significant=min(pearson_p, spearman_p) < 0.05
            ))
        
        results.sort(key=lambda x: abs(x.pearson_r), reverse=True)
        return results
    
    def fit_logistic_model(
        self,
        games_df: pd.DataFrame,
        use_cross_validation: bool = True,
        n_splits: int = 5
    ) -> Dict:
        """
        Fit logistic regression model using Four Factors.
        
        Returns:
            Dictionary with model, coefficients, and performance metrics
        """
        if 'delta_efg' not in games_df.columns:
            games_df = self.compute_four_factors_for_games(games_df)
        
        feature_cols = ['delta_efg', 'delta_tov', 'delta_orb', 'delta_ftr']
        
        # Prepare data
        X = games_df[feature_cols].values
        y = games_df['home_win'].values
        
        # Remove rows with NaN
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000
        )
        model.fit(X_scaled, y)
        
        # Cross-validation
        if use_cross_validation:
            cv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            brier_scores = []
            for train_idx, val_idx in cv.split(X_scaled):
                model_cv = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
                model_cv.fit(X_scaled[train_idx], y[train_idx])
                probs = model_cv.predict_proba(X_scaled[val_idx])[:, 1]
                brier_scores.append(brier_score_loss(y[val_idx], probs))
            cv_brier = np.mean(brier_scores)
        else:
            cv_scores = None
            cv_brier = None
        
        # Feature importance (coefficients)
        coefficients = dict(zip(feature_cols, model.coef_[0]))
        
        # In-sample predictions
        probs = model.predict_proba(X_scaled)[:, 1]
        brier = brier_score_loss(y, probs)
        
        return {
            'model': model,
            'scaler': self.scaler,
            'coefficients': coefficients,
            'intercept': model.intercept_[0],
            'brier_score': brier,
            'cv_accuracy': np.mean(cv_scores) if cv_scores is not None else None,
            'cv_brier': cv_brier,
            'n_samples': len(y),
            'feature_importance': sorted(
                coefficients.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
        }
    
    def fit_random_forest_model(
        self,
        games_df: pd.DataFrame,
        n_estimators: int = 100,
        max_depth: int = 5
    ) -> Dict:
        """
        Fit Random Forest for feature importance comparison.
        
        RF feature importance provides a different perspective than
        logistic regression coefficients.
        """
        if 'delta_efg' not in games_df.columns:
            games_df = self.compute_four_factors_for_games(games_df)
        
        feature_cols = ['delta_efg', 'delta_tov', 'delta_orb', 'delta_ftr']
        
        X = games_df[feature_cols].values
        y = games_df['home_win'].values
        
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        y = y[mask]
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        importances = dict(zip(feature_cols, model.feature_importances_))
        
        return {
            'model': model,
            'feature_importance': sorted(
                importances.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            'n_samples': len(y)
        }


def generate_research_output(
    games_df: pd.DataFrame,
    spread_column: Optional[str] = None
) -> Dict:
    """
    Generate comprehensive Four Factors research output.
    
    Returns structured output matching the research schema.
    """
    analyzer = FourFactorsAnalyzer()
    
    # Data quality
    quality = analyzer.assess_data_quality(games_df)
    
    # Compute factors
    df = analyzer.compute_four_factors_for_games(games_df)
    
    # Correlations
    win_correlations = analyzer.calculate_correlation_with_wins(df)
    
    # Spread correlations (if available)
    if spread_column and spread_column in df.columns:
        spread_correlations = analyzer.calculate_spread_correlation(df, spread_column)
    else:
        spread_correlations = []
    
    # Models
    logreg_results = analyzer.fit_logistic_model(df)
    rf_results = analyzer.fit_random_forest_model(df)
    
    # Compile output
    return {
        "key_metrics": [
            {
                "name": corr.metric_name,
                "correlation": round(corr.pearson_r, 3),
                "p_value": round(corr.pearson_p, 4),
                "significant": corr.is_significant
            }
            for corr in win_correlations
        ],
        "trends": [
            f"eFG% differential has r={win_correlations[0].pearson_r:.3f} correlation with wins"
            if win_correlations else "Insufficient data",
            f"Logistic model Brier score: {logreg_results['brier_score']:.3f}",
            f"Cross-validation accuracy: {logreg_results['cv_accuracy']:.1%}"
            if logreg_results['cv_accuracy'] else "CV not performed"
        ],
        "insights": [
            f"Top predictor: {logreg_results['feature_importance'][0][0]} "
            f"(coef={logreg_results['feature_importance'][0][1]:.3f})",
            f"Sample size: {quality.total_games} games, "
            f"{quality.completeness_pct:.0f}% complete",
            f"Data range: {quality.date_range[0]} to {quality.date_range[1]}"
        ],
        "model_coefficients": logreg_results['coefficients'],
        "data_quality": {
            "total_games": quality.total_games,
            "complete_pct": quality.completeness_pct,
            "warnings": quality.warnings
        }
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Four Factors Analysis Module")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_games = 100
    
    sample_df = pd.DataFrame({
        'game_date': pd.date_range('2024-01-01', periods=n_games, freq='2D'),
        'home_team': np.random.choice(['MEL', 'SYD', 'PER'], n_games),
        'away_team': np.random.choice(['BRI', 'ADL', 'NZB'], n_games),
        'home_fgm': np.random.randint(30, 45, n_games),
        'home_fg3m': np.random.randint(8, 18, n_games),
        'home_fga': np.random.randint(70, 90, n_games),
        'home_turnovers': np.random.randint(8, 18, n_games),
        'home_fta': np.random.randint(15, 30, n_games),
        'home_ftm': np.random.randint(12, 25, n_games),
        'home_orb': np.random.randint(8, 15, n_games),
        'home_drb': np.random.randint(22, 32, n_games),
        'away_fgm': np.random.randint(30, 45, n_games),
        'away_fg3m': np.random.randint(8, 18, n_games),
        'away_fga': np.random.randint(70, 90, n_games),
        'away_turnovers': np.random.randint(8, 18, n_games),
        'away_fta': np.random.randint(15, 30, n_games),
        'away_ftm': np.random.randint(12, 25, n_games),
        'away_orb': np.random.randint(8, 15, n_games),
        'away_drb': np.random.randint(22, 32, n_games),
        'home_score': np.random.randint(85, 115, n_games),
        'away_score': np.random.randint(85, 115, n_games),
    })
    
    # Run analysis
    output = generate_research_output(sample_df)
    
    print("\n📊 Key Metrics:")
    for m in output['key_metrics']:
        sig = "✓" if m['significant'] else "✗"
        print(f"  {m['name']}: r={m['correlation']:+.3f} (p={m['p_value']:.3f}) {sig}")
    
    print("\n📈 Trends:")
    for t in output['trends']:
        print(f"  • {t}")
    
    print("\n💡 Insights:")
    for i in output['insights']:
        print(f"  • {i}")

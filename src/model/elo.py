"""
ELO Rating System for NBL/WNBL Teams.

Implements a statistically rigorous ELO rating system for predicting
game outcomes. ELO ratings dynamically adjust based on game results.

Key Parameters:
    - K-factor: 20 (standard for basketball)
    - Home advantage: +100 ELO points
    - Initial rating: 1500

References:
    - Elo, A. (1978). "The Rating of Chessplayers, Past and Present"
    - Silver, N. (FiveThirtyEight NBA ELO methodology)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single game for ELO update."""
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    game_date: datetime
    is_playoff: bool = False
    is_overtime: bool = False


@dataclass
class ELOPrediction:
    """Prediction from ELO system."""
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    home_win_prob: float
    elo_diff: float
    expected_margin: float
    
    @property
    def away_win_prob(self) -> float:
        return 1.0 - self.home_win_prob
    
    def __str__(self) -> str:
        return (
            f"{self.home_team} ({self.home_elo:.0f}) vs "
            f"{self.away_team} ({self.away_elo:.0f})\n"
            f"Home Win Prob: {self.home_win_prob:.1%} | "
            f"Expected Margin: {self.expected_margin:+.1f}"
        )


class ELORatingSystem:
    """
    ELO-based prediction model for NBL/WNBL.
    
    ELO Formula:
        Expected Score: E = 1 / (1 + 10^((R_opponent - R_self) / 400))
        Rating Update: R_new = R_old + K * (S - E)
    
    Where:
        - K = 20 (K-factor, how much ratings change per game)
        - S = 1 for win, 0 for loss (with margin adjustment)
        - E = expected score based on rating difference
    
    NBL-Specific Adjustments:
        - Home court advantage: +100 ELO
        - Playoff K-factor: 25 (higher stakes)
        - Season regression: 1/3 toward mean
    
    Example:
        >>> elo = ELORatingSystem()
        >>> elo.initialize_ratings(['MEL', 'SYD', 'PER'])
        >>> pred = elo.predict_game('MEL', 'SYD')
        >>> print(f"Melbourne win prob: {pred.home_win_prob:.1%}")
    """
    
    # Default parameters
    DEFAULT_K = 20.0
    PLAYOFF_K = 25.0
    HOME_ADVANTAGE = 100.0
    INITIAL_RATING = 1500.0
    MARGIN_MULTIPLIER = 0.03  # Points per margin point
    SEASON_REGRESSION = 0.33  # Regress 1/3 toward mean
    
    def __init__(
        self,
        k_factor: float = DEFAULT_K,
        home_advantage: float = HOME_ADVANTAGE,
        initial_rating: float = INITIAL_RATING,
        use_margin: bool = True
    ):
        """
        Initialize ELO rating system.
        
        Args:
            k_factor: How much ratings change per game
            home_advantage: ELO boost for home team
            initial_rating: Starting ELO for new teams
            use_margin: Whether to adjust for margin of victory
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.use_margin = use_margin
        
        self._ratings: Dict[str, float] = {}
        self._history: List[Dict] = []
        self._games_processed: int = 0
    
    def initialize_ratings(
        self,
        teams: List[str],
        initial_ratings: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize ratings for a list of teams.
        
        Args:
            teams: List of team codes
            initial_ratings: Optional custom starting ratings
        """
        for team in teams:
            if initial_ratings and team in initial_ratings:
                self._ratings[team] = initial_ratings[team]
            else:
                self._ratings[team] = self.initial_rating
        
        logger.info(f"Initialized ELO ratings for {len(teams)} teams")
    
    def get_rating(self, team: str) -> float:
        """Get current ELO rating for a team."""
        return self._ratings.get(team, self.initial_rating)
    
    def set_rating(self, team: str, rating: float) -> None:
        """Set ELO rating for a team."""
        self._ratings[team] = rating
    
    def _expected_score(
        self,
        rating_a: float,
        rating_b: float,
        home_advantage: float = 0
    ) -> float:
        """
        Calculate expected score using ELO formula.
        
        Args:
            rating_a: Rating of team A
            rating_b: Rating of team B
            home_advantage: ELO boost for team A (if home)
        
        Returns:
            Expected score (0 to 1)
        """
        adjusted_a = rating_a + home_advantage
        exponent = (rating_b - adjusted_a) / 400.0
        return 1.0 / (1.0 + 10 ** exponent)
    
    def _margin_multiplier(
        self,
        margin: int,
        winner_rating: float,
        loser_rating: float
    ) -> float:
        """
        Calculate margin of victory multiplier.
        
        Large wins against weak opponents count less than
        close wins against strong opponents.
        
        From FiveThirtyEight methodology:
        MOV = ((margin + 3)^0.8) / (7.5 + 0.006 * rating_diff)
        """
        abs_margin = abs(margin)
        rating_diff = winner_rating - loser_rating
        
        numerator = (abs_margin + 3) ** 0.8
        denominator = 7.5 + 0.006 * rating_diff
        
        return numerator / denominator
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False
    ) -> ELOPrediction:
        """
        Predict game outcome using current ratings.
        
        Args:
            home_team: Home team code
            away_team: Away team code
            neutral: If True, no home court advantage
        
        Returns:
            ELOPrediction with win probabilities
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        home_adv = 0 if neutral else self.home_advantage
        
        home_win_prob = self._expected_score(
            home_rating, away_rating, home_adv
        )
        
        # ELO difference with home advantage
        elo_diff = (home_rating + home_adv) - away_rating
        
        # Expected margin (approximately 25 ELO points = 1 point)
        expected_margin = elo_diff / 25.0
        
        return ELOPrediction(
            home_team=home_team,
            away_team=away_team,
            home_elo=home_rating,
            away_elo=away_rating,
            home_win_prob=home_win_prob,
            elo_diff=elo_diff,
            expected_margin=expected_margin
        )
    
    def update_ratings(
        self,
        result: GameResult,
        use_playoff_k: bool = False
    ) -> Tuple[float, float]:
        """
        Update ratings after a game.
        
        Args:
            result: GameResult with teams and scores
            use_playoff_k: Use higher K-factor for playoffs
        
        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        home = result.home_team
        away = result.away_team
        
        # Get current ratings (or initialize)
        home_rating = self.get_rating(home)
        away_rating = self.get_rating(away)
        
        # Determine winner
        home_won = result.home_score > result.away_score
        margin = result.home_score - result.away_score
        
        # Expected scores
        expected_home = self._expected_score(
            home_rating, away_rating, self.home_advantage
        )
        expected_away = 1 - expected_home
        
        # Actual scores (1 for win, 0 for loss)
        actual_home = 1.0 if home_won else 0.0
        actual_away = 1.0 - actual_home
        
        # K-factor
        k = self.PLAYOFF_K if (result.is_playoff or use_playoff_k) else self.k_factor
        
        # Margin multiplier
        if self.use_margin:
            winner_rating = home_rating if home_won else away_rating
            loser_rating = away_rating if home_won else home_rating
            k *= self._margin_multiplier(margin, winner_rating, loser_rating)
        
        # Update ratings
        delta_home = k * (actual_home - expected_home)
        delta_away = k * (actual_away - expected_away)
        
        new_home = home_rating + delta_home
        new_away = away_rating + delta_away
        
        self._ratings[home] = new_home
        self._ratings[away] = new_away
        
        # Record history
        self._history.append({
            'game_id': result.game_id,
            'date': result.game_date,
            'home': home,
            'away': away,
            'home_score': result.home_score,
            'away_score': result.away_score,
            'home_elo_before': home_rating,
            'away_elo_before': away_rating,
            'home_elo_after': new_home,
            'away_elo_after': new_away,
            'home_delta': delta_home,
            'away_delta': delta_away,
        })
        
        self._games_processed += 1
        
        return new_home, new_away
    
    def process_season(
        self,
        games: pd.DataFrame,
        home_col: str = 'home_team',
        away_col: str = 'away_team',
        home_score_col: str = 'home_score',
        away_score_col: str = 'away_score',
        date_col: str = 'game_date',
        game_id_col: str = 'game_id'
    ) -> pd.DataFrame:
        """
        Process a full season of games to update ratings.
        
        Args:
            games: DataFrame with game results (sorted by date)
            *_col: Column names for each field
        
        Returns:
            DataFrame with ELO history
        """
        # Ensure sorted by date
        games = games.sort_values(date_col)
        
        for _, game in games.iterrows():
            result = GameResult(
                game_id=str(game.get(game_id_col, '')),
                home_team=game[home_col],
                away_team=game[away_col],
                home_score=int(game[home_score_col]),
                away_score=int(game[away_score_col]),
                game_date=pd.to_datetime(game[date_col])
            )
            self.update_ratings(result)
        
        logger.info(f"Processed {len(games)} games")
        return self.get_history_dataframe()
    
    def regress_to_mean(self, factor: Optional[float] = None) -> None:
        """
        Regress all ratings toward the mean.
        
        Called at the start of a new season to account for
        roster changes and regression to the mean.
        
        Args:
            factor: Regression factor (default: 1/3)
        """
        factor = factor or self.SEASON_REGRESSION
        mean_rating = self.initial_rating
        
        for team in self._ratings:
            current = self._ratings[team]
            self._ratings[team] = current + factor * (mean_rating - current)
        
        logger.info(f"Regressed {len(self._ratings)} teams by factor {factor}")
    
    def get_rankings(self) -> pd.DataFrame:
        """Get current team rankings."""
        rankings = pd.DataFrame([
            {'team': team, 'elo': rating}
            for team, rating in self._ratings.items()
        ])
        rankings = rankings.sort_values('elo', ascending=False)
        rankings['rank'] = range(1, len(rankings) + 1)
        rankings = rankings[['rank', 'team', 'elo']]
        return rankings.reset_index(drop=True)
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Get full rating history as DataFrame."""
        return pd.DataFrame(self._history)
    
    def save(self, filepath: str) -> None:
        """Save ratings to JSON file."""
        data = {
            'ratings': self._ratings,
            'config': {
                'k_factor': self.k_factor,
                'home_advantage': self.home_advantage,
                'initial_rating': self.initial_rating,
                'use_margin': self.use_margin,
            },
            'games_processed': self._games_processed,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved ELO ratings to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ELORatingSystem":
        """Load ratings from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = data.get('config', {})
        elo = cls(
            k_factor=config.get('k_factor', cls.DEFAULT_K),
            home_advantage=config.get('home_advantage', cls.HOME_ADVANTAGE),
            initial_rating=config.get('initial_rating', cls.INITIAL_RATING),
            use_margin=config.get('use_margin', True),
        )
        elo._ratings = data.get('ratings', {})
        elo._games_processed = data.get('games_processed', 0)
        
        logger.info(f"Loaded ELO ratings from {filepath}")
        return elo


# NBL Team Codes for initialization
NBL_TEAMS = [
    'MEL',  # Melbourne United
    'SYD',  # Sydney Kings
    'PER',  # Perth Wildcats
    'BRI',  # Brisbane Bullets
    'ADL',  # Adelaide 36ers
    'NZB',  # New Zealand Breakers
    'ILL',  # Illawarra Hawks
    'CAI',  # Cairns Taipans
    'TAS',  # Tasmania JackJumpers
    'SEM',  # South East Melbourne Phoenix
]


def create_nbl_elo_system() -> ELORatingSystem:
    """Create ELO system initialized with NBL teams."""
    elo = ELORatingSystem()
    elo.initialize_ratings(NBL_TEAMS)
    return elo


# Predictor interface for BacktestEngine compatibility
class ELOPredictor:
    """
    Wrapper to make ELORatingSystem compatible with BacktestEngine.
    
    Implements fit/predict_proba interface expected by backtest framework.
    """
    
    def __init__(
        self,
        k_factor: float = ELORatingSystem.DEFAULT_K,
        home_advantage: float = ELORatingSystem.HOME_ADVANTAGE
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self._elo = ELORatingSystem(k_factor, home_advantage)
        self._is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        home_col: str = 'home_team',
        away_col: str = 'away_team',
        home_score_col: str = 'home_score',
        away_score_col: str = 'away_score'
    ) -> "ELOPredictor":
        """
        Fit ELO ratings on historical games.
        
        Note: X must contain team and score columns alongside features.
        """
        # Initialize with unique teams
        all_teams = set(X[home_col].unique()) | set(X[away_col].unique())
        self._elo.initialize_ratings(list(all_teams))
        
        # Process games
        games_df = X.copy()
        games_df['home_win'] = y
        games_df['home_score'] = games_df.get(home_score_col, 0)
        games_df['away_score'] = games_df.get(away_score_col, 0)
        
        # If scores not available, synthesize from wins
        if games_df['home_score'].sum() == 0:
            games_df['home_score'] = np.where(y == 1, 100, 90)
            games_df['away_score'] = np.where(y == 1, 90, 100)
        
        self._elo.process_season(
            games_df,
            home_col=home_col,
            away_col=away_col,
            date_col='game_date' if 'game_date' in games_df.columns else X.columns[0]
        )
        
        self._is_fitted = True
        return self
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        home_col: str = 'home_team',
        away_col: str = 'away_team'
    ) -> np.ndarray:
        """Predict home win probabilities."""
        probs = []
        
        for _, row in X.iterrows():
            pred = self._elo.predict_game(row[home_col], row[away_col])
            probs.append(pred.home_win_prob)
        
        return np.array(probs)
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

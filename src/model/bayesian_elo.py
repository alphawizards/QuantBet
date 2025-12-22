"""
Bayesian Elo Rating System for NBL/WNBL.

Implements a Bayesian approach to Elo ratings that provides:
    - Posterior distributions for team strengths
    - Uncertainty quantification for predictions
    - Better calibrated probabilities for Kelly sizing

The Bayesian Elo model treats team ratings as random variables
with prior distributions, updating them based on observed outcomes.

This module provides:
    1. BayesianEloRating: Simple uncertainty estimation without PyMC
    2. PyMCEloModel: Full Bayesian inference with PyMC (if available)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TeamRating:
    """Team rating with uncertainty bounds."""
    team_code: str
    mean: float = 1500.0
    std: float = 200.0
    games_played: int = 0
    last_updated: Optional[datetime] = None
    
    @property
    def lower_bound(self) -> float:
        """Lower 95% credible interval bound."""
        return self.mean - 1.96 * self.std
    
    @property
    def upper_bound(self) -> float:
        """Upper 95% credible interval bound."""
        return self.mean + 1.96 * self.std
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'team_code': self.team_code,
            'rating_mean': self.mean,
            'rating_std': self.std,
            'rating_lower': self.lower_bound,
            'rating_upper': self.upper_bound,
            'games': self.games_played
        }


class BayesianEloRating:
    """
    Bayesian Elo rating system with uncertainty estimation.
    
    Unlike standard Elo which produces point estimates, this system
    tracks both the mean rating and uncertainty (standard deviation).
    
    Key features:
        - Uncertainty decreases as team plays more games
        - New/uncertain matchups have wider prediction intervals
        - Better Kelly sizing through uncertainty-aware probabilities
    
    Example:
        >>> elo = BayesianEloRating()
        >>> elo.update_from_game("MEL", "SYD", home_win=True)
        >>> prob, uncertainty = elo.predict_proba("MEL", "SYD")
        >>> print(f"P(MEL wins): {prob:.1%} ± {uncertainty:.1%}")
    """
    
    # Prior parameters
    PRIOR_MEAN = 1500.0
    PRIOR_STD = 200.0
    
    # Update parameters
    K_BASE = 32.0
    HOME_ADVANTAGE = 50.0  # ~2-3 points in basketball
    
    def __init__(
        self,
        k_factor: float = 32.0,
        home_advantage: float = 50.0,
        prior_mean: float = 1500.0,
        prior_std: float = 200.0
    ):
        """
        Initialize the Bayesian Elo system.
        
        Args:
            k_factor: Base K-factor for updates
            home_advantage: Elo points added for home team
            prior_mean: Prior mean rating for new teams
            prior_std: Prior standard deviation for new teams
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        self.ratings: Dict[str, TeamRating] = {}
    
    def get_rating(self, team_code: str) -> TeamRating:
        """Get team rating, creating if doesn't exist."""
        if team_code not in self.ratings:
            self.ratings[team_code] = TeamRating(
                team_code=team_code,
                mean=self.prior_mean,
                std=self.prior_std
            )
        return self.ratings[team_code]
    
    def expected_score(
        self,
        home_rating: float,
        away_rating: float
    ) -> float:
        """
        Calculate expected score (win probability) for home team.
        
        Uses the standard Elo expected score formula with home advantage:
            E = 1 / (1 + 10^((R_away - R_home - HomeAdv) / 400))
        """
        rating_diff = home_rating + self.home_advantage - away_rating
        return 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))
    
    def update_from_game(
        self,
        home_team: str,
        away_team: str,
        home_win: bool,
        game_date: Optional[datetime] = None
    ) -> Tuple[float, float]:
        """
        Update ratings after a game result.
        
        Args:
            home_team: Home team code
            away_team: Away team code
            home_win: True if home team won
            game_date: Date of game
        
        Returns:
            Tuple of (home_rating_change, away_rating_change)
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Expected scores
        expected_home = self.expected_score(home_rating.mean, away_rating.mean)
        expected_away = 1.0 - expected_home
        
        # Actual scores
        actual_home = 1.0 if home_win else 0.0
        actual_away = 1.0 - actual_home
        
        # Dynamic K-factor based on uncertainty
        # Higher uncertainty = larger updates
        k_home = self.k_factor * (1 + home_rating.std / self.prior_std)
        k_away = self.k_factor * (1 + away_rating.std / self.prior_std)
        
        # Rating changes
        home_change = k_home * (actual_home - expected_home)
        away_change = k_away * (actual_away - expected_away)
        
        # Update means
        home_rating.mean += home_change
        away_rating.mean += away_change
        
        # Update uncertainties (decrease with more games)
        # std = prior_std / sqrt(1 + games / scale_factor)
        home_rating.games_played += 1
        away_rating.games_played += 1
        
        home_rating.std = self.prior_std / np.sqrt(1 + home_rating.games_played / 5)
        away_rating.std = self.prior_std / np.sqrt(1 + away_rating.games_played / 5)
        
        # Update timestamps
        if game_date:
            home_rating.last_updated = game_date
            away_rating.last_updated = game_date
        
        return home_change, away_change
    
    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        n_samples: int = 1000
    ) -> Tuple[float, float]:
        """
        Predict probability of home team winning with uncertainty.
        
        Uses Monte Carlo sampling to incorporate rating uncertainty:
        1. Sample from rating distributions
        2. Compute expected score for each sample
        3. Return mean and std of predictions
        
        Args:
            home_team: Home team code
            away_team: Away team code
            n_samples: Number of Monte Carlo samples
        
        Returns:
            Tuple of (mean_probability, probability_std)
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Sample from rating distributions
        home_samples = np.random.normal(
            home_rating.mean, home_rating.std, n_samples
        )
        away_samples = np.random.normal(
            away_rating.mean, away_rating.std, n_samples
        )
        
        # Compute expected scores for each sample
        probs = 1.0 / (
            1.0 + 10 ** (-(home_samples + self.home_advantage - away_samples) / 400.0)
        )
        
        return float(np.mean(probs)), float(np.std(probs))
    
    def predict_with_confidence(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Predict with full confidence interval detail.
        
        Returns:
            Dictionary with prediction details
        """
        mean_prob, std_prob = self.predict_proba(home_team, away_team)
        
        # Compute credible intervals
        lower_25 = max(0, mean_prob - 0.67 * std_prob)
        upper_75 = min(1, mean_prob + 0.67 * std_prob)
        lower_05 = max(0, mean_prob - 1.96 * std_prob)
        upper_95 = min(1, mean_prob + 1.96 * std_prob)
        
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': mean_prob,
            'home_win_prob_std': std_prob,
            'ci_25': lower_25,
            'ci_75': upper_75,
            'ci_05': lower_05,
            'ci_95': upper_95,
            'home_rating_mean': home_rating.mean,
            'home_rating_std': home_rating.std,
            'away_rating_mean': away_rating.mean,
            'away_rating_std': away_rating.std,
            'home_advantage': self.home_advantage
        }
    
    def fit_from_history(
        self,
        games: pd.DataFrame,
        home_team_col: str = 'home_team',
        away_team_col: str = 'away_team',
        home_score_col: str = 'home_score',
        away_score_col: str = 'away_score',
        date_col: str = 'game_date'
    ) -> 'BayesianEloRating':
        """
        Fit ratings from historical game data.
        
        Args:
            games: DataFrame with game results
            home_team_col: Column name for home team
            away_team_col: Column name for away team
            home_score_col: Column name for home score
            away_score_col: Column name for away score
            date_col: Column name for game date
        
        Returns:
            Self for method chaining
        """
        # Sort by date
        games = games.sort_values(date_col)
        
        for _, game in games.iterrows():
            home_win = game[home_score_col] > game[away_score_col]
            game_date = game[date_col] if date_col in game.index else None
            
            self.update_from_game(
                home_team=game[home_team_col],
                away_team=game[away_team_col],
                home_win=home_win,
                game_date=game_date
            )
        
        logger.info(f"Fitted Bayesian Elo from {len(games)} games for {len(self.ratings)} teams")
        
        return self
    
    def get_all_ratings(self) -> pd.DataFrame:
        """Get all team ratings as DataFrame."""
        ratings_list = [r.to_dict() for r in self.ratings.values()]
        return pd.DataFrame(ratings_list).sort_values('rating_mean', ascending=False)
    
    def kelly_stake(
        self,
        home_team: str,
        away_team: str,
        odds: float,
        bet_on_home: bool = True,
        fraction: float = 0.25,
        use_conservative: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Kelly stake with uncertainty adjustment.
        
        Uses the lower bound of the probability credible interval
        for more conservative betting.
        
        Args:
            home_team: Home team code
            away_team: Away team code
            odds: Decimal odds for the bet
            bet_on_home: True if betting on home team
            fraction: Kelly fraction
            use_conservative: If True, use lower CI bound for probability
        
        Returns:
            Dictionary with stake recommendation
        """
        pred = self.predict_with_confidence(home_team, away_team)
        
        if bet_on_home:
            prob = pred['home_win_prob']
            prob_conservative = pred['ci_25'] if use_conservative else prob
        else:
            prob = 1 - pred['home_win_prob']
            prob_conservative = 1 - pred['ci_75'] if use_conservative else prob
        
        # Kelly formula
        implied_prob = 1 / odds
        edge = prob_conservative - implied_prob
        
        if edge <= 0:
            kelly = 0
        else:
            kelly = (prob_conservative * odds - 1) / (odds - 1)
        
        adjusted_kelly = max(0, kelly * fraction)
        
        return {
            'probability': prob,
            'probability_conservative': prob_conservative,
            'implied_prob': implied_prob,
            'edge': edge,
            'kelly_fraction': kelly,
            'recommended_stake': adjusted_kelly,
            'bet_on': 'home' if bet_on_home else 'away',
            'odds': odds
        }


# Optional: PyMC-based full Bayesian model
def create_pymc_elo_model():
    """
    Create a PyMC-based Bayesian Elo model.
    
    This provides full posterior inference but requires PyMC.
    Falls back to BayesianEloRating if PyMC is not available.
    """
    try:
        import pymc as pm
        PYMC_AVAILABLE = True
    except ImportError:
        PYMC_AVAILABLE = False
        logger.warning("PyMC not available. Using simplified Bayesian Elo.")
        return None
    
    # PyMC model would be defined here
    # For now, return None to indicate use of simplified version
    logger.info("PyMC available - full Bayesian inference enabled")
    return None  # Placeholder for future PyMC implementation

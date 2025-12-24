"""
Feature Store for NBL/WNBL Betting Model.

Centralized feature management with computation, caching, versioning,
and validation. Follows MLOps best practices for production ML systems.

Features Categories:
    - ELO: Team ratings and differentials
    - Form: Recent performance metrics
    - H2H: Head-to-head historical stats
    - Schedule: Rest days, travel, back-to-backs
    - Venue: Home court advantage, altitude

Implements SKILL.md patterns:
    - Production-First Design
    - Observability: Feature validation and logging
    - Caching: Avoid redundant computation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a computed feature."""
    name: str
    category: str
    description: str
    dtype: str
    missing_rate: float
    min_value: float
    max_value: float
    mean_value: float
    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class FeatureVector:
    """Complete feature vector for a single game."""
    game_id: str
    features: Dict[str, float]
    computed_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        return self.features
    
    def to_series(self) -> pd.Series:
        return pd.Series(self.features)


@dataclass
class ValidationReport:
    """Result of feature validation."""
    passed: bool
    errors: List[str]
    warnings: List[str]
    feature_count: int
    missing_count: int
    
    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"""
Feature Validation: {status}
  Features: {self.feature_count}
  Missing: {self.missing_count}
  Errors: {len(self.errors)}
  Warnings: {len(self.warnings)}
"""


class FeatureStore:
    """
    Centralized feature computation and storage.
    
    Manages all features for the NBL betting model, including:
    - On-demand feature computation
    - Caching to avoid redundant calculations
    - Feature versioning for reproducibility
    - Validation to catch data quality issues
    
    Example:
        >>> store = FeatureStore()
        >>> features = store.compute_features(game_data, historical_data)
        >>> store.validate_features(features)
    """
    
    # Feature definitions with expected ranges
    FEATURE_SPECS = {
        # ELO features
        'home_elo': {'min': 1000, 'max': 2000, 'category': 'elo'},
        'away_elo': {'min': 1000, 'max': 2000, 'category': 'elo'},
        'elo_diff': {'min': -500, 'max': 500, 'category': 'elo'},
        'elo_win_prob': {'min': 0, 'max': 1, 'category': 'elo'},
        
        # Form features (last N games)
        'home_win_rate_l5': {'min': 0, 'max': 1, 'category': 'form'},
        'away_win_rate_l5': {'min': 0, 'max': 1, 'category': 'form'},
        'home_avg_margin_l5': {'min': -50, 'max': 50, 'category': 'form'},
        'away_avg_margin_l5': {'min': -50, 'max': 50, 'category': 'form'},
        'home_streak': {'min': -10, 'max': 10, 'category': 'form'},
        'away_streak': {'min': -10, 'max': 10, 'category': 'form'},
        
        # Efficiency features
        'home_ortg_l5': {'min': 80, 'max': 140, 'category': 'efficiency'},
        'away_ortg_l5': {'min': 80, 'max': 140, 'category': 'efficiency'},
        'home_drtg_l5': {'min': 80, 'max': 140, 'category': 'efficiency'},
        'away_drtg_l5': {'min': 80, 'max': 140, 'category': 'efficiency'},
        'home_net_rtg_l5': {'min': -30, 'max': 30, 'category': 'efficiency'},
        'away_net_rtg_l5': {'min': -30, 'max': 30, 'category': 'efficiency'},
        
        # Head-to-head
        'h2h_home_wins': {'min': 0, 'max': 20, 'category': 'h2h'},
        'h2h_away_wins': {'min': 0, 'max': 20, 'category': 'h2h'},
        'h2h_home_win_rate': {'min': 0, 'max': 1, 'category': 'h2h'},
        'h2h_avg_margin': {'min': -30, 'max': 30, 'category': 'h2h'},
        
        # Schedule
        'home_days_rest': {'min': 0, 'max': 14, 'category': 'schedule'},
        'away_days_rest': {'min': 0, 'max': 14, 'category': 'schedule'},
        'home_back_to_back': {'min': 0, 'max': 1, 'category': 'schedule'},
        'away_back_to_back': {'min': 0, 'max': 1, 'category': 'schedule'},
        'home_travel_fatigue': {'min': 0, 'max': 1, 'category': 'schedule'},
        'away_travel_fatigue': {'min': 0, 'max': 1, 'category': 'schedule'},
        
        # Market
        'home_implied_prob': {'min': 0, 'max': 1, 'category': 'market'},
        'away_implied_prob': {'min': 0, 'max': 1, 'category': 'market'},
        'market_vig': {'min': 1, 'max': 1.2, 'category': 'market'},
        
        # Advanced metrics (BPM/SOS - calculated from box scores)
        'home_bpm': {'min': -10, 'max': 10, 'category': 'advanced'},
        'away_bpm': {'min': -10, 'max': 10, 'category': 'advanced'},
        'bpm_differential': {'min': -20, 'max': 20, 'category': 'advanced'},
        'home_sos': {'min': -1, 'max': 1, 'category': 'advanced'},
        'away_sos': {'min': -1, 'max': 1, 'category': 'advanced'},
        'home_sos_adj_win_pct': {'min': 0, 'max': 1, 'category': 'advanced'},
        'away_sos_adj_win_pct': {'min': 0, 'max': 1, 'category': 'advanced'},
        'sos_adj_win_pct_diff': {'min': -1, 'max': 1, 'category': 'advanced'},
        'home_expected_wins': {'min': 0, 'max': 30, 'category': 'advanced'},
        'away_expected_wins': {'min': 0, 'max': 30, 'category': 'advanced'},
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        version: str = "1.0"
    ):
        """
        Initialize feature store.
        
        Args:
            cache_dir: Directory for caching computed features
            version: Feature version string
        """
        self.version = version
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: Dict[str, FeatureVector] = {}
        self._metadata: Dict[str, FeatureMetadata] = {}
        
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_features(
        self,
        game: pd.Series,
        historical_data: pd.DataFrame,
        elo_ratings: Optional[Dict[str, float]] = None
    ) -> FeatureVector:
        """
        Compute all features for a single game.
        
        Args:
            game: Series with game info (home_team, away_team, date, etc.)
            historical_data: DataFrame with past game results
            elo_ratings: Optional pre-computed ELO ratings
        
        Returns:
            FeatureVector with all computed features
        """
        game_id = str(game.get('game_id', hash(str(game.values))))
        
        # Check cache
        if game_id in self._cache:
            return self._cache[game_id]
        
        features = {}
        
        home_team = game['home_team']
        away_team = game['away_team']
        game_date = pd.to_datetime(game.get('game_date', datetime.now()))
        
        # Filter to games before this one
        past_games = historical_data[
            pd.to_datetime(historical_data['game_date']) < game_date
        ]
        
        # ELO features
        if elo_ratings:
            features.update(self._compute_elo_features(
                home_team, away_team, elo_ratings
            ))
        else:
            features.update(self._default_elo_features())
        
        # Form features
        features.update(self._compute_form_features(
            home_team, away_team, past_games
        ))
        
        # Schedule features
        features.update(self._compute_schedule_features(
            home_team, away_team, game_date, past_games
        ))
        
        # H2H features
        features.update(self._compute_h2h_features(
            home_team, away_team, past_games
        ))
        
        # Market features (if odds available)
        if 'home_odds' in game:
            features.update(self._compute_market_features(game))
        
        # Create feature vector
        vector = FeatureVector(
            game_id=game_id,
            features=features,
            version=self.version
        )
        
        # Cache result
        self._cache[game_id] = vector
        
        return vector
    
    def compute_batch_features(
        self,
        games: pd.DataFrame,
        historical_data: pd.DataFrame,
        elo_ratings: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute features for multiple games.
        
        Returns DataFrame with one row per game and feature columns.
        """
        feature_rows = []
        
        for _, game in games.iterrows():
            # Get historical data up to this game
            game_date = pd.to_datetime(game.get('game_date', datetime.now()))
            past_data = historical_data[
                pd.to_datetime(historical_data['game_date']) < game_date
            ]
            
            vector = self.compute_features(game, past_data, elo_ratings)
            row = vector.features.copy()
            row['game_id'] = vector.game_id
            feature_rows.append(row)
        
        return pd.DataFrame(feature_rows)
    
    def _compute_elo_features(
        self,
        home_team: str,
        away_team: str,
        ratings: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute ELO-based features."""
        home_elo = ratings.get(home_team, 1500)
        away_elo = ratings.get(away_team, 1500)
        elo_diff = home_elo - away_elo + 100  # Include home advantage
        
        # ELO win probability
        win_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
        
        return {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'elo_win_prob': win_prob,
        }
    
    def _default_elo_features(self) -> Dict[str, float]:
        """Default ELO features when ratings unavailable."""
        return {
            'home_elo': 1500,
            'away_elo': 1500,
            'elo_diff': 100,  # Home advantage only
            'elo_win_prob': 0.55,
        }
    
    def _compute_form_features(
        self,
        home_team: str,
        away_team: str,
        past_games: pd.DataFrame,
        window: int = 5
    ) -> Dict[str, float]:
        """Compute recent form features."""
        features = {}
        
        for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
            # Get team's recent games
            team_games = past_games[
                (past_games['home_team'] == team) | 
                (past_games['away_team'] == team)
            ].tail(window)
            
            if len(team_games) == 0:
                features[f'{prefix}_win_rate_l{window}'] = 0.5
                features[f'{prefix}_avg_margin_l{window}'] = 0.0
                features[f'{prefix}_streak'] = 0
                continue
            
            # Calculate wins and margins
            wins = 0
            margins = []
            
            for _, g in team_games.iterrows():
                is_home = g['home_team'] == team
                home_score = g.get('home_score', 0)
                away_score = g.get('away_score', 0)
                
                if is_home:
                    win = home_score > away_score
                    margin = home_score - away_score
                else:
                    win = away_score > home_score
                    margin = away_score - home_score
                
                wins += int(win)
                margins.append(margin)
            
            features[f'{prefix}_win_rate_l{window}'] = wins / len(team_games)
            features[f'{prefix}_avg_margin_l{window}'] = float(np.mean(margins))
            
            # Streak (positive = winning, negative = losing)
            streak = self._calculate_streak(team, past_games)
            features[f'{prefix}_streak'] = streak
        
        return features
    
    def _calculate_streak(
        self,
        team: str,
        past_games: pd.DataFrame,
        max_lookback: int = 10
    ) -> int:
        """Calculate current win/loss streak."""
        team_games = past_games[
            (past_games['home_team'] == team) | 
            (past_games['away_team'] == team)
        ].tail(max_lookback)
        
        if len(team_games) == 0:
            return 0
        
        streak = 0
        last_result = None
        
        for _, g in team_games.iloc[::-1].iterrows():
            is_home = g['home_team'] == team
            home_score = g.get('home_score', 0)
            away_score = g.get('away_score', 0)
            
            won = (is_home and home_score > away_score) or \
                  (not is_home and away_score > home_score)
            
            if last_result is None:
                last_result = won
                streak = 1 if won else -1
            elif won == last_result:
                streak += 1 if won else -1
            else:
                break
        
        return streak
    
    def _compute_schedule_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        past_games: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute schedule and rest features."""
        features = {}
        
        for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
            # Find last game for team
            team_games = past_games[
                (past_games['home_team'] == team) | 
                (past_games['away_team'] == team)
            ]
            
            if len(team_games) == 0:
                days_rest = 7
            else:
                last_game = pd.to_datetime(team_games['game_date'].max())
                days_rest = (game_date - last_game).days
            
            features[f'{prefix}_days_rest'] = min(days_rest, 14)
            features[f'{prefix}_back_to_back'] = 1.0 if days_rest <= 1 else 0.0
            
            # Travel fatigue (simplified)
            features[f'{prefix}_travel_fatigue'] = 0.0  # Would need venue data
        
        return features
    
    def _compute_h2h_features(
        self,
        home_team: str,
        away_team: str,
        past_games: pd.DataFrame,
        max_games: int = 10
    ) -> Dict[str, float]:
        """Compute head-to-head features."""
        # Find matchups between these teams
        h2h_games = past_games[
            ((past_games['home_team'] == home_team) & 
             (past_games['away_team'] == away_team)) |
            ((past_games['home_team'] == away_team) & 
             (past_games['away_team'] == home_team))
        ].tail(max_games)
        
        if len(h2h_games) == 0:
            return {
                'h2h_home_wins': 0,
                'h2h_away_wins': 0,
                'h2h_home_win_rate': 0.5,
                'h2h_avg_margin': 0.0,
            }
        
        home_wins = 0
        away_wins = 0
        margins = []
        
        for _, g in h2h_games.iterrows():
            home_score = g.get('home_score', 0)
            away_score = g.get('away_score', 0)
            
            if g['home_team'] == home_team:
                # Same matchup orientation
                if home_score > away_score:
                    home_wins += 1
                else:
                    away_wins += 1
                margins.append(home_score - away_score)
            else:
                # Reversed matchup
                if away_score > home_score:
                    home_wins += 1
                else:
                    away_wins += 1
                margins.append(away_score - home_score)
        
        total = home_wins + away_wins
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_home_win_rate': home_wins / total if total > 0 else 0.5,
            'h2h_avg_margin': float(np.mean(margins)) if margins else 0.0,
        }
    
    def _compute_market_features(
        self,
        game: pd.Series
    ) -> Dict[str, float]:
        """Compute market-based features from odds."""
        home_odds = game.get('home_odds', 2.0)
        away_odds = game.get('away_odds', 2.0)
        
        # Implied probabilities
        home_prob = 1.0 / home_odds if home_odds > 1 else 0.5
        away_prob = 1.0 / away_odds if away_odds > 1 else 0.5
        
        # Market vig (overround)
        vig = home_prob + away_prob
        
        return {
            'home_implied_prob': min(home_prob / vig, 0.99),  # Remove vig
            'away_implied_prob': min(away_prob / vig, 0.99),
            'market_vig': vig,
        }
    
    def validate_features(
        self,
        features: pd.DataFrame
    ) -> ValidationReport:
        """
        Validate feature values against expected ranges.
        
        Args:
            features: DataFrame of computed features
        
        Returns:
            ValidationReport with any issues found
        """
        errors = []
        warnings = []
        missing_count = 0
        
        for col in features.columns:
            if col in ['game_id']:
                continue
            
            values = features[col]
            missing = values.isna().sum()
            missing_count += missing
            
            if missing > 0:
                warnings.append(f"{col}: {missing} missing values ({missing/len(values):.1%})")
            
            spec = self.FEATURE_SPECS.get(col)
            if spec:
                non_null = values.dropna()
                if len(non_null) > 0:
                    min_val = non_null.min()
                    max_val = non_null.max()
                    
                    if min_val < spec['min'] * 0.8:
                        warnings.append(
                            f"{col}: min {min_val:.2f} below expected {spec['min']}"
                        )
                    
                    if max_val > spec['max'] * 1.2:
                        warnings.append(
                            f"{col}: max {max_val:.2f} above expected {spec['max']}"
                        )
        
        passed = len(errors) == 0
        
        return ValidationReport(
            passed=passed,
            errors=errors,
            warnings=warnings,
            feature_count=len(features.columns),
            missing_count=missing_count
        )
    
    def get_feature_list(self) -> List[str]:
        """Get list of all feature names."""
        return list(self.FEATURE_SPECS.keys())
    
    def clear_cache(self) -> None:
        """Clear in-memory feature cache."""
        self._cache.clear()
        logger.info("Feature cache cleared")

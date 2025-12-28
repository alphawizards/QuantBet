"""
Walk-Forward Backtesting Engine Module.

Implements the core backtesting orchestrator with walk-forward validation
to prevent data leakage and provide honest performance estimates.

Walk-Forward Validation:
    - Train on historical window (e.g., 2021-2023)
    - Test on forward window (e.g., 2024)
    - Roll forward and repeat for rolling analysis

This approach ensures the model never sees future data during training,
mimicking real-world deployment conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Any
import logging
import numpy as np
import pandas as pd

from .metrics import BacktestMetrics, calculate_metrics
from .simulator import BetSimulator, SessionResult, StakingStrategy, FractionalKellyStake
from .validator import LeakageValidator, ValidationReport

from src.strategies.kelly import BetOpportunity


logger = logging.getLogger(__name__)


# ============================================================================
# Protocols and Type Definitions
# ============================================================================

class Predictor(Protocol):
    """Protocol for prediction models."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Predictor":
        """Fit the model on training data."""
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for test data."""
        ...


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.
    
    Attributes:
        train_window: Number of seasons/periods for training
        test_window: Number of seasons/periods for testing
        refit_frequency: How often to retrain ('season', 'month', 'week')
        min_training_samples: Minimum samples required for training
        slippage_rate: Expected odds slippage (0.02 = 2%)
        min_edge_threshold: Minimum edge required to place bet
        initial_bankroll: Starting bankroll amount
        validate_leakage: Whether to run leakage validation
    """
    train_window: int = 3
    test_window: int = 1
    refit_frequency: str = "season"
    min_training_samples: int = 100
    slippage_rate: float = 0.02
    min_edge_threshold: float = 0.02
    initial_bankroll: float = 1000.0
    validate_leakage: bool = True


@dataclass
class BacktestResult:
    """
    Complete result of a backtesting run.
    
    Attributes:
        metrics: Calculated performance metrics
        session: Bet-by-bet results
        equity_curve: Bankroll over time
        predictions: Model predictions for each game
        config: Configuration used
        train_periods: Periods used for training
        test_periods: Periods used for testing
        validation_report: Leakage validation results
    """
    metrics: BacktestMetrics
    session: SessionResult
    equity_curve: pd.Series
    predictions: pd.DataFrame
    config: BacktestConfig
    train_periods: List[str] = field(default_factory=list)
    test_periods: List[str] = field(default_factory=list)
    validation_report: Optional[ValidationReport] = None
    
    def summary(self) -> str:
        """Generate summary string."""
        return f"""
Backtest Summary
================
Train: {', '.join(self.train_periods)}
Test: {', '.join(self.test_periods)}

{self.metrics}
"""


# ============================================================================
# Main Backtest Engine
# ============================================================================

class BacktestEngine:
    """
    Walk-forward backtesting engine.
    
    Implements a rigorous backtesting methodology:
    
    1. Split data into train/test periods respecting temporal order
    2. Train model on historical data only
    3. Generate predictions for future period
    4. Simulate bet execution with realistic assumptions
    5. Calculate comprehensive performance metrics
    
    This class coordinates the entire backtesting workflow and ensures
    no data leakage occurs.
    
    Example:
        >>> engine = BacktestEngine()
        >>> result = engine.run_backtest(
        ...     model=NBLPredictor(),
        ...     data=games_df,
        ...     features=feature_columns,
        ...     staking=FractionalKellyStake(0.25)
        ... )
        >>> print(result.metrics)
    """
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        simulator: Optional[BetSimulator] = None,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the backtest engine.
        
        Args:
            config: Backtesting configuration
            simulator: Bet execution simulator
            random_seed: For reproducibility
        """
        self.config = config or BacktestConfig()
        self.simulator = simulator or BetSimulator(
            slippage_rate=self.config.slippage_rate,
            random_seed=random_seed
        )
        self.validator = LeakageValidator() if self.config.validate_leakage else None
        self.random_seed = random_seed
    
    def run_backtest(
        self,
        model: Predictor,
        data: pd.DataFrame,
        features: List[str],
        target: str = "home_win",
        date_column: str = "game_date",
        season_column: str = "season",
        staking: Optional[StakingStrategy] = None,
        train_seasons: Optional[List[str]] = None,
        test_seasons: Optional[List[str]] = None
    ) -> BacktestResult:
        """
        Run a complete backtest.
        
        Args:
            model: Prediction model with fit/predict_proba methods
            data: Complete dataset with features and target
            features: List of feature column names
            target: Target column name
            date_column: Date column name
            season_column: Season column name for splitting
            staking: Staking strategy to use
            train_seasons: Specific seasons for training (optional)
            test_seasons: Specific seasons for testing (optional)
        
        Returns:
            BacktestResult with complete analysis
        """
        staking = staking or FractionalKellyStake(0.25)
        
        # Validate data if enabled
        validation_report = None
        if self.validator:
            validation_report = self.validator.validate(
                data[features + [target, date_column]],
                date_column=date_column,
                target_column=target
            )
            if not validation_report.passed:
                logger.warning("Data leakage validation failed!")
                logger.warning(str(validation_report))
        
        # Determine train/test split
        if train_seasons is None or test_seasons is None:
            train_seasons, test_seasons = self._auto_split_seasons(
                data, season_column
            )
        
        # Split data
        train_data = data[data[season_column].isin(train_seasons)].sort_values(date_column)
        test_data = data[data[season_column].isin(test_seasons)].sort_values(date_column)
        
        if len(train_data) < self.config.min_training_samples:
            raise ValueError(
                f"Insufficient training data: {len(train_data)} samples "
                f"(minimum: {self.config.min_training_samples})"
            )
        
        logger.info(f"Training on {len(train_data)} samples from {train_seasons}")
        logger.info(f"Testing on {len(test_data)} samples from {test_seasons}")
        
        # Train model
        X_train = train_data[features]
        y_train = train_data[target]
        
        model.fit(X_train, y_train)
        
        # Generate predictions
        X_test = test_data[features]
        y_test = test_data[target]
        
        probabilities = model.predict_proba(X_test)
        
        # Handle 2D output (sklearn style)
        if len(probabilities.shape) > 1:
            probabilities = probabilities[:, 1]
        
        # Create predictions DataFrame
        predictions_df = test_data[[date_column, 'home_team', 'away_team']].copy()
        predictions_df['predicted_prob'] = probabilities
        predictions_df['actual_outcome'] = y_test.values
        
        # Get odds from test data
        if 'home_odds' in test_data.columns:
            predictions_df['odds'] = test_data['home_odds'].values
        else:
            # Default odds if not available (for testing)
            predictions_df['odds'] = 2.0
        
        # Create bet opportunities
        bets_with_outcomes = self._create_bet_opportunities(
            predictions_df, 
            min_edge=self.config.min_edge_threshold
        )
        
        # Simulate betting session
        session = self.simulator.simulate_session(
            bets=bets_with_outcomes,
            staking_strategy=staking,
            initial_bankroll=self.config.initial_bankroll
        )
        
        # Calculate metrics
        bet_results_df = session.to_dataframe()
        metrics = calculate_metrics(
            bet_results=bet_results_df,
            predictions=probabilities,
            actuals=y_test.values,
            initial_bankroll=self.config.initial_bankroll
        )
        
        # Create equity curve
        equity_curve = self._create_equity_curve(session)
        
        return BacktestResult(
            metrics=metrics,
            session=session,
            equity_curve=equity_curve,
            predictions=predictions_df,
            config=self.config,
            train_periods=train_seasons,
            test_periods=test_seasons,
            validation_report=validation_report
        )
    
    def rolling_walk_forward(
        self,
        model: Predictor,
        data: pd.DataFrame,
        features: List[str],
        target: str = "home_win",
        date_column: str = "game_date",
        season_column: str = "season",
        staking: Optional[StakingStrategy] = None
    ) -> List[BacktestResult]:
        """
        Run rolling walk-forward backtests.
        
        This runs multiple backtests, each time moving the train/test
        window forward by one period.
        
        Example with 5 seasons and train_window=3, test_window=1:
            - Backtest 1: Train [S1, S2, S3], Test [S4]
            - Backtest 2: Train [S2, S3, S4], Test [S5]
        
        Args:
            model: Prediction model
            data: Complete dataset
            features: Feature column names
            target: Target column name
            date_column: Date column name
            season_column: Season column name
            staking: Staking strategy
        
        Returns:
            List of BacktestResult for each roll
        """
        staking = staking or FractionalKellyStake(0.25)
        
        # Get unique seasons in order
        seasons = sorted(data[season_column].unique())
        
        results = []
        train_window = self.config.train_window
        test_window = self.config.test_window
        
        # Iterate through possible rolls
        for i in range(len(seasons) - train_window - test_window + 1):
            train_seasons = seasons[i:i + train_window]
            test_seasons = seasons[i + train_window:i + train_window + test_window]
            
            try:
                result = self.run_backtest(
                    model=model,
                    data=data,
                    features=features,
                    target=target,
                    date_column=date_column,
                    season_column=season_column,
                    staking=staking,
                    train_seasons=train_seasons,
                    test_seasons=test_seasons
                )
                results.append(result)
                
                logger.info(
                    f"Roll {i+1}: Train {train_seasons} -> Test {test_seasons}: "
                    f"ROI={result.metrics.roi:.2%}"
                )
            except Exception as e:
                logger.warning(f"Roll {i+1} failed: {e}")
        
        return results
    
    def _auto_split_seasons(
        self,
        data: pd.DataFrame,
        season_column: str
    ) -> Tuple[List[str], List[str]]:
        """Automatically determine train/test seasons."""
        seasons = sorted(data[season_column].unique())
        
        train_window = self.config.train_window
        test_window = self.config.test_window
        
        if len(seasons) < train_window + test_window:
            raise ValueError(
                f"Not enough seasons: {len(seasons)} available, "
                f"need {train_window + test_window}"
            )
        
        train_seasons = seasons[-(train_window + test_window):-test_window]
        test_seasons = seasons[-test_window:]
        
        return list(train_seasons), list(test_seasons)
    
    def _create_bet_opportunities(
        self,
        predictions: pd.DataFrame,
        min_edge: float = 0.02
    ) -> List[Tuple[BetOpportunity, bool]]:
        """
        Create bet opportunities from predictions.
        
        Args:
            predictions: DataFrame with predicted_prob, odds, actual_outcome
            min_edge: Minimum edge required to create bet
        
        Returns:
            List of (BetOpportunity, actual_outcome) tuples
        """
        bets = []
        
        for _, row in predictions.iterrows():
            prob = row['predicted_prob']
            odds = row['odds']
            actual = row['actual_outcome']
            
            # Calculate implied probability and edge
            implied_prob = 1 / odds if odds > 0 else 0
            edge = prob - implied_prob
            
            # Only bet if edge exceeds threshold
            if edge >= min_edge:
                bet = BetOpportunity(
                    prob=prob,
                    decimal_odds=odds,
                    bet_id=f"{row.get('home_team', 'H')}_vs_{row.get('away_team', 'A')}"
                )
                bets.append((bet, bool(actual)))
        
        return bets
    
    def _create_equity_curve(
        self,
        session: SessionResult
    ) -> pd.Series:
        """Create equity curve from session results."""
        equity = [session.initial_bankroll]
        
        for bet in session.bets:
            equity.append(bet.post_bankroll)
        
        return pd.Series(equity)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_backtest(
    model: Predictor,
    data: pd.DataFrame,
    features: List[str],
    target: str = "home_win",
    initial_bankroll: float = 1000.0
) -> BacktestResult:
    """
    Quick backtest with default settings.
    
    Args:
        model: Prediction model
        data: Complete dataset
        features: Feature column names
        target: Target column name
        initial_bankroll: Starting bankroll
    
    Returns:
        BacktestResult
    """
    config = BacktestConfig(initial_bankroll=initial_bankroll)
    engine = BacktestEngine(config=config)
    
    return engine.run_backtest(
        model=model,
        data=data,
        features=features,
        target=target
    )


def compare_staking_strategies(
    model: Predictor,
    data: pd.DataFrame,
    features: List[str],
    strategies: List[StakingStrategy],
    target: str = "home_win"
) -> Dict[str, BacktestResult]:
    """
    Compare multiple staking strategies on the same model.
    
    Args:
        model: Prediction model
        data: Complete dataset
        features: Feature column names
        strategies: List of staking strategies to compare
        target: Target column name
    
    Returns:
        Dictionary of strategy_name -> BacktestResult
    """
    engine = BacktestEngine()
    results = {}
    
    for strategy in strategies:
        result = engine.run_backtest(
            model=model,
            data=data,
            features=features,
            target=target,
            staking=strategy
        )
        results[strategy.name] = result
    
    return results

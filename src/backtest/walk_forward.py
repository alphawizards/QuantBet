"""
Walk-Forward Validation for QuantBet Models.

Prevents overfitting by using rolling windows to simulate
real-world prediction scenarios.

Instead of training on all data and testing on a held-out set,
we train on a window of data and test on the next period,
then roll the window forward and repeat.

This is the gold standard for time-series prediction validation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class ValidationWindow:
    """Single window in walk-forward validation."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    def __str__(self) -> str:
        return (
            f"Train: {self.train_start.date()} to {self.train_end.date()} | "
            f"Test: {self.test_start.date()} to {self.test_end.date()}"
        )


@dataclass
class WindowResults:
    """Results from a single validation window."""
    window: ValidationWindow
    n_train: int
    n_test: int
    
    # Performance metrics
    brier_score: float
    log_loss: float
    accuracy: float
    auc_roc: Optional[float] = None
    
    # Betting metrics
    roi: Optional[float] = None
    sharpe: Optional[float] = None
    total_bets: int = 0
    profitable_bets: int = 0
    
    # Model metadata
    model_params: Optional[Dict[str, Any]] = None


class WalkForwardValidator:
    """
    Walk-forward validation framework for time series models.
    
    Example usage:
        >>> validator = WalkForwardValidator(
        ...     train_window_days=365,
        ...     test_window_days=30,
        ...     step_days=30
        ... )
        >>> results = validator.validate(model, data)
        >>> print(f"Avg Brier: {results.mean_brier:.4f}")
    """
    
    def __init__(
        self,
        train_window_days: int = 365,
        test_window_days: int = 30,
        step_days: int = 30,
        min_train_samples: int = 100
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_window_days: Size of training window in days
            test_window_days: Size of test window in days
            step_days: How far to step forward each iteration
            min_train_samples: Minimum samples required to train
        """
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.step = timedelta(days=step_days)
        self.min_train_samples = min_train_samples
        
        logger.info(
            f"Walk-forward validator initialized: "
            f"train={train_window_days}d, test={test_window_days}d, "
            f"step={step_days}d"
        )
    
    def create_windows(
        self,
        data: pd.DataFrame,
        date_column: str = 'game_date'
    ) -> List[ValidationWindow]:
        """
        Create all validation windows for the dataset.
        
        Args:
            data: DataFrame with time-indexed data
            date_column: Name of date column
        
        Returns:
            List of ValidationWindow objects
        """
        # Ensure dates are datetime
        data[date_column] = pd.to_datetime(data[date_column])
        
        min_date = data[date_column].min()
        max_date = data[date_column].max()
        
        windows = []
        
        # Start with first possible window
        current_train_end = min_date + self.train_window
        
        while current_train_end + self.test_window <= max_date:
            train_start = current_train_end - self.train_window
            train_end = current_train_end
            test_start = train_end
            test_end = test_start + self.test_window
            
            # Verify sufficient training data
            train_mask = (
                (data[date_column] >= train_start) &
                (data[date_column] < train_end)
            )
            
            if train_mask.sum() >= self.min_train_samples:
                windows.append(ValidationWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end
                ))
            
            # Step forward
            current_train_end += self.step
        
        logger.info(f"Created {len(windows)} validation windows")
        return windows
    
    def validate(
        self,
        model_factory: Callable,
        data: pd.DataFrame,
        target_column: str = 'home_win',
        date_column: str = 'game_date',
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation on a model.
        
        Args:
            model_factory: Function that returns a fresh model instance
            data: DataFrame with game data
            target_column: Name of target variable
            date_column: Name of date column
            feature_columns: List of feature column names (None = auto-detect)
        
        Returns:
            Dictionary with aggregated validation results
        """
        windows = self.create_windows(data, date_column)
        
        if not windows:
            raise ValueError("No validation windows could be created")
        
        # Auto-detect features if not specified
        if feature_columns is None:
            exclude = [target_column, date_column, 'game_id', 'home_team', 'away_team']
            feature_columns = [
                col for col in data.columns
                if col not in exclude and data[col].dtype in [np.float64, np.int64]
            ]
        
        logger.info(f"Using {len(feature_columns)} features")
        
        window_results = []
        
        for i, window in enumerate(windows, 1):
            logger.info(f"Window {i}/{len(windows)}: {window}")
            
            # Split data
            train_mask = (
                (data[date_column] >= window.train_start) &
                (data[date_column] < window.train_end)
            )
            test_mask = (
                (data[date_column] >= window.test_start) &
                (data[date_column] < window.test_end)
            )
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            # Skip if insufficient data
            if len(train_data) < self.min_train_samples or len(test_data) == 0:
                logger.warning(f"Skipping window {i}: insufficient data")
                continue
            
            # Extract features and targets
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Remove NaN
            train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())
            
            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            X_test = X_test[test_valid]
            y_test = y_test[test_valid]
            
            # Train model
            try:
                model = model_factory()
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_proba = model.predict_proba(X_test)
                if hasattr(y_pred_proba, 'shape') and len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # Class 1 probability
                
                y_pred = (y_pred_proba >= 0.5).astype(int)
                
                # Calculate metrics
                brier = brier_score_loss(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                accuracy = (y_pred == y_test).mean()
                
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = None
                
                window_results.append(WindowResults(
                    window=window,
                    n_train=len(X_train),
                    n_test=len(X_test),
                    brier_score=brier,
                    log_loss=logloss,
                    accuracy=accuracy,
                    auc_roc=auc
                ))
                
                logger.info(
                    f"  Brier: {brier:.4f}, LogLoss: {logloss:.4f}, "
                    f"Accuracy: {accuracy:.2%}"
                )
                
            except Exception as e:
                logger.error(f"Error in window {i}: {e}")
                continue
        
        # Aggregate results
        if not window_results:
            raise ValueError("No successful validation windows")
        
        brier_scores = [r.brier_score for r in window_results]
        log_losses = [r.log_loss for r in window_results]
        accuracies = [r.accuracy for r in window_results]
        
        return {
            'n_windows': len(window_results),
            'mean_brier': np.mean(brier_scores),
            'std_brier': np.std(brier_scores),
            'mean_log_loss': np.mean(log_losses),
            'std_log_loss': np.std(log_losses),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'window_results': window_results,
            'summary': self._create_summary(window_results)
        }
    
    def _create_summary(self, results: List[WindowResults]) -> str:
        """Create human-readable summary of results."""
        if not results:
            return "No results"
        
        brier_scores = [r.brier_score for r in results]
        accuracies = [r.accuracy for r in results]
        
        summary = f"""
Walk-Forward Validation Summary
================================
Windows: {len(results)}
        
Brier Score:
  Mean: {np.mean(brier_scores):.4f}
  Std:  {np.std(brier_scores):.4f}
  Min:  {np.min(brier_scores):.4f}
  Max:  {np.max(brier_scores):.4f}

Accuracy:
  Mean: {np.mean(accuracies):.2%}
  Std:  {np.std(accuracies):.2%}
  Min:  {np.min(accuracies):.2%}
  Max:  {np.max(accuracies):.2%}

Trend:
  {'✅ Stable' if np.std(brier_scores) < 0.05 else '⚠️ Variable'}
        """
        
        return summary.strip()


def run_walk_forward_validation(
    model_factory: Callable,
    data_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run walk-forward validation from file.
    
    Args:
        model_factory: Function that returns a model instance
        data_path: Path to CSV with game data
        output_path: Optional path to save results JSON
    
    Returns:
        Validation results dictionary
    """
    data = pd.read_csv(data_path)
    
    validator = WalkForwardValidator(
        train_window_days=365,
        test_window_days=30,
        step_days=30
    )
    
    results = validator.validate(model_factory, data)
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            # Convert window_results to serializable format
            serializable_results = {
                k: v for k, v in results.items()
                if k != 'window_results'
            }
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    print(results['summary'])
    
    return results

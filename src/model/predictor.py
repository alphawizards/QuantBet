"""
XGBoost Predictor for NBL/WNBL Game Outcomes.

This module implements a Walk-Forward validation approach for training
and evaluating probabilistic game outcome predictions.

Walk-Forward Validation:
    - Train on historical data (e.g., 2021-2023 seasons)
    - Test on future data (e.g., 2024 season)
    - Never use future information in training
    - Critical for honest evaluation of betting models

Key Features:
    - Probability calibration via Platt/Isotonic scaling
    - Cross-validation for hyperparameter tuning
    - Feature importance analysis
    - Proper train/test splits respecting temporal order
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .calibration import PlattCalibrator, IsotonicCalibrator, evaluate_calibration


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    brier_score: float          # Lower is better, perfect = 0
    log_loss: float             # Lower is better
    roc_auc: float              # Higher is better, perfect = 1
    ece: float                  # Expected Calibration Error
    mce: float                  # Maximum Calibration Error
    accuracy: float             # Classification accuracy at 0.5 threshold
    
    def __str__(self) -> str:
        return (
            f"Brier Score: {self.brier_score:.4f}\n"
            f"Log Loss: {self.log_loss:.4f}\n"
            f"ROC AUC: {self.roc_auc:.4f}\n"
            f"ECE: {self.ece:.4f}\n"
            f"MCE: {self.mce:.4f}\n"
            f"Accuracy: {self.accuracy:.4f}"
        )


@dataclass
class PredictionResult:
    """Container for a single prediction with metadata."""
    game_id: str
    home_team: str
    away_team: str
    game_date: datetime
    raw_probability: float          # Uncalibrated p(home_win)
    calibrated_probability: float   # Calibrated p(home_win)
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    
    @property
    def away_probability(self) -> float:
        """Probability of away team winning."""
        return 1.0 - self.calibrated_probability


@dataclass
class XGBConfig:
    """
    XGBoost hyperparameter configuration.
    
    Default values are Optuna-tuned on NBL/WNBL historical data
    (2,136 games, 2009-2025) optimizing for Brier score.
    """
    n_estimators: int = 394
    max_depth: int = 5
    learning_rate: float = 0.038
    subsample: float = 0.94
    colsample_bytree: float = 0.77
    min_child_weight: int = 3
    gamma: float = 4.99
    reg_alpha: float = 1.32
    reg_lambda: float = 9.88
    scale_pos_weight: float = 1.0
    random_state: int = 42
    
    def to_dict(self) -> Dict:
        """Convert to XGBoost params dict."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
        }


class NBLPredictor:
    """
    XGBoost-based predictor for NBL/WNBL game outcomes.
    
    Uses Walk-Forward validation to honestly evaluate model performance,
    ensuring no future data leakage. Outputs calibrated probabilities
    suitable for Kelly Criterion betting calculations.
    
    Walk-Forward Validation Approach:
    
    .. code-block:: text
    
        Training Window               Test Period
        ┌─────────────────────────────┬───────────┐
        │     2021  │  2022  │  2023  │    2024   │
        └─────────────────────────────┴───────────┘
                                      ↑
                                Prediction starts here
    
    The model NEVER sees future data during training, mimicking real-world
    deployment where we predict upcoming games.
    
    Example:
        >>> config = XGBConfig(max_depth=4, n_estimators=200)
        >>> predictor = NBLPredictor(config=config)
        >>> 
        >>> # Train/test split
        >>> train_data = data[data['season'].isin(['2021-22', '2022-23', '2023-24'])]
        >>> test_data = data[data['season'] == '2024-25']
        >>> 
        >>> # Fit and predict
        >>> predictor.fit(train_data[features], train_data['home_win'])
        >>> predictions = predictor.predict_proba(test_data[features])
    
    Attributes:
        config: XGBoost hyperparameter configuration
        model: Fitted XGBoost model (None until fit is called)
        calibrator: Probability calibrator (Platt or Isotonic)
        feature_names: List of feature names used in training
    """
    
    def __init__(
        self,
        config: Optional[XGBConfig] = None,
        calibration_method: str = 'platt'
    ):
        """
        Initialize the predictor.
        
        Args:
            config: XGBoost configuration. Uses defaults if None.
            calibration_method: 'platt' or 'isotonic'
        """
        self.config = config or XGBConfig()
        self._model: Optional[xgb.XGBClassifier] = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        
        # Initialize calibrator
        if calibration_method.lower() == 'platt':
            self._calibrator = PlattCalibrator()
        elif calibration_method.lower() == 'isotonic':
            self._calibrator = IsotonicCalibrator()
        else:
            raise ValueError(f"Unknown calibration method: {calibration_method}")
        
        self._calibration_method = calibration_method
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibration_split: float = 0.2,
        early_stopping_rounds: Optional[int] = 20,
        verbose: bool = False
    ) -> "NBLPredictor":
        """
        Fit the XGBoost model with probability calibration.
        
        Training procedure:
            1. Split off calibration set (last portion of data)
            2. Train XGBoost on training portion
            3. Fit calibrator on held-out calibration set
        
        This ensures calibration is performed on unseen data for
        honest probability estimation.
        
        Args:
            X: Feature DataFrame
            y: Target Series (1 = home win, 0 = away win)
            calibration_split: Fraction of data for calibration (temporal)
            early_stopping_rounds: Stop training if no improvement
            verbose: Print training progress
        
        Returns:
            Self for method chaining
        """
        self._feature_names = list(X.columns)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Temporal split for calibration
        n_samples = len(X)
        split_idx = int(n_samples * (1 - calibration_split))
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_calib = X.iloc[split_idx:]
        y_calib = y.iloc[split_idx:]
        
        # Initialize and train XGBoost
        self._model = xgb.XGBClassifier(**self.config.to_dict())
        
        # Create evaluation set for early stopping
        if early_stopping_rounds:
            eval_set = [(X_calib, y_calib)]
            self._model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=verbose
            )
        else:
            self._model.fit(X_train, y_train)
        
        # Fit calibrator on calibration set
        raw_probs = self._model.predict_proba(X_calib)[:, 1]
        self._calibrator.fit(raw_probs, y_calib.values)
        
        self._is_fitted = True
        
        return self
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        calibrate: bool = True
    ) -> np.ndarray:
        """
        Predict probability of home team winning.
        
        Args:
            X: Feature DataFrame
            calibrate: Whether to apply calibration
        
        Returns:
            Array of probabilities p(home_win)
        
        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        
        # Handle missing values same as training
        X = X.fillna(X.median())
        
        # Get raw probabilities
        raw_probs = self._model.predict_proba(X)[:, 1]
        
        if calibrate and self._calibrator.is_fitted:
            return self._calibrator.calibrate(raw_probs)
        
        return raw_probs
    
    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict binary outcome (1 = home win, 0 = away win).
        
        Args:
            X: Feature DataFrame
            threshold: Probability threshold for positive class
        
        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X, calibrate=True)
        return (probs >= threshold).astype(int)
    
    def predict_games(
        self,
        games: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[PredictionResult]:
        """
        Generate detailed predictions for games.
        
        Args:
            games: DataFrame with game metadata (game_id, home_team, etc.)
            features: Feature DataFrame aligned with games
        
        Returns:
            List of PredictionResult objects
        """
        raw_probs = self.predict_proba(features, calibrate=False)
        calibrated_probs = self.predict_proba(features, calibrate=True)
        
        results = []
        for i, (_, game) in enumerate(games.iterrows()):
            result = PredictionResult(
                game_id=game.get('game_id', str(i)),
                home_team=game['home_team'],
                away_team=game['away_team'],
                game_date=game['game_date'],
                raw_probability=float(raw_probs[i]),
                calibrated_probability=float(calibrated_probs[i])
            )
            results.append(result)
        
        return results
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> ModelMetrics:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature DataFrame
            y: True labels
        
        Returns:
            ModelMetrics with all evaluation scores
        """
        probs = self.predict_proba(X, calibrate=True)
        preds = (probs >= 0.5).astype(int)
        
        # Core metrics
        brier = brier_score_loss(y, probs)
        logloss = log_loss(y, probs)
        auc = roc_auc_score(y, probs)
        accuracy = (preds == y).mean()
        
        # Calibration metrics
        ece, mce, _, _ = evaluate_calibration(probs, y.values)
        
        return ModelMetrics(
            brier_score=brier,
            log_loss=logloss,
            roc_auc=auc,
            ece=ece,
            mce=mce,
            accuracy=accuracy
        )
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Uses TimeSeriesSplit to maintain temporal ordering and
        prevent data leakage.
        
        Args:
            X: Feature DataFrame
            y: True labels
            n_splits: Number of CV folds
        
        Returns:
            Dictionary of metric name -> list of fold scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results: Dict[str, List[float]] = {
            'brier_score': [],
            'log_loss': [],
            'roc_auc': [],
            'accuracy': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create fresh predictor for each fold
            fold_predictor = NBLPredictor(
                config=self.config,
                calibration_method=self._calibration_method
            )
            fold_predictor.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            metrics = fold_predictor.evaluate(X_test, y_test)
            
            results['brier_score'].append(metrics.brier_score)
            results['log_loss'].append(metrics.log_loss)
            results['roc_auc'].append(metrics.roc_auc)
            results['accuracy'].append(metrics.accuracy)
        
        return results
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'
        
        Returns:
            DataFrame with features and importance scores, sorted descending
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted to get feature importance")
        
        importances = self._model.get_booster().get_score(
            importance_type=importance_type
        )
        
        df = pd.DataFrame([
            {'feature': f, 'importance': imp}
            for f, imp in importances.items()
        ])
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    @property
    def feature_names(self) -> List[str]:
        """List of features used in training."""
        return self._feature_names
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    @classmethod
    def tune_hyperparameters(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        timeout: Optional[int] = 3600,
        calibration_method: str = 'platt'
    ) -> Tuple["NBLPredictor", "XGBConfig"]:
        """
        Tune hyperparameters and return a fitted predictor with optimal config.
        
        Uses Optuna for Bayesian optimization with TimeSeriesSplit CV
        to find the best XGBoost hyperparameters for Brier score minimization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_trials: Maximum number of Optuna trials
            timeout: Maximum time in seconds (None = no limit)
            calibration_method: 'platt' or 'isotonic'
        
        Returns:
            Tuple of (fitted NBLPredictor with optimal config, XGBConfig)
        
        Example:
            >>> predictor, config = NBLPredictor.tune_hyperparameters(
            ...     X_train, y_train, n_trials=50
            ... )
            >>> print(f"Best Brier: {predictor.evaluate(X_test, y_test).brier_score}")
        """
        from .tuning import HyperparameterTuner
        
        tuner = HyperparameterTuner(calibrate=(calibration_method == 'platt'))
        result = tuner.tune(X, y, n_trials=n_trials, timeout=timeout)
        
        config = result.to_xgb_config()
        
        # Create and fit predictor with optimal config
        predictor = cls(config=config, calibration_method=calibration_method)
        predictor.fit(X, y)
        
        return predictor, config



def walk_forward_backtest(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'home_win',
    train_seasons: List[str] = ['2021-22', '2022-23', '2023-24'],
    test_season: str = '2024-25',
    config: Optional[XGBConfig] = None
) -> Tuple[NBLPredictor, List[PredictionResult], ModelMetrics]:
    """
    Perform Walk-Forward backtesting.
    
    This is the recommended workflow for honest evaluation:
        1. Train on historical seasons
        2. Predict on future season
        3. Evaluate calibrated probabilities
    
    Args:
        data: Full dataset with 'season' column
        feature_columns: List of feature column names
        target_column: Name of target column
        train_seasons: Seasons to train on
        test_season: Season to test on
        config: XGBoost configuration
    
    Returns:
        Tuple of (fitted predictor, predictions, metrics)
    
    Example:
        >>> predictor, predictions, metrics = walk_forward_backtest(
        ...     data=games_df,
        ...     feature_columns=['home_ortg_l5', 'away_ortg_l5', ...],
        ...     train_seasons=['2021-22', '2022-23', '2023-24'],
        ...     test_season='2024-25'
        ... )
        >>> print(metrics)
    """
    # Split data by season
    train_data = data[data['season'].isin(train_seasons)].sort_values('game_date')
    test_data = data[data['season'] == test_season].sort_values('game_date')
    
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    # Train predictor
    predictor = NBLPredictor(config=config)
    predictor.fit(X_train, y_train)
    
    # Generate predictions
    predictions = predictor.predict_games(test_data, X_test)
    
    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)
    
    return predictor, predictions, metrics

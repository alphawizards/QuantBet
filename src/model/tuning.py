"""
Hyperparameter Tuning Module for NBL/WNBL Prediction Models.

This module implements Bayesian hyperparameter optimization using Optuna
for XGBoost models with proper time-series cross-validation.

Key Features:
    - Optuna-based Bayesian optimization
    - TimeSeriesSplit cross-validation (prevents data leakage)
    - Brier score minimization objective
    - Configurable search spaces
    - Early stopping for efficiency
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .calibration import PlattCalibrator

logger = logging.getLogger(__name__)


# ============================================================================
# Search Space Configuration
# ============================================================================

@dataclass
class XGBSearchSpace:
    """
    Configurable search space for XGBoost hyperparameters.
    
    Default ranges are tuned for sports betting probability estimation
    where calibration and minimal overfitting are critical.
    
    Attributes:
        max_depth: Tree depth range (smaller = less overfitting)
        n_estimators: Number of trees range
        learning_rate: Step size shrinkage range (log-uniform)
        subsample: Row subsampling range
        colsample_bytree: Column subsampling range
        min_child_weight: Minimum sum of instance weight in child
        reg_alpha: L1 regularization (log-uniform)
        reg_lambda: L2 regularization (log-uniform)
        gamma: Minimum loss reduction for split
    """
    max_depth: Tuple[int, int] = (2, 8)
    n_estimators: Tuple[int, int] = (50, 500)
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    subsample: Tuple[float, float] = (0.6, 1.0)
    colsample_bytree: Tuple[float, float] = (0.6, 1.0)
    min_child_weight: Tuple[int, int] = (1, 10)
    reg_alpha: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda: Tuple[float, float] = (1e-8, 10.0)
    gamma: Tuple[float, float] = (0.0, 5.0)


@dataclass
class TuningResult:
    """
    Result of hyperparameter tuning.
    
    Attributes:
        best_params: Dictionary of optimal hyperparameters
        best_score: Best Brier score achieved
        n_trials: Number of trials completed
        study_history: List of (params, score) tuples for analysis
        cv_scores: Cross-validation scores for best params
    """
    best_params: Dict
    best_score: float
    n_trials: int
    study_history: List[Tuple[Dict, float]] = field(default_factory=list)
    cv_scores: List[float] = field(default_factory=list)
    
    def to_xgb_config(self):
        """Convert to XGBConfig for predictor initialization."""
        from .predictor import XGBConfig
        
        return XGBConfig(
            max_depth=self.best_params.get('max_depth', 4),
            n_estimators=self.best_params.get('n_estimators', 100),
            learning_rate=self.best_params.get('learning_rate', 0.1),
            subsample=self.best_params.get('subsample', 0.8),
            colsample_bytree=self.best_params.get('colsample_bytree', 0.8),
            min_child_weight=self.best_params.get('min_child_weight', 1),
            gamma=self.best_params.get('gamma', 0.0),
            reg_alpha=self.best_params.get('reg_alpha', 0.0),
            reg_lambda=self.best_params.get('reg_lambda', 1.0),
        )
    
    def __str__(self) -> str:
        return (
            f"TuningResult:\n"
            f"  Best Brier Score: {self.best_score:.4f}\n"
            f"  Trials: {self.n_trials}\n"
            f"  CV Score Mean: {np.mean(self.cv_scores):.4f} ± {np.std(self.cv_scores):.4f}\n"
            f"  Best Params:\n" + 
            "\n".join(f"    {k}: {v}" for k, v in self.best_params.items())
        )


# ============================================================================
# Hyperparameter Tuner Class
# ============================================================================

class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for XGBoost models.
    
    Uses Bayesian optimization (TPE sampler) to efficiently search the
    hyperparameter space, with proper time-series cross-validation to
    prevent data leakage.
    
    Example:
        >>> tuner = HyperparameterTuner(n_cv_splits=5)
        >>> result = tuner.tune(X_train, y_train, n_trials=100)
        >>> print(result.best_params)
        >>> 
        >>> # Use tuned config
        >>> predictor = NBLPredictor(config=result.to_xgb_config())
    
    Attributes:
        search_space: XGBSearchSpace defining parameter ranges
        n_cv_splits: Number of TimeSeriesSplit folds
        calibrate: Whether to apply probability calibration during tuning
        random_state: For reproducibility
    """
    
    def __init__(
        self,
        search_space: Optional[XGBSearchSpace] = None,
        n_cv_splits: int = 5,
        calibrate: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            search_space: Custom search space. Uses defaults if None.
            n_cv_splits: Number of time-series CV splits
            calibrate: Whether to calibrate probabilities in objective
            random_state: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )
        
        self.search_space = search_space or XGBSearchSpace()
        self.n_cv_splits = n_cv_splits
        self.calibrate = calibrate
        self.random_state = random_state
        
        self._study: Optional[optuna.Study] = None
    
    def _create_objective(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Callable:
        """
        Create the Optuna objective function.
        
        The objective minimizes Brier score using TimeSeriesSplit CV.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        space = self.search_space
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {
                'max_depth': trial.suggest_int(
                    'max_depth', space.max_depth[0], space.max_depth[1]
                ),
                'n_estimators': trial.suggest_int(
                    'n_estimators', space.n_estimators[0], space.n_estimators[1]
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate', space.learning_rate[0], space.learning_rate[1], log=True
                ),
                'subsample': trial.suggest_float(
                    'subsample', space.subsample[0], space.subsample[1]
                ),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree', space.colsample_bytree[0], space.colsample_bytree[1]
                ),
                'min_child_weight': trial.suggest_int(
                    'min_child_weight', space.min_child_weight[0], space.min_child_weight[1]
                ),
                'reg_alpha': trial.suggest_float(
                    'reg_alpha', space.reg_alpha[0], space.reg_alpha[1], log=True
                ),
                'reg_lambda': trial.suggest_float(
                    'reg_lambda', space.reg_lambda[0], space.reg_lambda[1], log=True
                ),
                'gamma': trial.suggest_float(
                    'gamma', space.gamma[0], space.gamma[1]
                ),
            }
            
            # Add fixed params
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': self.random_state,
                'use_label_encoder': False,
            })
            
            # Cross-validation
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                # Handle missing values
                X_train_cv = X_train_cv.fillna(X_train_cv.median())
                X_val_cv = X_val_cv.fillna(X_train_cv.median())
                
                # Train model
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    verbose=False
                )
                
                # Get predictions
                probs = model.predict_proba(X_val_cv)[:, 1]
                
                # Apply calibration if requested
                if self.calibrate:
                    # Use last 20% of training data for calibration
                    calib_split = int(len(X_train_cv) * 0.8)
                    X_calib = X_train_cv.iloc[calib_split:]
                    y_calib = y_train_cv.iloc[calib_split:]
                    calib_probs = model.predict_proba(X_calib)[:, 1]
                    
                    calibrator = PlattCalibrator()
                    calibrator.fit(calib_probs, y_calib.values)
                    probs = calibrator.calibrate(probs)
                
                # Calculate Brier score
                brier = brier_score_loss(y_val_cv, probs)
                cv_scores.append(brier)
            
            return np.mean(cv_scores)
        
        return objective
    
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True,
        pruning: bool = True
    ) -> TuningResult:
        """
        Run hyperparameter tuning.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_trials: Maximum number of trials
            timeout: Maximum time in seconds (None = no limit)
            show_progress: Whether to show Optuna progress bar
            pruning: Whether to use median pruning for early stopping
        
        Returns:
            TuningResult with best parameters and study history
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        
        if pruning:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        self._study = optuna.create_study(
            direction='minimize',  # Minimize Brier score
            sampler=sampler,
            pruner=pruner
        )
        
        # Create objective
        objective = self._create_objective(X, y)
        
        # Run optimization
        verbosity = optuna.logging.INFO if show_progress else optuna.logging.WARNING
        optuna.logging.set_verbosity(verbosity)
        
        self._study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )
        
        # Extract results
        best_trial = self._study.best_trial
        
        # Get CV scores for best params
        cv_scores = self._get_cv_scores(X, y, best_trial.params)
        
        # Build history
        history = [
            (trial.params, trial.value)
            for trial in self._study.trials
            if trial.value is not None
        ]
        
        result = TuningResult(
            best_params=best_trial.params,
            best_score=best_trial.value,
            n_trials=len(self._study.trials),
            study_history=history,
            cv_scores=cv_scores
        )
        
        logger.info(f"Tuning complete. Best Brier score: {result.best_score:.4f}")
        
        return result
    
    def _get_cv_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict
    ) -> List[float]:
        """Get individual CV fold scores for the final params."""
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        
        full_params = {
            **params,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'use_label_encoder': False,
        }
        
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train_cv = X.iloc[train_idx].fillna(X.iloc[train_idx].median())
            X_val_cv = X.iloc[val_idx].fillna(X.iloc[train_idx].median())
            y_train_cv = y.iloc[train_idx]
            y_val_cv = y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**full_params)
            model.fit(X_train_cv, y_train_cv, verbose=False)
            
            probs = model.predict_proba(X_val_cv)[:, 1]
            scores.append(brier_score_loss(y_val_cv, probs))
        
        return scores
    
    def get_param_importances(self) -> Dict[str, float]:
        """
        Get hyperparameter importance scores.
        
        Returns:
            Dictionary of parameter name -> importance score
        
        Raises:
            RuntimeError: If tune() hasn't been called yet
        """
        if self._study is None:
            raise RuntimeError("Must call tune() before getting importances")
        
        try:
            importances = optuna.importance.get_param_importances(self._study)
            return dict(importances)
        except Exception as e:
            logger.warning(f"Could not calculate param importances: {e}")
            return {}
    
    @property
    def study(self) -> Optional[optuna.Study]:
        """Access the underlying Optuna study for advanced analysis."""
        return self._study


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_tune(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    timeout: int = 600
) -> TuningResult:
    """
    Quick hyperparameter tuning with sensible defaults.
    
    Args:
        X: Feature DataFrame
        y: Target Series  
        n_trials: Maximum trials (default 50)
        timeout: Maximum time in seconds (default 10 minutes)
    
    Returns:
        TuningResult with best parameters
    
    Example:
        >>> result = quick_tune(X_train, y_train)
        >>> predictor = NBLPredictor(config=result.to_xgb_config())
    """
    tuner = HyperparameterTuner(n_cv_splits=3)
    return tuner.tune(X, y, n_trials=n_trials, timeout=timeout)


def tune_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100
) -> TuningResult:
    """
    Tune on training data and validate on held-out validation set.
    
    This is useful when you have a separate validation set and want
    to verify the tuned parameters generalize well.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_trials: Maximum trials
    
    Returns:
        TuningResult with validation score added
    """
    # Tune on training data
    tuner = HyperparameterTuner()
    result = tuner.tune(X_train, y_train, n_trials=n_trials)
    
    # Validate on held-out set
    config = result.to_xgb_config()
    
    from .predictor import NBLPredictor
    predictor = NBLPredictor(config=config)
    predictor.fit(X_train, y_train)
    
    val_probs = predictor.predict_proba(X_val)
    val_score = brier_score_loss(y_val, val_probs)
    
    logger.info(f"Validation Brier score: {val_score:.4f}")
    
    # Add validation score to result
    result.cv_scores.append(val_score)
    
    return result

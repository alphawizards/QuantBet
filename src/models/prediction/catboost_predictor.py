"""
CatBoost Predictor for NBL/WNBL Game Outcomes.

CatBoost excels at handling categorical features (teams, venues, referees)
which are common in sports betting models.

Key advantages:
    - Best-in-class categorical feature handling
    - Often better calibrated than XGBoost/LightGBM
    - GPU training out-of-box
    - Robust to overfitting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .calibration import PlattCalibrator, IsotonicCalibrator, evaluate_calibration

logger = logging.getLogger(__name__)


@dataclass
class CatBoostConfig:
    """CatBoost hyperparameter configuration.
    
    Tuned for NBL/WNBL game prediction with categorical features.
    """
    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    random_seed: int = 42
    
    # Categorical features
    cat_features: List[str] = field(default_factory=lambda: [
        'home_team', 'away_team', 'venue'
    ])
    
    # Early stopping
    early_stopping_rounds: int = 30
    
    # GPU/CPU
    task_type: str = 'CPU'  # 'GPU' if available
    
    def to_dict(self) -> Dict:
        """Convert to CatBoost params dict."""
        return {
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'l2_leaf_reg': self.l2_leaf_reg,
            'random_seed': self.random_seed,
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'task_type': self.task_type,
            'verbose': False,
        }


class CatBoostPredictor:
    """
    CatBoost-based predictor for NBL/WNBL game outcomes.
    
    Optimized for categorical feature handling which is critical for:
        - Team identifiers (home_team, away_team)
        - Venue information
        - Referee assignments
    
    CatBoost uses ordered target encoding for categoricals, avoiding
    target leakage that can occur with standard label encoding.
    
    Example:
        >>> predictor = CatBoostPredictor()
        >>> predictor.fit(X_train, y_train)
        >>> probs = predictor.predict_proba(X_test)
        >>> 
        >>> # Get feature importance
        >>> importance = predictor.get_feature_importance()
    """
    
    def __init__(
        self,
        config: Optional[CatBoostConfig] = None,
        calibration_method: str = 'platt'
    ):
        """
        Initialize the CatBoost predictor.
        
        Args:
            config: CatBoost configuration. Uses defaults if None.
            calibration_method: 'platt' or 'isotonic'
        """
        self.config = config or CatBoostConfig()
        self.calibration_method = calibration_method
        
        self.model: Optional[CatBoostClassifier] = None
        self.calibrator: Optional[PlattCalibrator | IsotonicCalibrator] = None
        self._feature_names: List[str] = []
        self._cat_feature_indices: List[int] = []
        self._is_fitted = False
    
    def _identify_cat_features(self, X: pd.DataFrame) -> List[int]:
        """Identify categorical feature indices."""
        cat_indices = []
        
        for i, col in enumerate(X.columns):
            # Check if in config
            if col in self.config.cat_features:
                cat_indices.append(i)
            # Or if dtype is object/category
            elif X[col].dtype in ['object', 'category']:
                cat_indices.append(i)
        
        return cat_indices
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibration_split: float = 0.2,
        verbose: bool = False
    ) -> 'CatBoostPredictor':
        """
        Fit the CatBoost model with probability calibration.
        
        Args:
            X: Feature DataFrame
            y: Binary target (1 = home win, 0 = away win)
            calibration_split: Fraction for calibration set
            verbose: Print training progress
        
        Returns:
            Self for method chaining
        """
        self._feature_names = list(X.columns)
        self._cat_feature_indices = self._identify_cat_features(X)
        
        # Split for calibration
        split_idx = int(len(X) * (1 - calibration_split))
        X_train, X_cal = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_cal = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Create CatBoost pools
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=self._cat_feature_indices if self._cat_feature_indices else None
        )
        
        cal_pool = Pool(
            data=X_cal,
            label=y_cal,
            cat_features=self._cat_feature_indices if self._cat_feature_indices else None
        )
        
        # Create and fit model
        params = self.config.to_dict()
        self.model = CatBoostClassifier(**params)
        
        self.model.fit(
            train_pool,
            eval_set=cal_pool,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=verbose
        )
        
        # Calibrate probabilities
        raw_probs = self.model.predict_proba(X_cal)[:, 1]
        
        if self.calibration_method == 'platt':
            self.calibrator = PlattCalibrator()
        else:
            self.calibrator = IsotonicCalibrator()
        
        self.calibrator.fit(raw_probs, y_cal.values)
        
        self._is_fitted = True
        
        best_iter = self.model.get_best_iteration() if hasattr(self.model, 'get_best_iteration') else self.config.iterations
        logger.info(
            f"CatBoost fitted: {best_iter} iterations, "
            f"{len(self._feature_names)} features, "
            f"{len(self._cat_feature_indices)} categorical"
        )
        
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
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        raw_probs = self.model.predict_proba(X)[:, 1]
        
        if calibrate and self.calibrator is not None:
            return self.calibrator.predict(raw_probs)
        
        return raw_probs
    
    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict binary outcome.
        
        Args:
            X: Feature DataFrame
            threshold: Probability threshold
        
        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: True labels
        
        Returns:
            Dictionary of metrics
        """
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        
        # Calibration metrics
        ece, mce = evaluate_calibration(y.values, probs)
        
        return {
            'brier_score': brier_score_loss(y, probs),
            'log_loss': log_loss(y, probs),
            'roc_auc': roc_auc_score(y, probs),
            'ece': ece,
            'mce': mce,
            'accuracy': (preds == y).mean()
        }
    
    def get_feature_importance(
        self,
        importance_type: str = 'FeatureImportance'
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'FeatureImportance', 'PredictionValuesChange',
                           or 'LossFunctionChange'
        
        Returns:
            DataFrame with features and importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        importance = self.model.get_feature_importance(type=importance_type)
        
        return pd.DataFrame({
            'feature': self._feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: True labels
            n_splits: Number of CV folds
        
        Returns:
            Dictionary of metric name -> list of fold scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {
            'brier_score': [],
            'log_loss': [],
            'roc_auc': [],
            'accuracy': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit temporary model
            temp_model = CatBoostPredictor(config=self.config)
            temp_model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            metrics = temp_model.evaluate(X_val, y_val)
            
            for metric_name, value in metrics.items():
                if metric_name in results:
                    results[metric_name].append(value)
            
            logger.debug(f"Fold {fold + 1}: Brier={metrics['brier_score']:.4f}")
        
        return results
    
    @property
    def feature_names(self) -> List[str]:
        """List of features used in training."""
        return self._feature_names.copy()
    
    @property
    def categorical_features(self) -> List[str]:
        """List of categorical features identified."""
        return [self._feature_names[i] for i in self._cat_feature_indices]

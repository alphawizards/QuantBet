"""
LightGBM Predictor for NBL/WNBL Game Outcomes.

Implements a LightGBM classifier as an ensemble member alongside XGBoost.
LightGBM offers:
    - 3-10x faster training than XGBoost
    - Better handling of categorical features
    - Leaf-wise tree growth (vs XGBoost's level-wise)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .calibration import PlattCalibrator, IsotonicCalibrator, evaluate_calibration

logger = logging.getLogger(__name__)


@dataclass
class LGBMConfig:
    """LightGBM hyperparameter configuration.
    
    Default values tuned for NBL/WNBL game prediction.
    """
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    
    # Categorical feature handling
    categorical_features: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to LightGBM params dict."""
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'verbose': -1,
            'force_col_wise': True,
        }


class LightGBMPredictor:
    """
    LightGBM-based predictor for NBL/WNBL game outcomes.
    
    This predictor serves as an ensemble member alongside XGBoost,
    providing diversity in the prediction ensemble.
    
    Key features:
        - Faster training than XGBoost
        - Native categorical feature support
        - Probability calibration via Platt scaling
    
    Example:
        >>> predictor = LightGBMPredictor()
        >>> predictor.fit(X_train, y_train)
        >>> probs = predictor.predict_proba(X_test)
    """
    
    def __init__(
        self,
        config: Optional[LGBMConfig] = None,
        calibration_method: str = 'platt'
    ):
        """
        Initialize the LightGBM predictor.
        
        Args:
            config: LightGBM configuration. Uses defaults if None.
            calibration_method: 'platt' or 'isotonic'
        """
        self.config = config or LGBMConfig()
        self.calibration_method = calibration_method
        
        self.model: Optional[lgb.LGBMClassifier] = None
        self.calibrator: Optional[PlattCalibrator | IsotonicCalibrator] = None
        self._feature_names: List[str] = []
        self._is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibration_split: float = 0.2,
        early_stopping_rounds: Optional[int] = 20,
        verbose: bool = False
    ) -> 'LightGBMPredictor':
        """
        Fit the LightGBM model with probability calibration.
        
        Args:
            X: Feature DataFrame
            y: Binary target (1 = home win, 0 = away win)
            calibration_split: Fraction for calibration set
            early_stopping_rounds: Early stopping patience
            verbose: Print training progress
        
        Returns:
            Self for method chaining
        """
        self._feature_names = list(X.columns)
        
        # Split for calibration
        split_idx = int(len(X) * (1 - calibration_split))
        X_train, X_cal = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_cal = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Create model
        params = self.config.to_dict()
        self.model = lgb.LGBMClassifier(**params)
        
        # Identify categorical features
        cat_features = [
            col for col in X.columns 
            if col in self.config.categorical_features or X[col].dtype == 'category'
        ]
        
        # Fit with early stopping
        if early_stopping_rounds:
            callbacks = [
                lgb.early_stopping(early_stopping_rounds, verbose=verbose),
                lgb.log_evaluation(period=50 if verbose else 0)
            ]
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_cal, y_cal)],
                callbacks=callbacks,
                categorical_feature=cat_features if cat_features else 'auto'
            )
        else:
            self.model.fit(
                X_train, y_train,
                categorical_feature=cat_features if cat_features else 'auto'
            )
        
        # Calibrate probabilities
        raw_probs = self.model.predict_proba(X_cal)[:, 1]
        
        if self.calibration_method == 'platt':
            self.calibrator = PlattCalibrator()
        else:
            self.calibrator = IsotonicCalibrator()
        
        self.calibrator.fit(raw_probs, y_cal.values)
        
        self._is_fitted = True
        logger.info(
            f"LightGBM fitted: {self.model.n_estimators_} trees, "
            f"{len(self._feature_names)} features"
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
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'gain', 'split', or 'cover'
        
        Returns:
            DataFrame with features and importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        importance = self.model.booster_.feature_importance(importance_type)
        
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
            temp_model = LightGBMPredictor(config=self.config)
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

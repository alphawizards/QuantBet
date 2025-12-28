"""
Ensemble Model for NBL/WNBL Predictions.

Combines multiple prediction models using optimized weights
for more robust and accurate predictions.

Ensemble Methods:
    - Simple Average: Equal weights
    - Weighted Average: Optimized via cross-validation
    - Stacking: Meta-learner on base model predictions

Implements Betfair Rule #4: Don't Overfit - uses out-of-sample
weight optimization to prevent overfitting to training data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class BaseModel(Protocol):
    """Protocol for ensemble-compatible models."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""
    home_win_prob: float
    model_predictions: Dict[str, float]
    weights: Dict[str, float]
    confidence: float = 0.0
    
    @property
    def away_win_prob(self) -> float:
        return 1.0 - self.home_win_prob
    
    @property
    def prediction_std(self) -> float:
        """Standard deviation of model predictions (disagreement)."""
        probs = list(self.model_predictions.values())
        return float(np.std(probs))
    
    def __str__(self) -> str:
        lines = [f"Ensemble P(home): {self.home_win_prob:.1%}"]
        for name, prob in self.model_predictions.items():
            weight = self.weights.get(name, 0)
            lines.append(f"  {name}: {prob:.1%} (w={weight:.2f})")
        return "\n".join(lines)


class EnsemblePredictor:
    """
    Weighted ensemble of multiple prediction models.
    
    Combines predictions from:
        - XGBoost classifier (feature-based)
        - ELO rating system (team strength)
        - Market implied probabilities (wisdom of crowds)
    
    Weight Optimization:
        Uses Brier score minimization on validation set to find
        optimal weights. Implements SKILL.md pattern for model
        evaluation and selection.
    
    Example:
        >>> ensemble = EnsemblePredictor()
        >>> ensemble.add_model('xgboost', xgb_model)
        >>> ensemble.add_model('elo', elo_model)
        >>> ensemble.fit(X_train, y_train)
        >>> probs = ensemble.predict_proba(X_test)
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        optimize_weights: bool = True,
        min_weight: float = 0.0,
        calibration_split: float = 0.2
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            weights: Manual weights per model (must sum to 1)
            optimize_weights: If True, optimize weights on validation set
            min_weight: Minimum weight per model
            calibration_split: Fraction of data for weight optimization
        """
        self._models: Dict[str, BaseModel] = {}
        self._weights: Dict[str, float] = weights or {}
        self._optimize_weights = optimize_weights
        self._min_weight = min_weight
        self._calibration_split = calibration_split
        self._is_fitted = False
    
    def add_model(
        self,
        name: str,
        model: BaseModel,
        weight: Optional[float] = None
    ) -> "EnsemblePredictor":
        """
        Add a model to the ensemble.
        
        Args:
            name: Unique identifier for model
            model: Model with fit/predict_proba interface
            weight: Optional manual weight
        
        Returns:
            Self for chaining
        """
        self._models[name] = model
        if weight is not None:
            self._weights[name] = weight
        
        logger.info(f"Added model to ensemble: {name}")
        return self
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self._models:
            del self._models[name]
            if name in self._weights:
                del self._weights[name]
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_base_models: bool = True
    ) -> "EnsemblePredictor":
        """
        Fit ensemble and optionally base models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            fit_base_models: If True, fit each base model on training data
        
        Returns:
            Self for chaining
        """
        if len(self._models) == 0:
            raise ValueError("No models added to ensemble")
        
        # Split for weight optimization
        n = len(X)
        split_idx = int(n * (1 - self._calibration_split))
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        # Fit base models
        if fit_base_models:
            for name, model in self._models.items():
                logger.info(f"Fitting base model: {name}")
                model.fit(X_train, y_train)
        
        # Optimize weights
        if self._optimize_weights and len(X_val) > 10:
            self._weights = self._optimize_ensemble_weights(X_val, y_val)
        elif not self._weights:
            # Equal weights if not specified
            n_models = len(self._models)
            self._weights = {name: 1.0 / n_models for name in self._models}
        
        # Normalize weights
        total = sum(self._weights.values())
        self._weights = {k: v / total for k, v in self._weights.items()}
        
        self._is_fitted = True
        logger.info(f"Ensemble fitted with weights: {self._weights}")
        
        return self
    
    def _optimize_ensemble_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        """
        Optimize weights to minimize Brier score.
        
        Uses scipy.optimize.minimize with constraints that
        weights are positive and sum to 1.
        """
        # Get predictions from each model
        model_names = list(self._models.keys())
        predictions = np.array([
            self._models[name].predict_proba(X_val)
            for name in model_names
        ])  # Shape: (n_models, n_samples)
        
        y_true = y_val.values
        
        def brier_score(weights: np.ndarray) -> float:
            """Objective: weighted Brier score."""
            weights = weights / weights.sum()  # Normalize
            ensemble_probs = np.average(predictions, axis=0, weights=weights)
            return np.mean((ensemble_probs - y_true) ** 2)
        
        # Initial weights (equal)
        n_models = len(model_names)
        x0 = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        
        # Bounds: weights between min_weight and 1
        bounds = [(self._min_weight, 1.0)] * n_models
        
        # Optimize
        result = minimize(
            brier_score,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = result.x
        else:
            logger.warning("Weight optimization failed, using equal weights")
            optimized_weights = x0
        
        return {name: float(w) for name, w in zip(model_names, optimized_weights)}
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        return_individual: bool = False
    ) -> np.ndarray:
        """
        Predict home win probabilities.
        
        Args:
            X: Feature DataFrame
            return_individual: If True, return dict with individual predictions
        
        Returns:
            Array of ensemble probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before predicting")
        
        # Get predictions from each model
        predictions = {}
        for name, model in self._models.items():
            preds = model.predict_proba(X)
            # Handle sklearn-style 2D output
            if len(preds.shape) > 1:
                preds = preds[:, 1]
            predictions[name] = preds
        
        # Weighted average
        ensemble_probs = np.zeros(len(X))
        for name, preds in predictions.items():
            weight = self._weights.get(name, 0)
            ensemble_probs += weight * preds
        
        if return_individual:
            return ensemble_probs, predictions
        
        return ensemble_probs
    
    def predict_game(
        self,
        X: pd.DataFrame
    ) -> List[EnsemblePrediction]:
        """
        Generate detailed ensemble predictions.
        
        Returns EnsemblePrediction objects with individual model
        predictions, weights, and confidence scores.
        """
        ensemble_probs, individual = self.predict_proba(X, return_individual=True)
        
        results = []
        for i in range(len(X)):
            model_preds = {name: float(preds[i]) for name, preds in individual.items()}
            
            # Confidence based on model agreement
            std = np.std(list(model_preds.values()))
            confidence = max(0, 1 - 2 * std)  # Higher when models agree
            
            pred = EnsemblePrediction(
                home_win_prob=float(ensemble_probs[i]),
                model_predictions=model_preds,
                weights=self._weights.copy(),
                confidence=confidence
            )
            results.append(pred)
        
        return results
    
    @property
    def weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self._weights.copy()
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def model_names(self) -> List[str]:
        return list(self._models.keys())


class MarketImpliedPredictor:
    """
    Predictor based on market odds.
    
    Uses bookmaker odds to derive implied probabilities,
    serving as a "wisdom of the crowd" baseline.
    """
    
    def __init__(
        self,
        odds_column: str = 'home_odds',
        remove_vig: bool = True
    ):
        """
        Args:
            odds_column: Column containing decimal odds
            remove_vig: Whether to adjust for bookmaker margin
        """
        self.odds_column = odds_column
        self.remove_vig = remove_vig
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarketImpliedPredictor":
        """No fitting required - just validates odds column exists."""
        if self.odds_column not in X.columns:
            logger.warning(f"Odds column '{self.odds_column}' not found")
        self._is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Convert odds to implied probabilities."""
        if self.odds_column not in X.columns:
            # Return 0.5 if no odds available
            return np.full(len(X), 0.5)
        
        odds = X[self.odds_column].values
        
        # Implied probability = 1 / odds
        probs = 1.0 / np.clip(odds, 1.01, 100)  # Clip to avoid division by zero
        
        if self.remove_vig:
            # Simple vig removal: scale to 1.0
            # In reality, need away odds too for proper removal
            probs = np.clip(probs, 0.01, 0.99)
        
        return probs
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


def create_default_ensemble() -> EnsemblePredictor:
    """
    Create ensemble with default NBL models.
    
    Models included:
        - market: Market implied probabilities
    
    XGBoost and ELO should be added separately with trained instances.
    """
    ensemble = EnsemblePredictor()
    ensemble.add_model('market', MarketImpliedPredictor())
    return ensemble

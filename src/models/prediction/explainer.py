"""
SHAP Model Explainer for NBL Predictions.

Provides interpretable explanations for model predictions using SHAP
(SHapley Additive exPlanations). This helps understand:
    - Which features drive predictions
    - Why the model predicts a certain outcome
    - Feature importance across the dataset

SHAP values show the contribution of each feature to the difference
between the actual prediction and the average prediction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import logging

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


@dataclass
class PredictionExplanation:
    """Explanation for a single prediction."""
    game_id: str
    home_team: str
    away_team: str
    predicted_prob: float
    base_value: float
    shap_values: Dict[str, float]
    top_positive_features: List[tuple]
    top_negative_features: List[tuple]
    
    def summary(self) -> str:
        """Human-readable explanation summary."""
        lines = [
            f"Prediction: {self.home_team} vs {self.away_team}",
            f"Home Win Probability: {self.predicted_prob:.1%}",
            f"Base (average) probability: {self.base_value:.1%}",
            "",
            "Top factors FAVORING home win:"
        ]
        
        for feat, val in self.top_positive_features[:3]:
            lines.append(f"  + {feat}: +{val:.3f}")
        
        lines.append("")
        lines.append("Top factors AGAINST home win:")
        
        for feat, val in self.top_negative_features[:3]:
            lines.append(f"  - {feat}: {val:.3f}")
        
        return "\n".join(lines)


class ModelExplainer:
    """
    SHAP-based model explainer for tree-based models.
    
    Provides feature importance analysis and prediction explanations
    for XGBoost, LightGBM, and CatBoost models.
    
    Example:
        >>> from src.models.prediction.predictor import NBLPredictor
        >>> predictor = NBLPredictor()
        >>> predictor.fit(X_train, y_train)
        >>> 
        >>> explainer = ModelExplainer(predictor.model, X_train)
        >>> explanation = explainer.explain_prediction(X_test.iloc[0:1], "MEL", "SYD")
        >>> print(explanation.summary())
    """
    
    def __init__(
        self,
        model: Any,
        X_background: Optional[pd.DataFrame] = None,
        max_background_samples: int = 100
    ):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained model (XGBoost, LightGBM, or scikit-learn)
            X_background: Background data for SHAP calculations
            max_background_samples: Limit background samples for speed
        """
        self.model = model
        
        # Sample background data for efficiency
        if X_background is not None:
            if len(X_background) > max_background_samples:
                X_background = X_background.sample(
                    n=max_background_samples, 
                    random_state=42
                )
            self.background = X_background
        else:
            self.background = None
        
        # Create SHAP explainer
        self._create_explainer()
        
        self.feature_names: List[str] = []
        if X_background is not None:
            self.feature_names = list(X_background.columns)
    
    def _create_explainer(self):
        """Create the appropriate SHAP explainer for the model type."""
        try:
            # TreeExplainer is fastest for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = 'tree'
            logger.info("Created TreeExplainer for tree-based model")
        except Exception:
            # Fallback to model-agnostic KernelExplainer
            if self.background is not None:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(self.background, 50)
                )
                self.explainer_type = 'kernel'
                logger.info("Created KernelExplainer as fallback")
            else:
                raise ValueError(
                    "Background data required for non-tree models"
                )
    
    def compute_shap_values(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute SHAP values for given samples.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            SHAP values array (n_samples, n_features)
        """
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output (binary classification returns list)
        if isinstance(shap_values, list):
            # Return SHAP values for positive class (home win)
            return shap_values[1]
        
        return shap_values
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        home_team: str,
        away_team: str,
        game_id: str = ""
    ) -> PredictionExplanation:
        """
        Explain a single game prediction.
        
        Args:
            X: Feature DataFrame for single game (1 row)
            home_team: Home team name/code
            away_team: Away team name/code
            game_id: Optional game identifier
        
        Returns:
            PredictionExplanation with feature contributions
        """
        if len(X) != 1:
            raise ValueError("Expected single row for explanation")
        
        # Get prediction
        probs = self.model.predict_proba(X)
        if probs.ndim > 1:
            predicted_prob = probs[0, 1]
        else:
            predicted_prob = probs[0]
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X)[0]
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5
        
        # Create feature contributions dict
        feature_names = list(X.columns) if hasattr(X, 'columns') else self.feature_names
        contributions = dict(zip(feature_names, shap_values))
        
        # Sort by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Separate positive and negative contributors
        positive = [(f, v) for f, v in sorted_features if v > 0]
        negative = [(f, v) for f, v in sorted_features if v < 0]
        
        return PredictionExplanation(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            predicted_prob=float(predicted_prob),
            base_value=float(base_value),
            shap_values=contributions,
            top_positive_features=positive,
            top_negative_features=negative
        )
    
    def get_feature_importance(
        self,
        X: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate global feature importance using mean |SHAP values|.
        
        Args:
            X: Data to calculate importance on. Uses background if None.
        
        Returns:
            DataFrame with features and importance scores
        """
        data = X if X is not None else self.background
        
        if data is None:
            raise ValueError("No data provided for importance calculation")
        
        shap_values = self.compute_shap_values(data)
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        return pd.DataFrame({
            'feature': data.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        plot_type: str = 'bar'
    ):
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to explain
            max_display: Max features to show
            plot_type: 'bar', 'dot', or 'violin'
        """
        shap_values = self.compute_shap_values(X)
        
        if plot_type == 'bar':
            shap.summary_plot(
                shap_values, X, 
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values, X,
                max_display=max_display,
                show=False
            )
    
    def plot_force(
        self,
        X: pd.DataFrame,
        index: int = 0
    ):
        """
        Create force plot for a single prediction.
        
        Args:
            X: Feature DataFrame
            index: Row index to explain
        """
        shap_values = self.compute_shap_values(X.iloc[[index]])
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        return shap.force_plot(
            base_value,
            shap_values[0],
            X.iloc[index]
        )


def explain_game_prediction(
    model: Any,
    features: pd.DataFrame,
    home_team: str,
    away_team: str,
    background_data: Optional[pd.DataFrame] = None
) -> PredictionExplanation:
    """
    Convenience function to explain a single game prediction.
    
    Args:
        model: Trained model
        features: Single row of game features
        home_team: Home team name
        away_team: Away team name
        background_data: Optional background for SHAP
    
    Returns:
        PredictionExplanation
    """
    explainer = ModelExplainer(model, background_data)
    return explainer.explain_prediction(features, home_team, away_team)

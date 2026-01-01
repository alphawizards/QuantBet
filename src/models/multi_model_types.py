# New Pydantic models for multi-model predictions
from pydantic import BaseModel
from typing import List

class ModelPrediction(BaseModel):
    """Prediction from a single model."""
    model_name: str
    predicted_home_prob: float
    recommended_bet: str  # "BET_HOME", "BET_AWAY", "SKIP"
    kelly_stake_pct: float
    edge: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"

class MultiModelGame(BaseModel):
    """Game with predictions from multiple models."""
    event_id: str
    home_team: str
    away_team: str
    commence_time: str
    home_odds: float
    away_odds: float
    best_bookmaker: str
    model_predictions: List[ModelPrediction]

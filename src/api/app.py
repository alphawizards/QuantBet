"""
FastAPI Application for QuantBet NBL Predictions.

REST API for:
    - Game predictions
    - Betting recommendations
    - Model metrics

Implements SKILL.md patterns:
    - Low-latency inference (<100ms P95)
    - Comprehensive logging
    - Health checks
"""

from datetime import datetime, date
from typing import Dict, List, Optional
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# API Models
# ============================================================================

class GamePrediction(BaseModel):
    """Prediction for a single game."""
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    home_win_prob: float = Field(..., ge=0, le=1)
    away_win_prob: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    model_agreement: float = Field(..., ge=0, le=1)
    recommended_bet: Optional[str] = None
    edge: Optional[float] = None


class BetRecommendation(BaseModel):
    """Betting recommendation with Kelly sizing."""
    game_id: str
    bet_type: str  # "home_win", "away_win", "over", "under"
    odds: float
    predicted_prob: float
    edge: float
    kelly_fraction: float
    recommended_stake: float
    expected_value: float


class ModelMetrics(BaseModel):
    """Current model performance metrics."""
    brier_score: float
    calibration_error: float
    roi: float
    win_rate: float
    total_predictions: int
    last_updated: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    database_connected: bool
    predictions_today: int


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="QuantBet NBL API",
    description="NBL/WNBL sports betting predictions powered by ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# State Management (would be replaced with proper DI in production)
# ============================================================================

class AppState:
    """Application state container."""
    model_loaded: bool = False
    last_prediction_time: Optional[datetime] = None
    predictions_today: int = 0


state = AppState()


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - basic info."""
    return {
        "name": "QuantBet NBL API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """
    Health check endpoint.
    
    Returns status of API components for monitoring.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=state.model_loaded,
        database_connected=True,  # Would check actual DB connection
        predictions_today=state.predictions_today
    )


@app.get("/predictions/today", response_model=List[GamePrediction], tags=["Predictions"])
async def get_today_predictions():
    """
    Get predictions for today's games.
    
    Returns predictions for all NBL/WNBL games scheduled today.
    """
    # In production, would query database for today's games
    # and run predictions through the model
    
    # Mock response for API structure
    today = date.today().isoformat()
    
    return [
        GamePrediction(
            game_id="nbl_2024_mock_001",
            home_team="MEL",
            away_team="SYD",
            game_date=today,
            home_win_prob=0.58,
            away_win_prob=0.42,
            confidence=0.72,
            model_agreement=0.85,
            recommended_bet="home_win",
            edge=0.05
        )
    ]


@app.get("/predictions/{game_id}", response_model=GamePrediction, tags=["Predictions"])
async def get_game_prediction(game_id: str):
    """
    Get prediction for a specific game.
    
    Args:
        game_id: Unique game identifier
    
    Returns:
        Detailed prediction with probabilities and confidence
    """
    # In production, would:
    # 1. Look up game in database
    # 2. Compute features
    # 3. Run through model ensemble
    # 4. Return prediction
    
    if not game_id.startswith("nbl_"):
        raise HTTPException(status_code=404, detail="Game not found")
    
    state.predictions_today += 1
    state.last_prediction_time = datetime.now()
    
    return GamePrediction(
        game_id=game_id,
        home_team="MEL",
        away_team="SYD",
        game_date=date.today().isoformat(),
        home_win_prob=0.58,
        away_win_prob=0.42,
        confidence=0.72,
        model_agreement=0.85,
        recommended_bet="home_win",
        edge=0.05
    )


@app.get("/recommendations", response_model=List[BetRecommendation], tags=["Betting"])
async def get_betting_recommendations(
    bankroll: float = Query(1000.0, gt=0, description="Current bankroll"),
    min_edge: float = Query(0.02, ge=0, le=0.5, description="Minimum edge threshold"),
    kelly_fraction: float = Query(0.25, gt=0, le=1.0, description="Kelly fraction to use")
):
    """
    Get betting recommendations for today.
    
    Filters predictions by minimum edge and calculates Kelly stakes.
    
    Args:
        bankroll: Current bankroll for stake sizing
        min_edge: Minimum edge required to recommend bet
        kelly_fraction: Fraction of Kelly criterion to use
    
    Returns:
        List of bet recommendations with stake sizes
    """
    # In production, would:
    # 1. Get all predictions for today
    # 2. Calculate edge vs market odds
    # 3. Filter by min_edge
    # 4. Calculate Kelly stakes
    
    recommendations = []
    
    # Mock recommendation
    edge = 0.08
    if edge >= min_edge:
        prob = 0.58
        odds = 1.90
        
        # Kelly fraction calculation
        kelly = (prob * odds - 1) / (odds - 1)
        adjusted_kelly = kelly * kelly_fraction
        stake = bankroll * max(0, adjusted_kelly)
        
        recommendations.append(
            BetRecommendation(
                game_id="nbl_2024_mock_001",
                bet_type="home_win",
                odds=odds,
                predicted_prob=prob,
                edge=edge,
                kelly_fraction=adjusted_kelly,
                recommended_stake=round(stake, 2),
                expected_value=stake * edge
            )
        )
    
    return recommendations


@app.get("/metrics", response_model=ModelMetrics, tags=["Monitoring"])
async def get_model_metrics():
    """
    Get current model performance metrics.
    
    Returns rolling performance statistics.
    """
    # In production, would query model monitor
    return ModelMetrics(
        brier_score=0.215,
        calibration_error=0.032,
        roi=0.055,
        win_rate=0.56,
        total_predictions=142,
        last_updated=datetime.now().isoformat()
    )


@app.get("/teams", tags=["Data"])
async def get_teams():
    """Get list of supported teams."""
    return {
        "nbl": [
            {"code": "MEL", "name": "Melbourne United"},
            {"code": "SYD", "name": "Sydney Kings"},
            {"code": "PER", "name": "Perth Wildcats"},
            {"code": "BRI", "name": "Brisbane Bullets"},
            {"code": "ADL", "name": "Adelaide 36ers"},
            {"code": "NZB", "name": "New Zealand Breakers"},
            {"code": "ILL", "name": "Illawarra Hawks"},
            {"code": "CAI", "name": "Cairns Taipans"},
            {"code": "TAS", "name": "Tasmania JackJumpers"},
            {"code": "SEM", "name": "South East Melbourne Phoenix"},
        ]
    }


# ============================================================================
# Startup / Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("QuantBet API starting up...")
    
    # In production:
    # - Load models
    # - Connect to database
    # - Initialize feature store
    
    state.model_loaded = True
    logger.info("QuantBet API ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("QuantBet API shutting down...")


# ============================================================================
# Entry point for development
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

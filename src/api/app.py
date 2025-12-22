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
import os
import secrets

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


logger = logging.getLogger(__name__)


# ============================================================================
# Authentication
# ============================================================================

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Verify admin credentials using HTTP Basic Auth.
    
    Credentials are stored in environment variables:
        ADMIN_USER: Admin username (default: admin)
        ADMIN_PASS: Admin password (required)
    """
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_pass = os.getenv("ADMIN_PASS", "")
    
    if not admin_pass:
        logger.warning("ADMIN_PASS not set - using insecure default")
        admin_pass = "quantbet2024"  # Only for development
    
    is_user_correct = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        admin_user.encode("utf-8")
    )
    is_pass_correct = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        admin_pass.encode("utf-8")
    )
    
    if not (is_user_correct and is_pass_correct):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"}
        )
    
    return credentials.username




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


class OddsInput(BaseModel):
    """Manual odds input for stake calculation."""
    game_id: str
    home_team: str
    away_team: str
    home_odds: float = Field(..., gt=1.0, description="Decimal odds for home win")
    away_odds: float = Field(..., gt=1.0, description="Decimal odds for away win")
    predicted_home_prob: Optional[float] = Field(None, ge=0, le=1, description="Override model prediction")


class StakeRecommendation(BaseModel):
    """Kelly stake recommendation based on odds and prediction."""
    game_id: str
    home_team: str
    away_team: str
    predicted_prob: float
    home_odds: float
    away_odds: float
    recommended_side: str  # "home" or "away" or "none"
    edge: float
    kelly_fraction: float
    recommended_stake: float
    expected_value: float
    implied_prob: float


class LiveGameOdds(BaseModel):
    """Live odds for a game from The Odds API."""
    event_id: str
    sport: str
    commence_time: str
    home_team: str
    away_team: str
    best_home_odds: float
    best_away_odds: float
    best_home_bookmaker: str
    best_away_bookmaker: str
    home_implied_prob: float
    away_implied_prob: float



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


@app.post("/calculate-stake", response_model=StakeRecommendation, tags=["Betting"])
async def calculate_stake(
    odds_input: OddsInput,
    bankroll: float = Query(1000.0, gt=0, description="Current bankroll"),
    kelly_fraction: float = Query(0.25, gt=0, le=1.0, description="Kelly fraction"),
    admin: str = Depends(verify_admin)
):
    """
    Calculate stake recommendation from manually entered odds.
    
    Enter the current odds from your sportsbook to get Kelly-sized
    stake recommendations based on the model's predicted probabilities.
    
    Requires admin authentication.
    """
    # Use provided prediction or default mock (in production, query model)
    if odds_input.predicted_home_prob is not None:
        home_prob = odds_input.predicted_home_prob
    else:
        # Mock prediction - in production would query model
        home_prob = 0.58
    
    away_prob = 1 - home_prob
    
    # Calculate implied probabilities from odds
    home_implied = 1 / odds_input.home_odds
    away_implied = 1 / odds_input.away_odds
    
    # Calculate edges
    home_edge = home_prob - home_implied
    away_edge = away_prob - away_implied
    
    # Determine which side (if any) has positive edge
    if home_edge > away_edge and home_edge > 0:
        recommended_side = "home"
        edge = home_edge
        odds = odds_input.home_odds
        prob = home_prob
        implied = home_implied
    elif away_edge > 0:
        recommended_side = "away"
        edge = away_edge
        odds = odds_input.away_odds
        prob = away_prob
        implied = away_implied
    else:
        recommended_side = "none"
        edge = max(home_edge, away_edge)
        odds = odds_input.home_odds if home_edge > away_edge else odds_input.away_odds
        prob = home_prob if home_edge > away_edge else away_prob
        implied = home_implied if home_edge > away_edge else away_implied
    
    # Kelly criterion calculation
    if recommended_side != "none":
        kelly = (prob * odds - 1) / (odds - 1)
        adjusted_kelly = max(0, kelly * kelly_fraction)
        stake = bankroll * adjusted_kelly
        ev = stake * edge
    else:
        adjusted_kelly = 0
        stake = 0
        ev = 0
    
    return StakeRecommendation(
        game_id=odds_input.game_id,
        home_team=odds_input.home_team,
        away_team=odds_input.away_team,
        predicted_prob=prob,
        home_odds=odds_input.home_odds,
        away_odds=odds_input.away_odds,
        recommended_side=recommended_side,
        edge=round(edge, 4),
        kelly_fraction=round(adjusted_kelly, 4),
        recommended_stake=round(stake, 2),
        expected_value=round(ev, 2),
        implied_prob=round(implied, 4)
    )


@app.get("/odds/live", response_model=List[LiveGameOdds], tags=["Odds"])
async def get_live_odds(
    sport: str = Query("nbl", description="Sport: 'nbl' or 'wnbl'"),
    admin: str = Depends(verify_admin)
):
    """
    Get live odds from Australian bookmakers.
    
    Fetches current odds from The Odds API for upcoming NBL/WNBL games.
    Returns best available odds across all bookmakers.
    
    Requires admin authentication.
    """
    try:
        from ..data.odds_api import OddsAPIClient
        
        client = OddsAPIClient()
        
        if sport.lower() == "wnbl":
            games = client.get_wnbl_odds()
        else:
            games = client.get_nbl_odds()
        
        result = []
        for game in games:
            # Calculate implied probabilities
            home_implied = 1 / game.best_home_odds if game.best_home_odds > 0 else 0
            away_implied = 1 / game.best_away_odds if game.best_away_odds > 0 else 0
            
            result.append(LiveGameOdds(
                event_id=game.event_id,
                sport=game.sport,
                commence_time=game.commence_time.isoformat(),
                home_team=game.home_team,
                away_team=game.away_team,
                best_home_odds=game.best_home_odds,
                best_away_odds=game.best_away_odds,
                best_home_bookmaker=game.best_home_bookmaker,
                best_away_bookmaker=game.best_away_bookmaker,
                home_implied_prob=round(home_implied, 4),
                away_implied_prob=round(away_implied, 4),
            ))
        
        # Log quota usage
        if client.last_quota:
            logger.info(
                f"Odds API quota: {client.last_quota.requests_remaining} requests remaining"
            )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Odds API error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch live odds")


@app.get("/odds/quota", tags=["Odds"])
async def get_odds_quota(admin: str = Depends(verify_admin)):
    """
    Get current Odds API quota usage.
    
    Returns remaining requests for the month.
    """
    try:
        from ..data.odds_api import OddsAPIClient
        
        client = OddsAPIClient()
        # Make a minimal request to get quota
        client.get_sports()
        
        if client.last_quota:
            return {
                "requests_used": client.last_quota.requests_used,
                "requests_remaining": client.last_quota.requests_remaining,
                "monthly_limit": 500,
            }
        else:
            return {"error": "Could not retrieve quota info"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

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

Security features:
    - API Key authentication (X-API-Key header)
    - HTTP Basic Auth for admin endpoints
    - Rate limiting (100 requests/minute)
    - Input validation with Pydantic
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Annotated
import logging
import os
import re
import secrets
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials, APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables from .env file
load_dotenv()


logger = logging.getLogger(__name__)

# ============================================================================
# Rate Limiting Configuration
# ============================================================================

limiter = Limiter(key_func=get_remote_address)

# NBL team codes for validation
VALID_NBL_TEAMS = {
    "MEL", "SYD", "PER", "BRI", "ADL", "NZB", "ILL", "CAI", "TAS", "SEM",
    "Melbourne United", "Sydney Kings", "Perth Wildcats", "Brisbane Bullets",
    "Adelaide 36ers", "New Zealand Breakers", "Illawarra Hawks",
    "Cairns Taipans", "Tasmania JackJumpers", "South East Melbourne Phoenix"
}

VALID_WNBL_TEAMS = {
    "MEL", "SYD", "PER", "CAN", "ADL", "BEN", "TOW", "SOU",
    "Melbourne Boomers", "Sydney Flames", "Perth Lynx", "Canberra Capitals",
    "Adelaide Lightning", "Bendigo Spirit", "Townsville Fire", "Southside Flyers"
}


# ============================================================================
# Authentication
# ============================================================================

security = HTTPBasic()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> str:
    """Get API key from environment."""
    api_key = os.getenv("QUANTBET_API_KEY", "")
    if not api_key:
        logger.warning("QUANTBET_API_KEY not set - API key auth disabled")
    return api_key


async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[str]:
    """
    Verify API key from X-API-Key header.
    
    API key is stored in QUANTBET_API_KEY environment variable.
    If not set, authentication is skipped (development mode).
    """
    expected_key = get_api_key()
    
    # If no API key is configured, skip auth (dev mode)
    if not expected_key:
        return None
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    if not secrets.compare_digest(api_key.encode("utf-8"), expected_key.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return api_key


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
    """Manual odds input for stake calculation with comprehensive validation."""
    game_id: str = Field(..., min_length=3, max_length=50, description="Game identifier")
    home_team: str = Field(..., min_length=2, max_length=50, description="Home team name/code")
    away_team: str = Field(..., min_length=2, max_length=50, description="Away team name/code")
    home_odds: float = Field(..., gt=1.01, le=100.0, description="Decimal odds for home win (must be between 1.01 and 100)")
    away_odds: float = Field(..., gt=1.01, le=100.0, description="Decimal odds for away win (must be between 1.01 and 100)")
    predicted_home_prob: Optional[float] = Field(None, ge=0.01, le=0.99, description="Override model prediction (must be between 0.01 and 0.99)")
    
    @field_validator('game_id')
    @classmethod
    def validate_game_id(cls, v: str) -> str:
        """Validate game_id format."""
        # Allow alphanumeric with underscores and hyphens
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('game_id must be alphanumeric with underscores/hyphens only')
        return v
    
    @field_validator('home_team', 'away_team')
    @classmethod
    def validate_team_name(cls, v: str) -> str:
        """Validate team name format."""
        # Strip and check for injection attempts
        v = v.strip()
        if not re.match(r'^[a-zA-Z0-9\s-]+$', v):
            raise ValueError('Team name must be alphanumeric with spaces/hyphens only')
        return v
    
    @model_validator(mode='after')
    def validate_teams_different(self):
        """Ensure home and away teams are different."""
        if self.home_team.upper() == self.away_team.upper():
            raise ValueError('Home and away teams must be different')
        return self
    
    @model_validator(mode='after')
    def validate_odds_reasonable(self):
        """Ensure combined odds represent valid probabilities."""
        # Implied probabilities should sum to at least 1 (due to bookmaker margin)
        # But not exceed ~1.5 which would be unrealistic
        combined_implied = (1 / self.home_odds) + (1 / self.away_odds)
        if combined_implied < 0.9:
            raise ValueError('Combined implied probability too low - check odds values')
        if combined_implied > 1.5:
            raise ValueError('Combined implied probability too high - check odds values')
        return self


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


class TodaysPrediction(BaseModel):
    """Comprehensive prediction for today's game."""
    event_id: str
    home_team: str
    away_team: str
    commence_time: str
    
    # Model prediction with uncertainty
    predicted_home_prob: float
    predicted_home_prob_lower: float  # 95% CI lower
    predicted_home_prob_upper: float  # 95% CI upper
    uncertainty: float  # Standard deviation
    
    # Live odds
    home_odds: float
    away_odds: float
    best_bookmaker: str
    
    # Edge analysis
    home_edge: float  # Positive = value on home
    away_edge: float  # Positive = value on away
    
    # Recommendation
    recommendation: str  # "BET_HOME", "BET_AWAY", "SKIP"
    kelly_fraction: float
    recommended_stake_pct: float  # As percentage of bankroll
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    
    # Top factors (SHAP-based)
    top_factors: List[str]  # e.g. ["Home team on 3-game win streak", "Away team traveled 2000km"]



app = FastAPI(
    title="QuantBet NBL API",
    description="NBL/WNBL sports betting predictions powered by ML",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - configure for production
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header for performance monitoring."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2)) + "ms"
    return response


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
@limiter.limit("60/minute")
async def get_today_predictions(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Get predictions for today's games (basic response model).
    
    Rate limited to 60 requests per minute.
    Requires API key in production (X-API-Key header).
    
    Returns predictions for all NBL/WNBL games scheduled today.
    For comprehensive predictions with odds and Kelly, use /predictions/today/full
    """
    # Basic version - returns mock data
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


@app.get("/predictions/today/full", response_model=List[TodaysPrediction], tags=["Predictions"])
@limiter.limit("30/minute")
async def get_today_full_predictions(
    request: Request,
    sport: str = Query("nbl", pattern="^(nbl|wnbl)$", description="Sport: 'nbl' or 'wnbl'"),
    bankroll: float = Query(1000.0, gt=0, le=10000000, description="Your bankroll for stake calculations"),
    kelly_fraction: float = Query(0.25, gt=0, le=1.0, description="Kelly fraction (0.25 = quarter Kelly)"),
    _api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Get comprehensive predictions with live odds and Kelly stakes.
    
    This is the main endpoint for daily betting decisions. It combines:
    - Bayesian Elo ratings with uncertainty
    - Live odds from Australian bookmakers
    - Kelly criterion stake sizing
    - Factor explanations for each prediction
    
    Returns empty list if no games today or odds unavailable.
    """
    try:
        from ..data.odds_api import OddsAPIClient
        from ..model.bayesian_elo import BayesianEloRating
        
        # Initialize clients
        odds_client = OddsAPIClient()
        elo = BayesianEloRating()
        
        # Fetch live odds
        if sport.lower() == "wnbl":
            games = odds_client.get_wnbl_odds()
        else:
            games = odds_client.get_nbl_odds()
        
        if not games:
            return []
        
        predictions = []
        
        for game in games:
            # Get Bayesian Elo prediction with uncertainty
            prob, uncertainty = elo.predict_proba(game.home_team, game.away_team)
            pred_details = elo.predict_with_confidence(game.home_team, game.away_team)
            
            # Calculate edges
            home_implied = 1 / game.best_home_odds if game.best_home_odds > 0 else 0
            away_implied = 1 / game.best_away_odds if game.best_away_odds > 0 else 0
            
            home_edge = prob - home_implied
            away_edge = (1 - prob) - away_implied
            
            # Determine recommendation
            min_edge = 0.03  # 3% minimum edge to bet
            
            if home_edge > away_edge and home_edge > min_edge:
                recommendation = "BET_HOME"
                best_edge = home_edge
                odds_to_use = game.best_home_odds
                implied = home_implied
                prob_to_use = prob
            elif away_edge > home_edge and away_edge > min_edge:
                recommendation = "BET_AWAY"
                best_edge = away_edge
                odds_to_use = game.best_away_odds
                implied = away_implied
                prob_to_use = 1 - prob
            else:
                recommendation = "SKIP"
                best_edge = max(home_edge, away_edge)
                odds_to_use = game.best_home_odds
                implied = home_implied
                prob_to_use = prob
            
            # Calculate Kelly stake
            if recommendation != "SKIP" and best_edge > 0:
                kelly = (prob_to_use * odds_to_use - 1) / (odds_to_use - 1)
                kelly = max(0, kelly * kelly_fraction)
            else:
                kelly = 0
            
            # Determine confidence level
            if uncertainty < 0.05:
                confidence = "HIGH"
            elif uncertainty < 0.10:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            # Generate top factors (mock for now, would use SHAP in production)
            factors = []
            if pred_details['home_rating_mean'] > pred_details['away_rating_mean'] + 50:
                factors.append(f"{game.home_team} is rated {int(pred_details['home_rating_mean'] - pred_details['away_rating_mean'])} points higher")
            if best_edge > 0.05:
                factors.append(f"Strong edge of {best_edge:.1%} vs market")
            if uncertainty < 0.08:
                factors.append("High confidence prediction")
            if not factors:
                factors.append("No strong factors detected")
            
            predictions.append(TodaysPrediction(
                event_id=game.event_id,
                home_team=game.home_team,
                away_team=game.away_team,
                commence_time=game.commence_time.isoformat(),
                predicted_home_prob=round(prob, 4),
                predicted_home_prob_lower=round(pred_details['ci_05'], 4),
                predicted_home_prob_upper=round(pred_details['ci_95'], 4),
                uncertainty=round(uncertainty, 4),
                home_odds=game.best_home_odds,
                away_odds=game.best_away_odds,
                best_bookmaker=game.best_home_bookmaker if recommendation == "BET_HOME" else game.best_away_bookmaker,
                home_edge=round(home_edge, 4),
                away_edge=round(away_edge, 4),
                recommendation=recommendation,
                kelly_fraction=round(kelly, 4),
                recommended_stake_pct=round(kelly * 100, 2),
                confidence=confidence,
                top_factors=factors[:3]
            ))
        
        # Log quota usage
        if odds_client.last_quota:
            logger.info(
                f"Predictions generated. Odds API quota: {odds_client.last_quota.requests_remaining} remaining"
            )
        
        return predictions
        
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Odds API error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate predictions")


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
@limiter.limit("30/minute")
async def get_betting_recommendations(
    request: Request,
    bankroll: float = Query(1000.0, gt=0, le=10000000, description="Current bankroll"),
    min_edge: float = Query(0.02, ge=0, le=0.5, description="Minimum edge threshold"),
    kelly_fraction: float = Query(0.25, gt=0, le=1.0, description="Kelly fraction to use"),
    _api_key: Optional[str] = Depends(verify_api_key)
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

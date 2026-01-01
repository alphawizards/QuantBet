"""
Analytics API Endpoints for QuantBet.

Provides calibration analysis, performance metrics, and model monitoring endpoints.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

from src.monitoring.prediction_logger import get_production_logger
from src.monitoring.calibration_analysis import get_calibration_metrics, CalibrationBin
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# Response Models
class CalibrationBinResponse(BaseModel):
    """Calibration bin data."""
    bin_range: tuple[float, float]
    predicted_prob: float
    observed_freq: float
    count: int
    brier_contribution: float


class CalibrationResponse(BaseModel):
    """Calibration analysis response."""
    brier_score: float
    calibration_slope: float
    calibration_in_large: float
    expected_calibration_error: float
    calibration_bins: List[Dict]
    sample_size: int
    period_days: int
    generated_at: str


class PerformanceSummaryResponse(BaseModel):
    """Performance summary response."""
    total_predictions: int
    predictions_with_outcomes: int
    win_rate: Optional[float]
    average_edge: Optional[float]
    brier_score: Optional[float]
    period_days: int


@router.get("/calibration", response_model=CalibrationResponse)
async def get_calibration(
    days: int = Query(30, description="Days of prediction history to analyze", ge=1, le=365)
):
    """
    Get calibration analysis for recent predictions.
    
    Calibration measures whether predicted probabilities match observed frequencies.
    A well-calibrated model predicting 60% should win 60% of the time.
    
    Args:
        days: Number of days to analyze (default 30)
    
    Returns:
        Calibration metrics including Brier score, calibration curve, and ECE
    
    Example:
        GET /analytics/calibration?days=30
        
        Response:
        {
            "brier_score": 0.21,
            "calibration_slope": 0.95,
            "expected_calibration_error": 0.03,
            "calibration_bins": [...],
            "sample_size": 145
        }
    """
    try:
        # Get predictions with outcomes
        prod_logger = get_production_logger()
        predictions = prod_logger.get_predictions_with_outcomes(days=days)
        
        if len(predictions) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions with outcomes found in last {days} days"
            )
        
        # Extract prediction probabilities and outcomes
        predicted_probs = [p['predicted_home_prob'] for p in predictions]
        actual_outcomes = [1 if p['home_won'] else 0 for p in predictions]
        
        # Calculate calibration metrics
        metrics = get_calibration_metrics(predicted_probs, actual_outcomes)
        
        # Convert CalibrationBin objects to dicts
        bins_dict = [
            {
                'bin_range': bin.bin_range,
                'predicted_prob': bin.predicted_prob,
                'observed_freq': bin.observed_freq,
                'count': bin.count,
                'brier_contribution': bin.brier_contribution
            }
            for bin in metrics['bins']
        ]
        
        return CalibrationResponse(
            brier_score=metrics['brier_score'],
            calibration_slope=metrics['calibration_slope'],
            calibration_in_large=metrics['calibration_in_large'],
            expected_calibration_error=metrics['expected_calibration_error'],
            calibration_bins=bins_dict,
            sample_size=metrics['sample_size'],
            period_days=days,
            generated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating calibration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating calibration metrics: {str(e)}"
        )


@router.get("/performance-summary", response_model=PerformanceSummaryResponse)
async def get_performance_summary(
    days: int = Query(7, description="Days to summarize", ge=1, le=365)
):
    """
    Get performance summary for recent predictions.
    
    Args:
        days: Number of days to summarize (default 7)
    
    Returns:
        Summary statistics including win rate, average edge, and sample size
    """
    try:
        prod_logger = get_production_logger()
        
        # Get all predictions
        all_predictions = prod_logger.get_recent_predictions(days=days)
        
        # Get predictions with outcomes
        predictions_with_outcomes = prod_logger.get_predictions_with_outcomes(days=days)
        
        # Calculate basic stats
        total_predictions = len(all_predictions)
        predictions_with_outcomes_count = len(predictions_with_outcomes)
        
        win_rate = None
        average_edge = None
        brier_score = None
        
        if predictions_with_outcomes_count > 0:
            # Win rate (how often did home team win when we predicted > 50%)
            wins = sum(1 for p in predictions_with_outcomes if p['home_won'])
            win_rate = wins / predictions_with_outcomes_count
            
            # Average edge
            edges = [p['edge'] for p in predictions_with_outcomes if 'edge' in p]
            if edges:
                average_edge = sum(edges) / len(edges)
            
            # Brier score
            predicted_probs = [p['predicted_home_prob'] for p in predictions_with_outcomes]
            actual_outcomes = [1 if p['home_won'] else 0 for p in predictions_with_outcomes]
            
            from src.monitoring.calibration_analysis import calculate_calibration
            brier, _ = calculate_calibration(predicted_probs, actual_outcomes)
            brier_score = brier
        
        return PerformanceSummaryResponse(
            total_predictions=total_predictions,
            predictions_with_outcomes=predictions_with_outcomes_count,
            win_rate=win_rate,
            average_edge=average_edge,
            brier_score=brier_score,
            period_days=days
        )
        
    except Exception as e:
        logger.error(f"Error generating performance summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating performance summary: {str(e)}"
        )


@router.get("/health")
async def analytics_health():
    """Health check for analytics endpoints."""
    return {
        "status": "healthy",
        "service": "analytics",
        "timestamp": datetime.now().isoformat()
    }

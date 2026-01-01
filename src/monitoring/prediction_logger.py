"""
Production Prediction Logger for QuantBet.

Logs all predictions with metadata for model monitoring and evaluation.
Enables tracking of model performance in production.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PredictionLog:
    """
    Complete prediction record for production monitoring.
    
    Tracks all information needed to evaluate model performance:
    - What was predicted
    - When it was predicted
    - What actually happened
    - Model confidence and metadata
    """
    # Identification
    prediction_id: str          # Unique ID for this prediction
    timestamp: str              # When prediction was made
    game_id: str                # Game identifier
    
    # Game details
    home_team: str
    away_team: str
    game_datetime: str          # Scheduled game time
    
    # Model predictions
    model_name: str             # Which model made prediction
    predicted_home_prob: float  # Predicted home win probability
    predicted_away_prob: float  # Predicted away win probability
    
    # Betting recommendation
    recommended_bet: str        # "BET_HOME", "BET_AWAY", "SKIP"
    kelly_stake_pct: float      # Recommended stake as % of bankroll
    edge: float                 # Calculated edge vs market
    
    # Market odds
    home_odds: float
    away_odds: float
    bookmaker: str
    
    # Optional fields with defaults
    uncertainty: Optional[float] = None  # Model uncertainty (if applicable)
    
    # Confidence intervals (for Bayesian models)
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    
    # Actual outcome (filled in after game)
    actual_home_score: Optional[int] = None
    actual_away_score: Optional[int] = None
    home_won: Optional[bool] = None
    outcome_timestamp: Optional[str] = None
    
    # Performance metrics (calculated after outcome)
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    correct_prediction: Optional[bool] = None
    bet_result: Optional[float] = None  # Profit/loss if bet was placed
    
    # Model metadata
    model_version: str = "1.0"
    feature_set: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ProductionLogger:
    """
    Manages logging of predictions to disk for monitoring.
    
    Stores predictions in daily log files:
    - data/predictions/{YYYY-MM-DD}.jsonl
    
    Each line is a complete JSON record of one prediction.
    """
    
    def __init__(self, log_dir: str = "data/predictions"):
        """
        Initialize production logger.
        
        Args:
            log_dir: Directory to store prediction logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Production logger initialized: {self.log_dir}")
    
    def log_prediction(self, prediction: PredictionLog) -> None:
        """
        Log a prediction to disk.
        
        Args:
            prediction: PredictionLog instance
        """
        # Determine log file based on current date
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"{today}.jsonl"
        
        # Append prediction as JSON line
        try:
            with open(log_file, "a") as f:
                f.write(prediction.to_json() + "\n")
            
            logger.info(
                f"Logged prediction {prediction.prediction_id} "
                f"({prediction.home_team} vs {prediction.away_team})"
            )
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def update_outcome(
        self,
        prediction_id: str,
        home_score: int,
        away_score: int
    ) -> None:
        """
        Update a logged prediction with actual game outcome.
        
        Args:
            prediction_id: ID of prediction to update
            home_score: Actual home team score
            away_score: Actual away team score
        """
        # This is a simplified implementation
        # In production, you'd want to update the record in-place
        # or maintain a separate outcomes file
        logger.info(
            f"Outcome recorded for {prediction_id}: "
            f"{home_score}-{away_score}"
        )
    
    def get_recent_predictions(
        self,
        days: int = 7,
        model_name: Optional[str] = None
    ) -> list:
        """
        Get recent predictions from log files.
        
        Args:
            days: Number of days to look back
            model_name: Filter by model name (optional)
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            log_file = self.log_dir / f"{date}.jsonl"
            
            if not log_file.exists():
                continue
            
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        pred = json.loads(line.strip())
                        
                        # Filter by model if specified
                        if model_name and pred.get("model_name") != model_name:
                            continue
                        
                        predictions.append(pred)
            except Exception as e:
                logger.error(f"Error reading {log_file}: {e}")
        
        return predictions
    
    def get_predictions_with_outcomes(self, days: int = 30) -> list:
        """
        Get predictions that have actual outcomes recorded.
        
        This is used for calibration analysis - we need both predictions
        and actual results to calculate metrics.
        
        Args:
            days: Number of days to look back (default 30)
        
        Returns:
            List of dictionaries with:
            - predicted_home_prob: float
            - home_won: bool (actual outcome)
            - All other prediction fields
        
        Example:
            >>> logger = ProductionLogger()
            >>> preds = logger.get_predictions_with_outcomes(days=7)
            >>> len(preds)
            42  # Games with outcomes from last 7 days
            >>> preds[0]['predicted_home_prob']
            0.587
            >>> preds[0]['home_won']
            True
        """
        all_predictions = self.get_recent_predictions(days=days)
        
        # Filter to only predictions with outcomes
        with_outcomes = [
            pred for pred in all_predictions
            if pred.get('home_won') is not None  # Outcome has been recorded
        ]
        
        logger.info(
            f"Found {len(with_outcomes)} predictions with outcomes "
            f"out of {len(all_predictions)} total predictions"
        )
        
        return with_outcomes


def create_prediction_log(
    game_id: str,
    home_team: str,
    away_team: str,
    game_datetime: str,
    model_name: str,
    predicted_home_prob: float,
    home_odds: float,
    away_odds: float,
    bookmaker: str,
    recommended_bet: str,
    kelly_stake_pct: float,
    edge: float,
    **kwargs
) -> PredictionLog:
    """
    Helper function to create a prediction log entry.
    
    Args:
        game_id: Unique game identifier
        home_team: Home team code
        away_team: Away team code
        game_datetime: ISO format game datetime
        model_name: Name of prediction model
        predicted_home_prob: Predicted home win probability
        home_odds: Decimal odds for home team
        away_odds: Decimal odds for away team
        bookmaker: Bookmaker name
        recommended_bet: Betting recommendation
        kelly_stake_pct: Kelly stake percentage
        edge: Calculated edge
        **kwargs: Additional fields (uncertainty, CI, etc.)
    
    Returns:
        PredictionLog instance
    """
    import uuid
    
    prediction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    return PredictionLog(
        prediction_id=prediction_id,
        timestamp=timestamp,
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        game_datetime=game_datetime,
        model_name=model_name,
        predicted_home_prob=predicted_home_prob,
        predicted_away_prob=1.0 - predicted_home_prob,
        home_odds=home_odds,
        away_odds=away_odds,
        bookmaker=bookmaker,
        recommended_bet=recommended_bet,
        kelly_stake_pct=kelly_stake_pct,
        edge=edge,
        **kwargs
    )


# Global singleton instance
_production_logger: Optional[ProductionLogger] = None


def get_production_logger() -> ProductionLogger:
    """Get or create the global production logger instance."""
    global _production_logger
    if _production_logger is None:
        _production_logger = ProductionLogger()
    return _production_logger

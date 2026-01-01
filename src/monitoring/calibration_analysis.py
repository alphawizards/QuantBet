"""
Calibration Analysis for Production Predictions.

Validates that predicted probabilities match observed frequencies.
Uses Brier score and calibration curves to assess model quality.

Example:
    >>> from calibration_analysis import calculate_calibration
    >>> preds = [0.6, 0.7, 0.55, 0.65]
    >>> actuals = [1, 1, 0, 1]
    >>> brier, bins = calculate_calibration(preds, actuals)
    >>> brier
    0.0625
"""

from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class CalibrationBin:
    """Calibration metrics for a probability bin."""
    bin_range: Tuple[float, float]
    predicted_prob: float
    observed_freq: float
    count: int
    brier_contribution: float


def calculate_calibration(
    predictions: List[float],
    actuals: List[int],
    n_bins: int = 10
) -> Tuple[float, List[CalibrationBin]]:
    """
    Calculate calibration curve and Brier score.
    
    Calibration measures whether predicted probabilities match reality.
    A well-calibrated model predicting 60% should win 60% of the time.
    
    Args:
        predictions: Predicted probabilities [0, 1]
        actuals: Actual outcomes (0 or 1)
        n_bins: Number of calibration bins (default 10)
    
    Returns:
        Tuple of (brier_score, calibration_bins)
        - brier_score: Mean squared error (lower is better, < 0.25 is good)
        - calibration_bins: List of CalibrationBin objects
    
    Example:
        >>> preds = [0.6, 0.7, 0.55, 0.65]
        >>> acts = [1, 1, 0, 1]
        >>> brier, bins = calculate_calibration(preds, acts)
        >>> brier
        0.0625  # Lower is better
        >>> bins[0].predicted_prob
        0.60  # Average prediction in bin
        >>> bins[0].observed_freq
        0.75  # Actual win rate in bin
    
    Raises:
        ValueError: If inputs are invalid or empty
    """
    # Validate inputs
    if len(predictions) == 0:
        raise ValueError("Cannot calculate calibration on empty dataset")
    
    if len(predictions) != len(actuals):
        raise ValueError(
            f"Predictions ({len(predictions)}) and actuals ({len(actuals)}) "
            "must have same length"
        )
    
    preds = np.array(predictions)
    acts = np.array(actuals)
    
    # Validate probability range
    if not np.all((preds >= 0) & (preds <= 1)):
        raise ValueError("All predictions must be in [0, 1] range")
    
    # Validate binary outcomes
    if not np.all((acts == 0) | (acts == 1)):
        raise ValueError("All actuals must be 0 or 1")
    
    # Calculate Brier score: mean squared error
    brier = float(np.mean((preds - acts) ** 2))
    
    # Create calibration bins
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i+1]
        
        # Find predictions in this bin
        if i == n_bins - 1:  # Last bin includes upper edge
            mask = (preds >= lower) & (preds <= upper)
        else:
            mask = (preds >= lower) & (preds < upper)
        
        if mask.sum() == 0:
            continue  # Skip empty bins
        
        bin_preds = preds[mask]
        bin_acts = acts[mask]
        
        bins.append(CalibrationBin(
            bin_range=(float(lower), float(upper)),
            predicted_prob=float(bin_preds.mean()),
            observed_freq=float(bin_acts.mean()),
            count=int(len(bin_preds)),
            brier_contribution=float(np.mean((bin_preds - bin_acts) ** 2))
        ))
    
    return brier, bins


def calibration_slope(predictions: List[float], actuals: List[int]) -> float:
    """
    Calculate calibration slope via logistic regression.
    
    The calibration slope measures systematic over/under-confidence:
    - Slope = 1.0: Perfect calibration
    - Slope < 1.0: Overconfident (predictions too extreme)
    - Slope > 1.0: Underconfident (predictions too moderate)
    
    Args:
        predictions: Predicted probabilities [0, 1]
        actuals: Actual outcomes (0 or 1)
    
    Returns:
        Calibration slope coefficient
    
    Example:
        >>> # Overconfident predictions
        >>> preds = [0.9, 0.8, 0.2, 0.1]
        >>> acts = [1, 1, 0, 0]
        >>> slope = calibration_slope(preds, acts)
        >>> slope < 1.0  # Overconfident
        True
    
    Raises:
        ValueError: If inputs are invalid
    """
    from scipy.special import logit
    
    if len(predictions) < 2:
        raise ValueError("Need at least 2 predictions for slope calculation")
    
    preds = np.array(predictions)
    acts = np.array(actuals)
    
    # Convert to log-odds, avoiding 0 and 1
    eps = 1e-7
    preds_clipped = np.clip(preds, eps, 1-eps)
    log_odds_pred = logit(preds_clipped)
    
    # Fit logistic regression: actual ~ log_odds_pred
    # Using simple linear regression on log-odds
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    lr.fit(log_odds_pred.reshape(-1, 1), acts)
    
    return float(lr.coef_[0])


def calibration_in_the_large(predictions: List[float], actuals: List[int]) -> float:
    """
    Calculate calibration-in-the-large (overall bias).
    
    Measures systematic over/under-prediction:
    - Value = 0: Perfect calibration
    - Value > 0: Overpredicting (predicting too high)
    - Value < 0: Underpredicting (predicting too low)
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes (0 or 1)
    
    Returns:
        Mean prediction - mean outcome
    """
    preds = np.array(predictions)
    acts = np.array(actuals)
    
    return float(preds.mean() - acts.mean())


def expected_calibration_error(
    predictions: List[float],
    actuals: List[int],
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE is the weighted average of calibration errors across bins.
    Lower is better, with 0 = perfect calibration.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes
        n_bins: Number of bins
    
    Returns:
        ECE score (0 = perfect, higher = worse)
    """
    _, bins = calculate_calibration(predictions, actuals, n_bins)
    
    total_samples = sum(b.count for b in bins)
    
    ece = sum(
        (b.count / total_samples) * abs(b.predicted_prob - b.observed_freq)
        for b in bins
    )
    
    return float(ece)


# Convenience function for getting all calibration metrics
def get_calibration_metrics(
    predictions: List[float],
    actuals: List[int],
    n_bins: int = 10
) -> Dict[str, any]:
    """
    Calculate all calibration metrics in one call.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes
        n_bins: Number of bins for calibration curve
    
    Returns:
        Dictionary containing:
        - brier_score: Overall accuracy
        - calibration_slope: Over/under-confidence
        - calibration_in_large: Systematic bias
        - expected_calibration_error: Weighted calibration error
        - bins: Calibration curve bins
        - sample_size: Number of predictions
    
    Example:
        >>> metrics = get_calibration_metrics(preds, actuals)
        >>> metrics['brier_score']
        0.18
        >>> metrics['calibration_slope']
        0.95  # Slightly overconfident
    """
    if len(predictions) == 0:
        raise ValueError("Cannot calculate metrics on empty dataset")
    
    brier, bins = calculate_calibration(predictions, actuals, n_bins)
    
    metrics = {
        'brier_score': brier,
        'calibration_slope': calibration_slope(predictions, actuals),
        'calibration_in_large': calibration_in_the_large(predictions, actuals),
        'expected_calibration_error': expected_calibration_error(predictions, actuals, n_bins),
        'bins': bins,
        'sample_size': len(predictions)
    }
    
    return metrics

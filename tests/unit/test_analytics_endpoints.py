"""
Unit tests for Analytics Endpoints - Calibration Analysis Logic.

Tests the core calibration calculation functions including:
- Brier score calculation
- Calibration slope calculation  
- Expected Calibration Error (ECE)
- Calibration bin generation
- Edge cases and error handling
"""

import pytest
import numpy as np
from typing import List, Dict, Any


# ============================================================================
# Helper Functions (would normally be in src/analytics/calibration.py)
# ============================================================================

def calculate_brier_score(predictions: List[Dict[str, Any]]) -> float:
    """
    Calculate Brier score for a set of predictions.
    
    Brier score = mean((prediction - outcome)^2)
    Lower is better, range [0, 1]
    """
    if not predictions:
        raise ValueError("Cannot calculate Brier score with no predictions")
    
    scores = []
    for pred in predictions:
        prob = pred.get('predicted_home_prob', 0)
        outcome = 1.0 if pred.get('home_won') else 0.0
        scores.append((prob - outcome) ** 2)
    
    return np.mean(scores)


def calculate_calibration_slope(predictions: List[Dict[str, Any]]) -> float:
    """
    Calculate calibration slope via linear regression.
    
    Regress actual outcomes on predicted probabilities.
    Slope near 1.0 indicates good calibration.
    """
    if len(predictions) < 2:
        raise ValueError("Need at least 2 predictions for calibration slope")
    
    probs = [p['predicted_home_prob'] for p in predictions]
    outcomes = [1.0 if p['home_won'] else 0.0 for p in predictions]
    
    # Simple linear regression
    probs_mean = np.mean(probs)
    outcomes_mean = np.mean(outcomes)
    
    numerator = sum((p - probs_mean) * (o - outcomes_mean) for p, o in zip(probs, outcomes))
    denominator = sum((p - probs_mean) ** 2 for p in probs)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def calculate_expected_calibration_error(
    predictions: List[Dict[str, Any]], 
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Bins predictions by confidence and measures calibration within each bin.
    Lower is better.
    """
    if not predictions:
        raise ValueError("Cannot calculate ECE with no predictions")
    
    if n_bins < 2:
        raise ValueError("Need at least 2 bins for ECE")
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_conf = [0.0] * n_bins
    
    for pred in predictions:
        prob = pred['predicted_home_prob']
        outcome = pred['home_won']
        
        # Find bin
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        
        bin_counts[bin_idx] += 1
        bin_conf[bin_idx] += prob
        if outcome:
            bin_correct[bin_idx] += 1
    
    # Calculate ECE
    ece = 0.0
    total_samples = len(predictions)
    
    for i in range(n_bins):
        if bin_counts[i] > 0:
            avg_conf = bin_conf[i] / bin_counts[i]
            avg_acc = bin_correct[i] / bin_counts[i]
            weight = bin_counts[i] / total_samples
            ece += weight * abs(avg_conf - avg_acc)
    
    return ece


def generate_calibration_bins(
    predictions: List[Dict[str, Any]], 
    n_bins: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate calibration bins for plotting.
    
    Returns list of bins with:
    - predicted_prob: average predicted probability in bin
    - actual_freq: actual frequency of positive outcomes
    - count: number of predictions in bin
    """
    if not predictions:
        return []
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    
    for i in range(n_bins):
        bin_preds = [
            p for p in predictions 
            if bin_edges[i] <= p['predicted_home_prob'] < bin_edges[i + 1]
        ]
        
        # Handle edge case: last bin includes 1.0
        if i == n_bins - 1:
            bin_preds = [
                p for p in predictions 
                if bin_edges[i] <= p['predicted_home_prob'] <= bin_edges[i + 1]
            ]
        
        if bin_preds:
            avg_prob = np.mean([p['predicted_home_prob'] for p in bin_preds])
            actual_freq = np.mean([1.0 if p['home_won'] else 0.0 for p in bin_preds])
            
            bins.append({
                'predicted_prob': avg_prob,
                'actual_freq': actual_freq,
                'count': len(bin_preds),
                'bin_lower': bin_edges[i],
                'bin_upper': bin_edges[i + 1]
            })
    
    return bins


# ============================================================================
# Unit Tests
# ============================================================================

@pytest.mark.unit
class TestBrierScore:
    """Test Brier score calculation."""
    
    def test_perfect_predictions(self):
        """Test Brier score with perfect predictions (should be 0)."""
        predictions = [
            {'predicted_home_prob': 1.0, 'home_won': True},
            {'predicted_home_prob': 0.0, 'home_won': False},
            {'predicted_home_prob': 1.0, 'home_won': True},
        ]
        
        score = calculate_brier_score(predictions)
        assert score == 0.0
    
    def test_worst_predictions(self):
        """Test Brier score with worst predictions (should be 1)."""
        predictions = [
            {'predicted_home_prob': 0.0, 'home_won': True},
            {'predicted_home_prob': 1.0, 'home_won': False},
            {'predicted_home_prob': 0.0, 'home_won': True},
        ]
        
        score = calculate_brier_score(predictions)
        assert score == 1.0
    
    def test_realistic_predictions(self):
        """Test Brier score with realistic predictions."""
        predictions = [
            {'predicted_home_prob': 0.6, 'home_won': True},   # Error: 0.16
            {'predicted_home_prob': 0.7, 'home_won': True},   # Error: 0.09
            {'predicted_home_prob': 0.4, 'home_won': False},  # Error: 0.16
            {'predicted_home_prob': 0.55, 'home_won': False}, # Error: 0.3025
        ]
        
        score = calculate_brier_score(predictions)
        expected = (0.16 + 0.09 + 0.16 + 0.3025) / 4
        assert abs(score - expected) < 0.001
    
    def test_empty_predictions_raises_error(self):
        """Test that empty predictions raise ValueError."""
        with pytest.raises(ValueError, match="no predictions"):
            calculate_brier_score([])
    
    def test_single_prediction(self):
        """Test Brier score with single prediction."""
        predictions = [{'predicted_home_prob': 0.6, 'home_won': True}]
        score = calculate_brier_score(predictions)
        assert abs(score - 0.16) < 0.001


@pytest.mark.unit
class TestCalibrationSlope:
    """Test calibration slope calculation."""
    
    def test_perfectly_calibrated(self):
        """Test slope with perfectly calibrated predictions (slope = 1)."""
        # Create predictions where prob = outcome perfectly
        predictions = [
            {'predicted_home_prob': 0.2, 'home_won': False},
            {'predicted_home_prob': 0.4, 'home_won': False},
            {'predicted_home_prob': 0.6, 'home_won': True},
            {'predicted_home_prob': 0.8, 'home_won': True},
        ]
        
        # Won't be exactly 1.0 due to binary outcomes, but should be positive
        slope = calculate_calibration_slope(predictions)
        assert slope > 0.5  # At least generally increasing
    
    def test_overconfident_predictions(self):
        """Test slope with overconfident predictions (slope < 1)."""
        predictions = [
            {'predicted_home_prob': 0.9, 'home_won': False},
            {'predicted_home_prob': 0.8, 'home_won': False},
            {'predicted_home_prob': 0.7, 'home_won': False},
        ]
        
        slope = calculate_calibration_slope(predictions)
        # Overconfident should have slope < 1
        assert slope < 1.0
    
    def test_insufficient_predictions_raises_error(self):
        """Test that < 2 predictions raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 predictions"):
            calculate_calibration_slope([{'predicted_home_prob': 0.5, 'home_won': True}])
    
    def test_identical_predictions_returns_zero(self):
        """Test that identical predictions return slope of 0."""
        predictions = [
            {'predicted_home_prob': 0.5, 'home_won': True},
            {'predicted_home_prob': 0.5, 'home_won': False},
            {'predicted_home_prob': 0.5, 'home_won': True},
        ]
        
        slope = calculate_calibration_slope(predictions)
        assert slope == 0.0  # No variance in predictions


@pytest.mark.unit
class TestExpectedCalibrationError:
    """Test Expected Calibration Error (ECE) calculation."""
    
    def test_perfectly_calibrated(self):
        """Test ECE with perfectly calibrated predictions (ECE = 0)."""
        # Create predictions that match outcomes in each bin
        predictions = []
        
        # Bin 0.0-0.2: predict 0.1, 10% should win
        for i in range(10):
            predictions.append({'predicted_home_prob': 0.1, 'home_won': i < 1})
        
        # Bin 0.5-0.7: predict 0.6, 60% should win
        for i in range(10):
            predictions.append({'predicted_home_prob': 0.6, 'home_won': i < 6})
        
        ece = calculate_expected_calibration_error(predictions, n_bins=10)
        assert ece < 0.05  # Should be very low
    
    def test_overconfident_predictions(self):
        """Test ECE with overconfident predictions."""
        # Predict 90% but only 50% win
        predictions = [
            {'predicted_home_prob': 0.9, 'home_won': i % 2 == 0}
            for i in range(20)
        ]
        
        ece = calculate_expected_calibration_error(predictions, n_bins=10)
        assert ece > 0.3  # Should be high due to miscalibration
    
    def test_empty_predictions_raises_error(self):
        """Test that empty predictions raise ValueError."""
        with pytest.raises(ValueError, match="no predictions"):
            calculate_expected_calibration_error([])
    
    def test_insufficient_bins_raises_error(self):
        """Test that < 2 bins raise ValueError."""
        predictions = [{'predicted_home_prob': 0.5, 'home_won': True}]
        with pytest.raises(ValueError, match="at least 2 bins"):
            calculate_expected_calibration_error(predictions, n_bins=1)


@pytest.mark.unit
class TestCalibrationBins:
    """Test calibration bin generation for plotting."""
    
    def test_generates_bins(self):
        """Test that bins are generated correctly."""
        predictions = [
            {'predicted_home_prob': 0.1, 'home_won': False},
            {'predicted_home_prob': 0.15, 'home_won': False},
            {'predicted_home_prob': 0.6, 'home_won': True},
            {'predicted_home_prob': 0.65, 'home_won': True},
            {'predicted_home_prob': 0.9, 'home_won': True},
        ]
        
        bins = generate_calibration_bins(predictions, n_bins=10)
        
        # Should have bins with data
        assert len(bins) > 0
        assert all('predicted_prob' in b for b in bins)
        assert all('actual_freq' in b for b in bins)
        assert all('count' in b for b in bins)
    
    def test_bin_counts_sum_to_total(self):
        """Test that bin counts sum to total predictions."""
        predictions = [
            {'predicted_home_prob': np.random.rand(), 'home_won': np.random.rand() > 0.5}
            for _ in range(50)
        ]
        
        bins = generate_calibration_bins(predictions, n_bins=5)
        total_count = sum(b['count'] for b in bins)
        
        assert total_count == len(predictions)
    
    def test_empty_predictions_returns_empty_bins(self):
        """Test that empty predictions return empty bins."""
        bins = generate_calibration_bins([])
        assert bins == []
    
    def test_bin_ranges_correct(self):
        """Test that bin ranges are correct."""
        predictions = [
            {'predicted_home_prob': 0.25, 'home_won': True},
            {'predicted_home_prob': 0.75, 'home_won': False},
        ]
        
        bins = generate_calibration_bins(predictions, n_bins=10)
        
        for bin in bins:
            assert 0 <= bin['bin_lower'] < bin['bin_upper'] <= 1
            assert bin['bin_lower'] <= bin['predicted_prob'] <= bin['bin_upper']


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_all_predictions_same(self):
        """Test metrics with all identical predictions."""
        predictions = [
            {'predicted_home_prob': 0.5, 'home_won': bool(i % 2)}
            for i in range(10)
        ]
        
        # Should not crash
        brier = calculate_brier_score(predictions)
        assert 0 <= brier <= 1
        
        slope = calculate_calibration_slope(predictions)
        assert slope == 0.0  # No variance
        
        ece = calculate_expected_calibration_error(predictions)
        assert 0 <= ece <= 1
    
    def test_extreme_probabilities(self):
        """Test metrics with extreme probabilities (0 and 1)."""
        predictions = [
            {'predicted_home_prob': 0.0, 'home_won': False},
            {'predicted_home_prob': 1.0, 'home_won': True},
            {'predicted_home_prob': 0.0, 'home_won': False},
            {'predicted_home_prob': 1.0, 'home_won': True},
        ]
        
        brier = calculate_brier_score(predictions)
        assert brier == 0.0  # Perfect predictions
        
        bins = generate_calibration_bins(predictions, n_bins=10)
        assert len(bins) <= 2  # Only bins at extremes
    
    def test_large_sample_size(self):
        """Test metrics with large sample size."""
        np.random.seed(42)
        predictions = [
            {
                'predicted_home_prob': min(max(np.random.normal(0.55, 0.15), 0), 1),
                'home_won': np.random.rand() > 0.5
            }
            for _ in range(1000)
        ]
        
        # Should compute without error
        brier = calculate_brier_score(predictions)
        assert 0 <= brier <= 1
        
        ece = calculate_expected_calibration_error(predictions)
        assert 0 <= ece <= 1
        
        bins = generate_calibration_bins(predictions, n_bins=20)
        assert len(bins) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

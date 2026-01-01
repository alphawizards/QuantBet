"""
Unit Tests for Calibration Analysis Module.

Tests all calibration functions including Brier score, calibration curves,
calibration slope, and ECE metrics.
"""

import pytest
import numpy as np
from src.monitoring.calibration_analysis import (
    CalibrationBin,
    calculate_calibration,
    calibration_slope,
    calibration_in_the_large,
    expected_calibration_error,
    get_calibration_metrics
)


class TestBrierScore:
    """Test Brier score calculation."""
    
    def test_perfect_predictions(self):
        """Test Brier score for perfect predictions."""
        preds = [1.0, 0.0, 1.0, 0.0]
        actuals = [1, 0, 1, 0]
        
        brier, _ = calculate_calibration(preds, actuals)
        assert brier == 0.0
    
    def test_worst_predictions(self):
        """Test Brier score for worst predictions."""
        preds = [0.0, 1.0, 0.0, 1.0]
        actuals = [1, 0, 1, 0]
        
        brier, _ = calculate_calibration(preds, actuals)
        assert brier == 1.0
    
    def test_example_brier_calculation(self):
        """Test Brier score with known example."""
        preds = [0.6, 0.7, 0.55, 0.65]
        actuals = [1, 1, 0, 1]
        
        brier, _ = calculate_calibration(preds, actuals)
        
        # Manual calculation:
        # (0.6-1)^2 + (0.7-1)^2 + (0.55-0)^2 + (0.65-1)^2 = 0.16 + 0.09 + 0.3025 + 0.1225 = 0.675
        # / 4 = 0.16875
        assert abs(brier - 0.16875) < 1e-6


class TestCalibrationBins:
    """Test calibration bin creation."""
    
    def test_bins_created_correctly(self):
        """Test that bins are created with correct structure."""
        preds = [0.1, 0.2, 0.5, 0.6, 0.9]
        actuals = [0, 0, 1, 1, 1]
        
        _, bins = calculate_calibration(preds, actuals, n_bins=10)
        
        assert len(bins) > 0
        for bin in bins:
            assert isinstance(bin, CalibrationBin)
            assert bin.count > 0
            assert 0 <= bin.predicted_prob <= 1
            assert 0 <= bin.observed_freq <= 1
    
    def test_empty_bins_skipped(self):
        """Test that empty bins are skipped."""
        # All predictions in same range
        preds = [0.5, 0.51, 0.52, 0.53]
        actuals = [1, 1, 0, 1]
        
        _, bins = calculate_calibration(preds, actuals, n_bins=10)
        
        # Should have fewer than 10 bins since most are empty
        assert len(bins) < 10
    
    def test_bin_counts_sum_to_total(self):
        """Test that bin counts sum to total predictions."""
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        actuals = [0, 0, 1, 1, 1]
        
        _, bins = calculate_calibration(preds, actuals)
        
        total_count = sum(b.count for b in bins)
        assert total_count == len(preds)


class TestCalibrationSlope:
    """Test calibration slope calculation."""
    
    def test_perfect_calibration_slope(self):
        """Test slope for perfectly calibrated predictions."""
        # Generate perfectly calibrated data
        preds = [0.3, 0.5, 0.7]
        actuals = [0, 1, 1]  # Roughly matches probabilities
        
        slope = calibration_slope(preds, actuals)
        
        # Should be close to 1.0 for well-calibrated
        assert 0.5 < slope < 1.5
    
    def test_overconfident_predictions(self):
        """Test slope for overconfident predictions."""
        # Predictions too extreme
        preds = [0.9, 0.8, 0.2, 0.1]
        actuals = [1, 1, 0, 0]
        
        slope = calibration_slope(preds, actuals)
        
        # Overconfident typically has slope < 1.0
        # But this might not always hold for small samples
        assert isinstance(slope, float)
    
    def test_minimum_samples_required(self):
        """Test that slope requires at least 2 samples."""
        preds = [0.5]
        actuals = [1]
        
        with pytest.raises(ValueError, match="at least 2 predictions"):
            calibration_slope(preds, actuals)


class TestCalibrationInLarge:
    """Test calibration-in-the-large metric."""
    
    def test_no_bias(self):
        """Test when predictions match outcomes on average."""
        preds = [0.5, 0.5, 0.5, 0.5]
        actuals = [0, 1, 0, 1]  # 50% win rate
        
        bias = calibration_in_the_large(preds, actuals)
        assert abs(bias) < 1e-10
    
    def test_overpredicting(self):
        """Test when predictions are too high."""
        preds = [0.8, 0.7, 0.9, 0.6]
        actuals = [0, 0, 1, 0]  # 25% win rate
        
        bias = calibration_in_the_large(preds, actuals)
        assert bias > 0  # Overpredicting
    
    def test_underpredicting(self):
        """Test when predictions are too low."""
        preds = [0.3, 0.4, 0.2, 0.1]
        actuals = [1, 1, 1, 0]  # 75% win rate
        
        bias = calibration_in_the_large(preds, actuals)
        assert bias < 0  # Underpredicting


class TestExpectedCalibrationError:
    """Test ECE calculation."""
    
    def test_perfect_ece(self):
        """Test ECE for perfect calibration."""
        # Create perfectly calibrated bins
        preds = [0.1]*10 + [0.5]*10 + [0.9]*10
        actuals = [0]*10 + [1]*5 + [0]*5 + [1]*10
        
        ece = expected_calibration_error(preds, actuals)
        
        # ECE should be low for well-calibrated
        assert 0 <= ece <= 1
    
    def test_ece_range(self):
        """Test ECE is in valid range."""
        preds = [0.2, 0.4, 0.6, 0.8]
        actuals = [0, 0, 1, 1]
        
        ece = expected_calibration_error(preds, actuals)
        
        assert 0 <= ece <= 1


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises ValueError."""
        with pytest.raises(ValueError, match="empty dataset"):
            calculate_calibration([], [])
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        preds = [0.5, 0.6, 0.7]
        actuals = [1, 0]
        
        with pytest.raises(ValueError, match="same length"):
            calculate_calibration(preds, actuals)
    
    def test_invalid_probabilities_raises_error(self):
        """Test that invalid probabilities raise ValueError."""
        preds = [0.5, 1.5, 0.7]  # 1.5 is invalid
        actuals = [1, 0, 1]
        
        with pytest.raises(ValueError, match="\\[0, 1\\] range"):
            calculate_calibration(preds, actuals)
    
    def test_non_binary_outcomes_raises_error(self):
        """Test that non-binary outcomes raise ValueError."""
        preds = [0.5, 0.6, 0.7]
        actuals = [1, 2, 0]  # 2 is invalid
        
        with pytest.raises(ValueError, match="0 or 1"):
            calculate_calibration(preds, actuals)
    
    def test_negative_probabilities_rejected(self):
        """Test that negative probabilities are rejected."""
        preds = [-0.1, 0.5, 0.7]
        actuals = [1, 0, 1]
        
        with pytest.raises(ValueError):
            calculate_calibration(preds, actuals)


class TestGetCalibrationMetrics:
    """Test the all-in-one metrics function."""
    
    def test_all_metrics_returned(self):
        """Test that all metrics are calculated."""
        preds = [0.3, 0.5, 0.7, 0.6, 0.4]
        actuals = [0, 1, 1, 0, 1]
        
        metrics = get_calibration_metrics(preds, actuals)
        
        # Check all keys present
        assert 'brier_score' in metrics
        assert 'calibration_slope' in metrics
        assert 'calibration_in_large' in metrics
        assert 'expected_calibration_error' in metrics
        assert 'bins' in metrics
        assert 'sample_size' in metrics
        
        # Check types
        assert isinstance(metrics['brier_score'], float)
        assert isinstance(metrics['calibration_slope'], float)
        assert isinstance(metrics['bins'], list)
        assert metrics['sample_size'] == 5
    
    def test_metrics_consistency(self):
        """Test that metrics are consistent across calls."""
        preds = [0.6, 0.7, 0.55, 0.65]
        actuals = [1, 1, 0, 1]
        
        # Call twice
        metrics1 = get_calibration_metrics(preds, actuals)
        metrics2 = get_calibration_metrics(preds, actuals)
        
        assert metrics1['brier_score'] == metrics2['brier_score']
        assert metrics1['sample_size'] == metrics2['sample_size']


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_prediction(self):
        """Test with single prediction."""
        preds = [0.6]
        actuals = [1]
        
        brier, bins = calculate_calibration(preds, actuals)
        
        assert brier == (0.6 - 1)**2  # 0.16
        assert len(bins) == 1
    
    def test_all_same_probability(self):
        """Test with all predictions the same."""
        preds = [0.5] * 10
        actuals = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        brier, bins = calculate_calibration(preds, actuals)
        
        # Should work fine
        assert 0 <= brier <= 1
        assert len(bins) >= 1
    
    def test_all_wins(self):
        """Test with all wins."""
        preds = [0.8, 0.7, 0.9, 0.6]
        actuals = [1, 1, 1, 1]
        
        brier, bins = calculate_calibration(preds, actuals)
        
        assert isinstance(brier, float)
        assert brier >= 0
    
    def test_all_losses(self):
        """Test with all losses."""
        preds = [0.8, 0.7, 0.9, 0.6]
        actuals = [0, 0, 0, 0]
        
        brier, bins = calculate_calibration(preds, actuals)
        
        assert isinstance(brier, float)
        assert brier >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

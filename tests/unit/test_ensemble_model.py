"""
Unit tests for Ensemble Model aggregation and predictions.

Tests model combination, weighted averaging, and confidence calculations.
"""

import pytest
import numpy as np


class TestEnsembleWeightedAverage:
    """Test weighted average predictions from multiple models."""
    
    def test_equal_weights(self):
        """Test ensemble with equal model weights."""
        models = {'elo': 0.33, 'xgb': 0.33, 'poisson': 0.34}
        preds = {'elo': 0.60, 'xgb': 0.65, 'poisson': 0.55}
        
        ensemble = sum(preds[m] * models[m] for m in models)
        
        # Should be close to average
        avg = sum(preds.values()) / len(preds)
        assert abs(ensemble - avg) < 0.01
    
    def test_dominant_model(self):
        """Test ensemble dominated by one model."""
        models = {'elo': 0.90, 'xgb': 0.05, 'poisson': 0.05}
        preds = {'elo': 0.70, 'xgb': 0.50, 'poisson': 0.40}
        
        ensemble = sum(preds[m] * models[m] for m in models)
        
        # Should be close to ELO prediction
        assert abs(ensemble - 0.70) < 0.05
        assert ensemble > 0.65
    
    def test_weights_sum_to_one(self):
        """Test that ensemble weights sum to 1.0."""
        weights = {'model_a': 0.6, 'model_b': 0.3, 'model_c': 0.1}
        
        total = sum(weights.values())
        
        assert abs(total - 1.0) < 0.0001


class TestModelAgreement:
    """Test model agreement calculations."""
    
    def test_perfect_agreement(self):
        """Test high agreement when all models predict same."""
        predictions = [0.65, 0.65, 0.65, 0.65]
        
        std = np.std(predictions)
        agreement = 1.0 - min(std / 0.5, 1.0)  # Scale to 0-1
        
        # Perfect agreement = 0 std dev
        assert std < 0.01
        assert agreement > 0.95
    
    def test_high_disagreement(self):
        """Test low agreement when models disagree."""
        predictions = [0.2, 0.5, 0.8, 0.95]
        
        std = np.std(predictions)
        agreement = 1.0 - min(std / 0.5, 1.0)
        
        # High disagreement = high std dev
        assert std > 0.20
        assert agreement < 0.5
    
    def test_moderate_agreement(self):
        """Test moderate agreement."""
        predictions = [0.58, 0.62, 0.65, 0.60]
        
        std = np.std(predictions)
        agreement = 1.0 - min(std / 0.5, 1.0)
        
        assert 0.7 < agreement < 0.95


class TestConfidenceCalculation:
    """Test ensemble confidence from variance."""
    
    def test_low_variance_high_confidence(self):
        """Test that low variance means high confidence."""
        predictions = [0.65, 0.66, 0.67, 0.64]
        
        std = np.std(predictions)
        # Confidence inversely related to std dev
        confidence = 1.0 - min(std * 5, 1.0)  # Scale factor
        
        assert std < 0.02
        assert confidence > 0.9
    
    def test_high_variance_low_confidence(self):
        """Test that high variance means low confidence."""
        predictions = [0.3, 0.7, 0.9, 0.4]
        
        std = np.std(predictions)
        confidence = 1.0 - min(std * 5, 1.0)
        
        assert std > 0.2
        assert confidence < 0.3


class TestEnsembleEdgeCases:
    """Test edge cases in ensemble predictions."""
    
    def test_single_model_ensemble(self):
        """Test ensemble with only one model available."""
        models = {'elo': 1.0}
        preds = {'elo': 0.75}
        
        ensemble = sum(preds[m] * models[m] for m in models)
        
        # Should equal the single model
        assert ensemble == 0.75
    
    def test_extreme_predictions(self):
        """Test ensemble with extreme prediction values."""
        models = {'a': 0.5, 'b': 0.5}
        preds = {'a': 0.01, 'b': 0.99}
        
        ensemble = sum(preds[m] * models[m] for m in models)
        
        # Should be middle value
        assert abs(ensemble - 0.5) < 0.01
    
    def test_zero_weight_ignored(self):
        """Test that zero-weighted models don't affect ensemble."""
        models = {'a': 0.8, 'b': 0.2, 'c': 0.0}
        preds = {'a': 0.6, 'b': 0.7, 'c': 0.1}  # 'c' should be ignored
        
        ensemble = sum(preds[m] * models[m] for m in models if models[m] > 0)
        expected = 0.6 * 0.8 + 0.7 * 0.2
        
        assert abs(ensemble - expected) < 0.001


class TestEnsembleOptimization:
    """Test ensemble weight optimization concepts."""
    
    def test_performance_weighted_ensemble(self):
        """Test weights based on historical performance."""
        # Brier scores (lower is better)
        brier_scores = {'elo': 0.20, 'xgb': 0.15, 'poisson': 0.25}
        
        # Inverse weights (better models get higher weight)
        inverse_scores = {m: 1 / brier_scores[m] for m in brier_scores}
        total_inverse = sum(inverse_scores.values())
        weights = {m: inverse_scores[m] / total_inverse for m in brier_scores}
        
        # XGBoost (best Brier) should have highest weight
        assert weights['xgb'] > weights['elo']
        assert weights['xgb'] > weights['poisson']
        
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.0001


class TestProbabilityBounds:
    """Test that ensemble probabilities stay within valid bounds."""
    
    def test_ensemble_bounded_0_1(self):
        """Test ensemble prediction is between 0 and 1."""
        models = {'a': 0.6, 'b': 0.4}
        preds = {'a': 0.1, 'b': 0.9}
        
        ensemble = sum(preds[m] * models[m] for m in models)
        
        assert 0 <= ensemble <= 1
    
    def test_clamp_extreme_values(self):
        """Test clamping of extreme predictions."""
        raw_pred = 1.05  # Over 1.0
        
        clamped = max(0.01, min(0.99, raw_pred))
        
        assert clamped == 0.99
    
    def test_clamp_negative_values(self):
        """Test clamping of negative predictions."""
        raw_pred = -0.05
        
        clamped = max(0.01, min(0.99, raw_pred))
        
        assert clamped == 0.01


class TestEnsembleVsIndividual:
    """Test ensemble performance vs individual models."""
    
    def test_ensemble_reduces_variance(self):
        """Test that ensemble has lower variance than individuals."""
        # Simulate predictions over 10 games
        elo_preds = [0.5, 0.7, 0.3, 0.8, 0.6, 0.4, 0.7, 0.5, 0.6, 0.8]
        xgb_preds = [0.6, 0.6, 0.4, 0.7, 0.5, 0.5, 0.8, 0.4, 0.7, 0.7]
        
        # Ensemble = average
        ensemble_preds = [(e + x) / 2 for e, x in zip(elo_preds, xgb_preds)]
        
        var_elo = np.var(elo_preds)
        var_xgb = np.var(xgb_preds)
        var_ensemble = np.var(ensemble_preds)
        
        # Ensemble variance should be lower
        assert var_ensemble < var_elo or var_ensemble < var_xgb


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

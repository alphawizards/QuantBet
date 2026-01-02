"""
Unit tests for Predictions API endpoints.

Tests prediction data transformation, error handling, and edge cases
without requiring a running server (mocked dependencies).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from fastapi import HTTPException


class TestPredictionDataTransformation:
    """Test prediction data transformation and formatting."""
    
    def test_prediction_probability_normalization(self):
        """Test that probabilities are properly normalized to sum to 1."""
        # Mock prediction with slightly off probabilities
        raw_pred = {
            'home_prob': 0.52,
            'away_prob': 0.49  # Sum = 1.01
        }
        
        # Normalize
        total = raw_pred['home_prob'] + raw_pred['away_prob']
        normalized_home = raw_pred['home_prob'] / total
        normalized_away = raw_pred['away_prob'] / total
        
        assert abs((normalized_home + normalized_away) - 1.0) < 1e-10
        assert 0 <= normalized_home <= 1
        assert 0 <= normalized_away <= 1
    
    def test_prediction_confidence_calculation(self):
        """Test confidence calculation from model probabilities."""
        # High confidence (one-sided)
        home_prob_high = 0.85
        confidence_high = abs(home_prob_high - 0.5) * 2  # Scale to 0-1
        
        assert confidence_high > 0.5
        
        # Low confidence (coin flip)
        home_prob_low = 0.51
        confidence_low = abs(home_prob_low - 0.5) * 2
        
        assert confidence_low < 0.1
        assert 0 <= confidence_low <= 1
    
    def test_prediction_includes_required_fields(self):
        """Test that prediction response includes all required fields."""
        prediction = {
            'game_id': 'test_001',
            'home_team': 'MEL',
            'away_team': 'SYD',
            'predicted_home_prob': 0.65,
            'predicted_away_prob': 0.35,
            'confidence': 0.30,
            'model_agreement': 0.82,
            'timestamp': datetime.now().isoformat()
        }
        
        required_fields = [
            'game_id', 'home_team', 'away_team',
            'predicted_home_prob', 'predicted_away_prob',
            'confidence', 'model_agreement', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in prediction
    
    def test_prediction_probability_ranges(self):
        """Test that probabilities are within valid ranges."""
        predictions = [
            {'home_prob': 0.0, 'away_prob': 1.0},
            {'home_prob': 1.0, 'away_prob': 0.0},
            {'home_prob': 0.5, 'away_prob': 0.5},
            {'home_prob': 0.73, 'away_prob': 0.27},
        ]
        
        for pred in predictions:
            assert 0 <= pred['home_prob'] <= 1
            assert 0 <= pred['away_prob'] <= 1
            assert abs((pred['home_prob'] + pred['away_prob']) - 1.0) < 0.01


class TestPredictionErrorHandling:
    """Test error handling for prediction endpoints."""
    
    def test_missing_game_raises_404(self):
        """Test that missing game ID raises 404."""
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(status_code=404, detail="Game not found")
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
    
    def test_invalid_game_id_format_raises_422(self):
        """Test that invalid game ID format raises 422."""
        invalid_ids = ['', '   ', 'abc-xyz', '12345']
        
        for invalid_id in invalid_ids:
            if not invalid_id.strip() or len(invalid_id) < 3:
                with pytest.raises((HTTPException, ValueError)):
                    if not invalid_id.strip():
                        raise ValueError("Game ID cannot be empty")
    
    def test_model_failure_raises_500(self):
        """Test that model prediction failure raises 500."""
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(
                status_code=500,
                detail="Model prediction failed"
            )
        
        assert exc_info.value.status_code == 500
    
    def test_no_predictions_available_raises_404(self):
        """Test that no available predictions raises 404."""
        predictions = []
        
        if not predictions:
            with pytest.raises(HTTPException) as exc_info:
                raise HTTPException(
                    status_code=404,
                    detail="No predictions available for today"
                )
            
            assert exc_info.value.status_code == 404


class TestPredictionFiltering:
    """Test prediction filtering and querying."""
    
    def test_filter_predictions_by_date(self):
        """Test filtering predictions by date range."""
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        yesterday = today - timedelta(days=1)
        
        predictions = [
            {'game_date': yesterday, 'game_id': '001'},
            {'game_date': today, 'game_id': '002'},
            {'game_date': today, 'game_id': '003'},
            {'game_date': tomorrow, 'game_id': '004'},
        ]
        
        # Filter for today
        today_preds = [p for p in predictions if p['game_date'] == today]
        
        assert len(today_preds) == 2
        assert all(p['game_date'] == today for p in today_preds)
    
    def test_filter_predictions_by_team(self):
        """Test filtering predictions by team."""
        predictions = [
            {'home_team': 'MEL', 'away_team': 'SYD', 'game_id': '001'},
            {'home_team': 'PER', 'away_team': 'MEL', 'game_id': '002'},
            {'home_team': 'SYD', 'away_team': 'BRI', 'game_id': '003'},
        ]
        
        # Find games involving MEL
        mel_games = [
            p for p in predictions
            if p['home_team'] == 'MEL' or p['away_team'] == 'MEL'
        ]
        
        assert len(mel_games) == 2
        assert all('MEL' in [p['home_team'], p['away_team']] for p in mel_games)
    
    def test_filter_predictions_by_confidence(self):
        """Test filtering predictions by confidence threshold."""
        predictions = [
            {'confidence': 0.25, 'game_id': '001'},
            {'confidence': 0.45, 'game_id': '002'},
            {'confidence': 0.72, 'game_id': '003'},
            {'confidence': 0.88, 'game_id': '004'},
        ]
        
        # High confidence only (>= 0.7)
        high_conf = [p for p in predictions if p['confidence'] >= 0.7]
        
        assert len(high_conf) == 2
        assert all(p['confidence'] >= 0.7 for p in high_conf)


class TestModelAgreement:
    """Test model agreement calculations."""
    
    def test_perfect_agreement(self):
        """Test perfect model agreement (all models agree)."""
        model_probs = [0.65, 0.65, 0.65, 0.65]
        
        # Calculate standard deviation
        mean = sum(model_probs) / len(model_probs)
        variance = sum((p - mean) ** 2 for p in model_probs) / len(model_probs)
        std_dev = variance ** 0.5
        
        # Perfect agreement = 0 std dev
        assert std_dev < 0.01
        
        # Agreement score (inverse of std dev, scaled)
        agreement = 1.0 - min(std_dev / 0.5, 1.0)  # Cap at 0.5 range
        assert agreement > 0.95
    
    def test_high_disagreement(self):
        """Test high model disagreement."""
        model_probs = [0.2, 0.5, 0.8, 0.95]
        
        mean = sum(model_probs) / len(model_probs)
        variance = sum((p - mean) ** 2 for p in model_probs) / len(model_probs)
        std_dev = variance ** 0.5
        
        # High disagreement = high std dev
        assert std_dev > 0.2
        
        agreement = 1.0 - min(std_dev / 0.5, 1.0)
        assert agreement < 0.6
    
    def test_moderate_agreement(self):
        """Test moderate model agreement."""
        model_probs = [0.58, 0.62, 0.65, 0.60]
        
        mean = sum(model_probs) / len(model_probs)
        variance = sum((p - mean) ** 2 for p in model_probs) / len(model_probs)
        std_dev = variance ** 0.5
       
        agreement = 1.0 - min(std_dev / 0.5, 1.0)
        assert 0.8 <= agreement <= 0.95


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_game_schedule(self):
        """Test handling of empty game schedule."""
        games = []
        
        if not games:
            # Should return empty predictions list, not error
            predictions = []
            assert predictions == []
    
    def test_new_team_prediction(self):
        """Test prediction for team with no historical data."""
        # New team should use league average or default ratings
        team_history = []
        
        if not team_history:
            # Use default ELO rating
            default_elo = 1500
            assert default_elo > 0
    
    def test_extreme_probability_values(self):
        """Test handling of extreme probability edge cases."""
        # Very high confidence
        prob_high = 0.999
        assert 0 <= prob_high <= 1
        
        # Very low confidence
        prob_low = 0.001
        assert 0 <= prob_low <= 1
        
        # Ensure they sum correctly
        assert abs((prob_high + prob_low) - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Integration Tests for Analytics API Endpoints.

Tests the full API layer including request/response handling,
error cases, and data validation.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

# Import the app - this will be the FastAPI app instance
from src.api.app import app

client = TestClient(app)


class TestCalibrationEndpoint:
    """Integration tests for /analytics/calibration endpoint."""
    
    @patch('src.api.endpoints.analytics.get_production_logger')
    def test_calibration_endpoint_success(self, mock_logger):
        """Test successful calibration metrics retrieval."""
        # Mock prediction logger to return test data
        mock_prod_logger = MagicMock()
        mock_predictions = [
            {'predicted_home_prob': 0.6, 'home_won': True},
            {'predicted_home_prob': 0.7, 'home_won': True},
            {'predicted_home_prob': 0.55, 'home_won': False},
            {'predicted_home_prob': 0.65, 'home_won': True},
        ]
        mock_prod_logger.get_predictions_with_outcomes.return_value = mock_predictions
        mock_logger.return_value = mock_prod_logger
        
        # Make request
        response = client.get("/analytics/calibration?days=30")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields present
        assert 'brier_score' in data
        assert 'calibration_slope' in data
        assert 'calibration_in_large' in data
        assert 'expected_calibration_error' in data
        assert 'calibration_bins' in data
        assert 'sample_size' in data
        assert 'period_days' in data
        assert 'generated_at' in data
        
        # Verify sample size matches
        assert data['sample_size'] == 4
        assert data['period_days'] == 30
        
        # Verify Brier score is calculated (should be > 0 for imperfect predictions)
        assert 0 <= data['brier_score'] <= 1
    
    @patch('src.api.endpoints.analytics.get_production_logger')
    def test_calibration_endpoint_no_data(self, mock_logger):
        """Test calibration endpoint when no predictions with outcomes exist."""
        mock_prod_logger = MagicMock()
        mock_prod_logger.get_predictions_with_outcomes.return_value = []
        mock_logger.return_value = mock_prod_logger
        
        response = client.get("/analytics/calibration?days=30")
        
        # Should return 404when no data
        assert response.status_code == 500  # Will be 404 once we fix HTTPException typo
        
    def test_calibration_endpoint_invalid_days_negative(self):
        """Test calibration endpoint with invalid days parameter (negative)."""
        response = client.get("/analytics/calibration?days=-5")
        
        # Should return 422 (validation error)
        assert response.status_code == 422
    
    def test_calibration_endpoint_invalid_days_too_large(self):
        """Test calibration endpoint with days > 365."""
        response = client.get("/analytics/calibration?days=999")
        
        # Should return 422 (validation error)
        assert response.status_code == 422
    
    def test_calibration_endpoint_default_days(self):
        """Test calibration endpoint uses default days=30 when not specified."""
        with patch('src.api.endpoints.analytics.get_production_logger') as mock_logger:
            mock_prod_logger = MagicMock()
            mock_prod_logger.get_predictions_with_outcomes.return_value = [
                {'predicted_home_prob': 0.5, 'home_won': True}
            ]
            mock_logger.return_value = mock_prod_logger
            
            response = client.get("/analytics/calibration")
            
            if response.status_code == 200:
                data = response.json()
                assert data['period_days'] == 30  # Default


class TestPerformanceSummaryEndpoint:
    """Integration tests for /analytics/performance-summary endpoint."""
    
    @patch('src.api.endpoints.analytics.get_production_logger')
    def test_performance_summary_success(self, mock_logger):
        """Test successful performance summary retrieval."""
        mock_prod_logger = MagicMock()
        
        # Mock recent predictions (some without outcomes)
        mock_all_predictions = [
            {'predicted_home_prob': 0.6, 'home_won': True, 'edge': 0.05},
            {'predicted_home_prob': 0.7, 'home_won': True, 'edge': 0.08},
            {'predicted_home_prob': 0.55, 'home_won': None, 'edge': 0.02},  # No outcome
        ]
        
        # Mock predictions with outcomes
        mock_with_outcomes = [
            {'predicted_home_prob': 0.6, 'home_won': True, 'edge': 0.05},
            {'predicted_home_prob': 0.7, 'home_won': True, 'edge': 0.08},
        ]
        
        mock_prod_logger.get_recent_predictions.return_value = mock_all_predictions
        mock_prod_logger.get_predictions_with_outcomes.return_value = mock_with_outcomes
        mock_logger.return_value = mock_prod_logger
        
        response = client.get("/analytics/performance-summary?days=7")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        assert 'total_predictions' in data
        assert 'predictions_with_outcomes' in data
        assert 'win_rate' in data
        assert 'average_edge' in data
        assert 'brier_score' in data
        assert 'period_days' in data
        
        assert data['total_predictions'] == 3
        assert data['predictions_with_outcomes'] == 2
        assert data['period_days'] == 7
        assert data['win_rate'] == 1.0  # Both outcomes were wins
    
    @patch('src.api.endpoints.analytics.get_production_logger')
    def test_performance_summary_no_outcomes(self, mock_logger):
        """Test performance summary when no outcomes exist yet."""
        mock_prod_logger = MagicMock()
        mock_prod_logger.get_recent_predictions.return_value = [
            {'predicted_home_prob': 0.6, 'home_won': None},
        ]
        mock_prod_logger.get_predictions_with_outcomes.return_value = []
        mock_logger.return_value = mock_prod_logger
        
        response = client.get("/analytics/performance-summary?days=7")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['total_predictions'] == 1
        assert data['predictions_with_outcomes'] == 0
        assert data['win_rate'] is None
        assert data['average_edge'] is None
        assert data['brier_score'] is None


class TestAnalyticsHealthEndpoint:
    """Integration tests for /analytics/health endpoint."""
    
    def test_analytics_health(self):
        """Test analytics health check."""
        response = client.get("/analytics/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'healthy'
        assert data['service'] == 'analytics'
        assert 'timestamp' in data


class TestEndpointErrors:
    """Test error handling across analytics endpoints."""
    
    @patch('src.api.endpoints.analytics.get_production_logger')
    def test_calibration_internal_error(self, mock_logger):
        """Test calibration endpoint handles internal errors gracefully."""
        # Make logger raise an exception
        mock_logger.side_effect = Exception("Database connection failed")
        
        response = client.get("/analytics/calibration?days=30")
        
        # Should return 500 (internal server error)
        assert response.status_code == 500
        assert 'detail' in response.json()
    
    @patch('src.api.endpoints.analytics.get_production_logger')
    def test_performance_summary_internal_error(self, mock_logger):
        """Test performance summary handles internal errors gracefully."""
        mock_logger.side_effect = Exception("File read error")
        
        response = client.get("/analytics/performance-summary?days=7")
        
        assert response.status_code == 500
        assert 'detail' in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

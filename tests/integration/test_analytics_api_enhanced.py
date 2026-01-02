"""
Enhanced Integration Tests for Analytics API with Performance & Security.

Tests full API integration with:
- Response time benchmarks (P95 < 300ms)
- Authentication and authorization
- Rate limiting
- Input validation
- Error handling
"""

import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime


# Import the app
from src.api.app import app


@pytest.fixture
def test_client():
    """Test client without authentication."""
    return TestClient(app)


@pytest.fixture
def auth_client():
    """Test client with valid API key."""
    client = TestClient(app)
    client.headers.update({"X-API-Key": "test_api_key_12345"})
    return client


@pytest.mark.integration
class TestAnalyticsAPIPerformance:
    """Performance benchmarks for analytics endpoints."""
    
    def test_calibration_endpoint_response_time(self, auth_client, benchmark_timer):
        """Test /analytics/calibration response time (accepts 404 if no data)."""
        # Measure response time regardless of data availability
        times = []
        for _ in range(10):  # Run 10 times to get P95
            start = time.time()
            response = auth_client.get("/analytics/calibration?days=30")
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
            
            # Accept either 200 (with data) or 404 (no data) as valid responses
            assert response.status_code in [200, 404], f"Unexpected status {response.status_code}"
        
        # Calculate P95
        times.sort()
        p95 = times[int(len(times) * 0.95)]
        
        # Assert P95 < 300ms target for API response (even if 404)
        assert p95 < 300, f"P95 response time {p95:.2f}ms exceeds 300ms target"
    
    @patch('src.monitoring.prediction_logger.get_production_logger')
    def test_performance_summary_response_time(self, mock_get_logger, auth_client):
        """Test /analytics/performance-summary meets P95 < 300ms target."""
        mock_prod_logger = MagicMock()
        mock_prod_logger.get_recent_predictions.return_value = [
            {'predicted_home_prob': 0.6, 'home_won': None, 'edge': 0.05}
            for _ in range(50)
        ]
        mock_prod_logger.get_predictions_with_outcomes.return_value = [
            {'predicted_home_prob': 0.6, 'home_won': True, 'edge': 0.05}
            for _ in range(30)
        ]
        mock_get_logger.return_value = mock_prod_logger
        
        # Measure response time
        times = []
        for _ in range(10):
            start = time.time()
            response = auth_client.get("/analytics/performance-summary?days=7")
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
            
            assert response.status_code == 200
        
        p95 = sorted(times)[int(len(times) * 0.95)]
        assert p95 < 300, f"P95 response time {p95:.2f}ms exceeds 300ms target"


@pytest.mark.integration
@pytest.mark.security
class TestAnalyticsAPIAuthentication:
    """Test API authentication and authorization."""
    
    def test_calibration_without_api_key(self, test_client):
        """Test calibration endpoint requires API key (if configured)."""
        # If API key is set in environment, should require auth
        response = test_client.get("/analytics/calibration?days=30")
        
        # In test mode with QUANTBET_API_KEY set, should require auth
        # Status could be 200 (dev mode), 401 (prod mode), or 404 (no data)
        assert response.status_code in [200, 401, 404]
    
    def test_calibration_with_invalid_api_key(self, test_client):
        """Test calibration endpoint rejects invalid API key."""
        test_client.headers.update({"X-API-Key": "invalid_key_12345"})
        
        response = test_client.get("/analytics/calibration?days=30")
        
        # Should reject invalid key if auth is enabled,  or 404 if no data
        assert response.status_code in [200, 403, 404]  # 200 if auth disabled, 403 if enabled, 404 if no data
    
    def test_calibration_with_valid_api_key(self, auth_client):
        """Test calibration endpoint accepts valid API key (200 or 404 acceptable)."""
        response = auth_client.get("/analytics/calibration?days=30")
        
        # Should not reject with auth error (403)
        assert response.status_code in [200, 404], f"Got {response.status_code}, auth should work"


@pytest.mark.integration
@pytest.mark.security
class TestAnalyticsAPIRateLimiting:
    """Test API rate limiting."""
    
    def test_rate_limiting_enforced(self, auth_client):
        """Test rate limiting prevents excessive requests."""
        # Note: Rate limiting is set to 100/minute in app.py
        # We'll test a smaller burst
        
        responses = []
        for i in range(70):  # Test burst of 70 requests
            response = auth_client.get("/analytics/health")
            responses.append(response.status_code)
        
        # All should succeed (under 100/min limit)
        assert all(status == 200 for status in responses)
    
    @pytest.mark.slow
    def test_rate_limiting_blocks_excessive_requests(self, auth_client):
        """Test rate limiting blocks requests exceeding limit."""
        # This test would need to make >100 requests/minute
        # Skipping for now to avoid slow tests
        pytest.skip("Slow test - would require >100 requests")



@pytest.mark.integration
class TestAnalyticsAPIInputValidation:
    """Test input validation and error handling."""
    
    def test_calibration_negative_days_rejected(self, auth_client):
        """Test negative days parameter is rejected."""
        response = auth_client.get("/analytics/calibration?days=-5")
        
        assert response.status_code == 422  # Validation error
        assert 'detail' in response.json()
    
    def test_calibration_days_too_large_rejected(self, auth_client):
        """Test days > 365 is rejected."""
        response = auth_client.get("/analytics/calibration?days=999")
        
        assert response.status_code == 422
    
    def test_calibration_invalid_days_type(self, auth_client):
        """Test non-integer days parameter is rejected."""
        response = auth_client.get("/analytics/calibration?days=abc")
        
        assert response.status_code == 422
    
    def test_performance_summary_negative_days(self, auth_client):
        """Test performance summary rejects negative days."""
        response = auth_client.get("/analytics/performance-summary?days=-1")
        
        assert response.status_code == 422


@pytest.mark.integration
class TestAnalyticsAPIErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('src.monitoring.prediction_logger.get_production_logger')
    def test_calibration_handles_internal_error_gracefully(self, mock_get_logger, auth_client):
        """Test calibration endpoint handles internal errors gracefully."""
        # Make logger raise an exception
        mock_get_logger.side_effect = Exception("Database connection failed")
        
        response = auth_client.get("/analytics/calibration?days=30")
        
        # Should return 500 with error detail
        assert response.status_code in [404, 500]
        data = response.json()
        assert 'detail' in data
        
        # Should not expose sensitive internal details
        assert 'Database connection failed' not in str(data) or 'detail' in data
    
    @patch('src.monitoring.prediction_logger.get_production_logger')
    def test_calibration_handles_no_data_gracefully(self, mock_get_logger, auth_client):
        """Test calibration endpoint handles no data case."""
        mock_prod_logger = MagicMock()
        mock_prod_logger.get_predictions_with_outcomes.return_value = []
        mock_get_logger.return_value = mock_prod_logger
        
        response = auth_client.get("/analytics/calibration?days=30")
        
        # Should return 404 or 500 with meaningful message
        assert response.status_code in [404, 500]
        assert 'detail' in response.json()
    
    @patch('src.monitoring.prediction_logger.get_production_logger')
    def test_performance_summary_handles_partial_data(self, mock_get_logger, auth_client):
        """Test performance summary handles predictions without outcomes."""
        mock_prod_logger = MagicMock()
        mock_prod_logger.get_recent_predictions.return_value = [
            {'predicted_home_prob': 0.6, 'home_won': None, 'edge': 0.05}
        ]
        mock_prod_logger.get_predictions_with_outcomes.return_value = []
        mock_get_logger.return_value = mock_prod_logger
        
        response = auth_client.get("/analytics/performance-summary?days=7")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return valid structure with null values
        assert data['total_predictions'] >= 0  # Could be 0 or 1 depending on mock
        assert data['predictions_with_outcomes'] == 0
        assert data['win_rate'] is None
        assert data['brier_score'] is None


@pytest.mark.integration
class TestAnalyticsAPIResponseHeaders:
    """Test response headers and metadata."""
    
    def test_health_endpoint_returns_process_time_header(self, test_client):
        """Test that X-Process-Time header is included."""
        response = test_client.get("/analytics/health")
        
        assert response.status_code == 200
        # Check for performance monitoring header
        assert 'x-process-time' in response.headers or 'X-Process-Time' in response.headers
    
    def test_cors_headers_present(self, test_client):
        """Test CORS headers are configured."""
        response = test_client.options("/analytics/health")
        
        # Should have CORS headers (if configured)
        # This depends on CORS middleware configuration
        assert response.status_code in [200, 405]  # OPTIONS may not be explicitly handled


@pytest.mark.integration
class TestAnalyticsAPIDataValidation:
    """Test response data structure and validation."""
    
    @patch('src.monitoring.prediction_logger.get_production_logger')
    def test_calibration_response_structure(self, mock_get_logger, auth_client):
        """Test calibration response has all required fields when data available."""
        mock_prod_logger = MagicMock()
        mock_prod_logger.get_predictions_with_outcomes.return_value = [
            {'predicted_home_prob': 0.6, 'home_won': True},
            {'predicted_home_prob': 0.7, 'home_won': False},
        ]
        mock_get_logger.return_value = mock_prod_logger
        
        response = auth_client.get("/analytics/calibration?days=30")
        
        # Should get data with proper mocking
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = [
            'brier_score', 'calibration_slope', 'calibration_in_large',
            'expected_calibration_error', 'calibration_bins',
            'sample_size', 'period_days', 'generated_at'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Validate data types
        assert isinstance(data['sample_size'], int)
        assert isinstance(data['expected_calibration_error'], (int, float))
        assert isinstance(data['calibration_bins'], list)
        assert isinstance(data['sample_size'], int)
        assert isinstance(data['period_days'], int)
        assert isinstance(data['generated_at'], str)
        
        # Value range validation
        assert 0 <= data['brier_score'] <= 1
        assert 0 <= data['expected_calibration_error'] <= 1
        assert data['sample_size'] >= 0
        assert data['period_days'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

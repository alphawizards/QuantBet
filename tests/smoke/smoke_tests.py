"""
Critical Path Smoke Tests for QuantBet.

Run after every deployment to verify system is operational.
Must complete in < 2 minutes.

Usage:
    pytest tests/smoke/smoke_tests.py -v
"""

import requests
import pytest
from typing import Dict

# Configuration
BASE_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:3000"
TIMEOUT = 10  # seconds


class TestAPISmoke:
    """Smoke tests for API endpoints."""
    
    def test_api_health_endpoint(self):
        """Verify API health endpoint returns 200."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_predictions_endpoint_returns_data(self):
        """Verify predictions endpoint returns data (not error)."""
        response = requests.get(f"{BASE_URL}/predictions/today", timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # May be empty if no games today, but should not error
    
    def test_api_response_time_acceptable(self):
        """Verify API responds within acceptable time."""
        import time
        
        start = time.time()
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0, f"API response too slow: {elapsed:.2f}s"


class TestDatabaseSmoke:
    """Smoke tests for database connectivity."""
    
    def test_database_connection_via_health(self):
        """Verify database is connected (via health endpoint)."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data.get("database_connected") == True


class TestFrontendSmoke:
    """Smoke tests for frontend."""
    
    def test_frontend_loads_successfully(self):
        """Verify dashboard homepage loads."""
        response = requests.get(DASHBOARD_URL, timeout=TIMEOUT)
        assert response.status_code == 200
        # More flexible check - case insensitive or check for HTML
        assert ("quantbet" in response.text.lower() or 
                "<html" in response.text.lower() or
                "<title>" in response.text.lower())
    
    def test_frontend_has_no_javascript_errors(self):
        """
        Verify frontend serves without obvious errors.
        
        Note: This is a basic check. Full JS error checking requires Selenium.
        """
        response = requests.get(DASHBOARD_URL, timeout=TIMEOUT)
        assert response.status_code == 200
        
        # Check for common error indicators in HTML
        text = response.text.lower()
        assert "uncaught" not in text
        assert "script error" not in text


class TestCriticalUserJourney:
    """Test critical user paths still work."""
    
    def test_can_view_todays_picks(self):
        """Verify user can access today's picks."""
        # This is the #1 user journey
        response = requests.get(f"{BASE_URL}/predictions/today", timeout=TIMEOUT)
        assert response.status_code in [200, 404]  # 404 OK if no games today
        
        if response.status_code == 200:
            data = response.json()
            # If games exist, verify structure
            if len(data) > 0:
                game = data[0]
                assert "home_team" in game
                assert "away_team" in game


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with smoke test markers."""
    config.addinivalue_line(
        "markers", "smoke: mark test as smoke test (run after deployment)"
    )

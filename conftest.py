"""
Pytest configuration and shared fixtures for QuantBet testing.

This file provides:
- Test database fixtures (SQLite in-memory and file-based)
- FastAPI test client with authentication
- Mock data factories
- Environment configuration for tests
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient


# ============================================================================
# Path Setup
# ============================================================================

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ============================================================================
# Environment Configuration
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure environment variables for testing."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["QUANTBET_API_KEY"] = "test_api_key_12345"
    os.environ["ADMIN_USER"] = "test_admin"
    os.environ["ADMIN_PASS"] = "test_password"
    os.environ["DATABASE_URL"] = "sqlite:///test.db"
    
    yield
    
    # Cleanup test database if exists
    test_db = Path("test.db")
    if test_db.exists():
        test_db.unlink()


# ============================================================================
# FastAPI Test Client
# ============================================================================

@pytest.fixture
def api_client() -> Generator[TestClient, None, None]:
    """
    FastAPI test client without authentication.
    
    Usage:
        def test_endpoint(api_client):
            response = api_client.get("/health")
            assert response.status_code == 200
    """
    from src.api.app import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def authenticated_client() -> Generator[TestClient, None, None]:
    """
    FastAPI test client with API key authentication.
    
    Usage:
        def test_protected_endpoint(authenticated_client):
            response = authenticated_client.get("/predictions/today")
            assert response.status_code == 200
    """
    from src.api.app import app
    
    with TestClient(app) as client:
        # Add API key header
        client.headers.update({"X-API-Key": "test_api_key_12345"})
        yield client


@pytest.fixture
def admin_client() -> Generator[TestClient, None, None]:
    """
    FastAPI test client with admin authentication.
    
    Usage:
        def test_admin_endpoint(admin_client):
            response = admin_client.get("/admin/stats")
            assert response.status_code == 200
    """
    from src.api.app import app
    
    with TestClient(app) as client:
        # Add Basic Auth
        client.headers.update({
            "Authorization": "Basic dGVzdF9hZG1pbjp0ZXN0X3Bhc3N3b3Jk"  # test_admin:test_password
        })
        yield client


# ============================================================================
# Test Data Factories
# ============================================================================

@pytest.fixture
def sample_game_data() -> Dict[str, Any]:
    """Sample NBL game data for testing."""
    return {
        "game_id": "test_game_001",
        "home_team": "MEL",
        "away_team": "SYD",
        "game_date": datetime.now().strftime("%Y-%m-%d"),
        "home_score": None,
        "away_score": None,
        "season": "2024-25"
    }


@pytest.fixture
def sample_prediction_data() -> Dict[str, Any]:
    """Sample prediction data for testing."""
    return {
        "game_id": "test_game_001",
        "predicted_home_prob": 0.65,
        "predicted_away_prob": 0.35,
        "confidence": 0.75,
        "model_agreement": 0.82,
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def sample_bet_data() -> Dict[str, Any]:
    """Sample bet data for testing."""
    return {
        "game_id": "test_game_001",
        "user_id": "test_user_001",
        "bet_type": "home_win",
        "odds": 1.85,
        "stake": 50.0,
        "predicted_prob": 0.65,
        "result": None,
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def sample_odds_data() -> Dict[str, Any]:
    """Sample live odds data for testing."""
    return {
        "event_id": "test_event_001",
        "home_team": "Melbourne United",
        "away_team": "Sydney Kings",
        "best_home_odds": 1.85,
        "best_away_odds": 2.10,
        "best_home_bookmaker": "Sportsbet",
        "best_away_bookmaker": "TAB",
        "commence_time": (datetime.now() + timedelta(hours=2)).isoformat()
    }


@pytest.fixture
def create_test_predictions(sample_prediction_data):
    """
    Factory fixture to create multiple test predictions.
    
    Usage:
        def test_calibration(create_test_predictions):
            predictions = create_test_predictions(count=10)
    """
    def _create(count: int = 5, **overrides):
        predictions = []
        for i in range(count):
            pred = sample_prediction_data.copy()
            pred["game_id"] = f"test_game_{i:03d}"
            pred.update(overrides)
            predictions.append(pred)
        return predictions
    
    return _create


@pytest.fixture
def create_test_bets(sample_bet_data):
    """
    Factory fixture to create multiple test bets.
    
    Usage:
        def test_bet_stats(create_test_bets):
            bets = create_test_bets(count=10, user_id="user123")
    """
    def _create(count: int = 5, **overrides):
        bets = []
        for i in range(count):
            bet = sample_bet_data.copy()
            bet["game_id"] = f"test_game_{i:03d}"
            bet.update(overrides)
            bets.append(bet)
        return bets
    
    return _create


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Temporary database file path for integration tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


# ============================================================================
# Mock API Responses
# ============================================================================

@pytest.fixture
def mock_odds_api_response(sample_odds_data):
    """Mock response from The Odds API."""
    return [sample_odds_data]


@pytest.fixture
def mock_nbl_schedule_response():
    """Mock response from NBL schedule scraper."""
    return [
        {
            "game_id": "nbl_2024_001",
            "home_team": "Melbourne United",
            "away_team": "Sydney Kings",
            "date_str": "2024-01-15",
            "time_str": "19:30",
            "venue": "John Cain Arena"
        }
    ]


# ============================================================================
# Performance Testing Helpers
# ============================================================================

@pytest.fixture
def benchmark_timer():
    """
    Timer for performance benchmarking.
    
    Usage:
        def test_endpoint_performance(api_client, benchmark_timer):
            with benchmark_timer(max_ms=300) as timer:
                response = api_client.get("/analytics/calibration")
            assert timer.elapsed_ms < 300
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def _timer(max_ms: float = None):
        class Timer:
            def __init__(self):
                self.start = None
                self.end = None
                self.elapsed_ms = None
        
        timer = Timer()
        timer.start = time.time()
        
        yield timer
        
        timer.end = time.time()
        timer.elapsed_ms = (timer.end - timer.start) * 1000
        
        if max_ms and timer.elapsed_ms > max_ms:
            pytest.fail(f"Benchmark failed: {timer.elapsed_ms:.2f}ms > {max_ms}ms")
    
    return _timer


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "smoke: mark test as smoke test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

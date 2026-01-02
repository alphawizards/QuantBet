"""
E2E test configuration and fixtures.

Provides browser, page, and URL fixtures for E2E tests.
"""

import pytest
from playwright.sync_api import Page, Browser


@pytest.fixture(scope="session")
def browser_context_args():
    """Browser context configuration."""
    return {
        "viewport": {"width": 1920, "height": 1080},
        "ignore_https_errors": True,
        "user_agent": "QuantBet-E2E-Tests/1.0"
    }


@pytest.fixture
def api_url():
    """API base URL."""
    return "http://localhost:8000"


@pytest.fixture
def dashboard_url():
    """Dashboard base URL."""
    return "http://localhost:3000"

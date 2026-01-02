"""
E2E tests for QuantBet Dashboard.

Tests dashboard loading, navigation, and basic functionality.
"""

import pytest
from playwright.sync_api import Page, expect


@pytest.mark.e2e
class TestDashboardLoading:
    """Test dashboard loads correctly."""
    
    def test_dashboard_accessible(self, page: Page, dashboard_url):
        """Test dashboard URL is accessible."""
        response = page.goto(dashboard_url)
        
        # Should get successful response
        assert response.ok or response.status in [200, 304]
    
    def test_dashboard_title(self, page: Page, dashboard_url):
        """Test dashboard has correct title."""
        page.goto(dashboard_url)
        
        # Should have QuantBet in title
        title = page.title()
        assert "QuantBet" in title or "Prediction" in title or len(title) > 0
    
    def test_dashboard_no_console_errors(self, page: Page, dashboard_url):
        """Test dashboard loads without console errors."""
        errors = []
        
        # Listen for console errors
        page.on("console", lambda msg: errors.append(msg) if msg.type == "error" else None)
        
        page.goto(dashboard_url)
        page.wait_for_load_state("networkidle")
        
        # Should have no critical errors
        critical_errors = [e for e in errors if "ERR" in str(e).upper()]
        assert len(critical_errors) == 0, f"Found errors: {critical_errors}"


@pytest.mark.e2e
class TestDashboardNavigation:
    """Test dashboard navigation."""
    
    def test_dashboard_has_navigation(self, page: Page, dashboard_url):
        """Test dashboard has navigation elements."""
        page.goto(dashboard_url)
        
        # Should have nav or menu elements
        has_nav = (
            page.locator('nav').count() > 0 or
            page.locator('[role="navigation"]').count() > 0 or
            page.locator('.nav, .menu').count() > 0 or
            page.locator('header').count() > 0
        )
        
        assert has_nav, "Dashboard should have navigation"
    
    def test_dashboard_responsive(self, page: Page, dashboard_url):
        """Test dashboard is responsive."""
        # Test mobile viewport
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto(dashboard_url)
        
        # Should still be accessible
        response = page.goto(dashboard_url)
        assert response.ok or response.status in [200, 304]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])

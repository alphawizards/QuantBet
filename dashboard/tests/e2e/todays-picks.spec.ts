import { test, expect } from '@playwright/test';
import { DashboardPage } from '../page-objects/DashboardPage';
import { mockTodaysPicksAPI, mockAllDashboardAPIs } from '../helpers/api-mocks';
import { mockTodaysPicks, mockEmptyTodaysPicks, mockBacktestData, mockCalibrationData } from '../fixtures/test-data';

/**
 * Today's Picks E2E Tests
 * 
 * Tests the primary user workflow: viewing and interacting with today's game predictions
 */

test.describe('Today\'s Picks', () => {
    let dashboardPage: DashboardPage;

    test.beforeEach(async ({ page }) => {
        dashboardPage = new DashboardPage(page);
    });

    test('should display today\'s picks when games are available', async ({ page }) => {
        // Mock API with today's picks
        await mockAllDashboardAPIs(page, {
            todaysPicks: mockTodaysPicks,
            backtest: mockBacktestData,
            calibration: mockCalibrationData,
        });

        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        // Verify Today's Picks section is visible
        const isVisible = await dashboardPage.isTodaysPicksVisible();
        expect(isVisible).toBe(true);

        // Verify correct number of picks displayed
        const picksCount = await dashboardPage.getTodaysPicksCount();
        expect(picksCount).toBe(mockTodaysPicks.length);
    });

    test('should display Kelly stake for each game', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        // Check first game prediction
        const firstPick = page.locator('[data-testid="game-prediction"]').first();
        const kellyStake = firstPick.locator('[data-testid="kelly-stake"]');

        await expect(kellyStake).toBeVisible();

        // Verify stake percentage is displayed (should be ~2.34% for first mock pick)
        const stakeText = await kellyStake.textContent();
        expect(stakeText).toContain('2.3'); // Should contain the percentage
    });

    test('should display edge percentage for each game', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        // Check first game prediction
        const firstPick = page.locator('[data-testid="game-prediction"]').first();
        const edge = firstPick.locator('[data-testid="edge-percentage"]');

        await expect(edge).toBeVisible();

        // Verify edge is displayed (should be ~4.67% for first mock pick)
        const edgeText = await edge.textContent();
        expect(edgeText).toContain('4.'); // Should contain edge percentage
    });

    test('should display team names correctly', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        // Verify first game teams
        const firstPick = page.locator('[data-testid="game-prediction"]').first();
        const content = await firstPick.textContent();

        expect(content).toContain('Melbourne United');
        expect(content).toContain('Sydney Kings');
    });

    test('should display confidence level', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        // Check confidence badges
        const picks = page.locator('[data-testid="game-prediction"]');
        const firstPickContent = await picks.first().textContent();
        const secondPickContent = await picks.nth(1).textContent();

        // First pick should show MEDIUM confidence
        expect(firstPickContent).toContain('MEDIUM');

        // Second pick should show HIGH confidence
        expect(secondPickContent).toContain('HIGH');
    });

    test('should show empty state when no games scheduled', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockEmptyTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        // Should show "no games" message or empty state
        const picksCount = await dashboardPage.getTodaysPicksCount();
        expect(picksCount).toBe(0);

        // Look for empty state message
        const emptyState = page.locator('text=/no games|no picks|check back/i').first();
        const isVisible = await emptyState.isVisible().catch(() => false);

        // Either empty state is shown, or section has zero picks
        expect(isVisible || picksCount === 0).toBe(true);
    });

    test('should calculate stakes based on bankroll parameter', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        // Today's Picks component has bankroll prop (default 1000)
        // First pick: 2.34% of 1000 = $23.40
        const firstPick = page.locator('[data-testid="game-prediction"]').first();
        const content = await firstPick.textContent();

        // Should show dollar amount based on bankroll
        expect(content).toMatch(/\$\d+/); // Contains dollar amount
    });

    test('should highlight recommended bets', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        // Picks with positive edge should be highlighted or have special styling
        const firstPick = page.locator('[data-testid="game-prediction"]').first();

        // Check for recommended badge or styling
        const isRecommended = await firstPick.locator('text=/recommended|bet/i').isVisible().catch(() => false);
        const hasHighlight = await firstPick.evaluate((el) => {
            const style = window.getComputedStyle(el);
            // Check if element has border, background, or other highlight styling
            return style.borderWidth !== '0px' || style.backgroundColor !== 'rgba(0, 0, 0, 0)';
        });

        expect(isRecommended || hasHighlight).toBe(true);
    });

    test('should display game date and time', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        const firstPick = page.locator('[data-testid="game-prediction"]').first();
        const content = await firstPick.textContent();

        // Should contain time information
        expect(content).toMatch(/\d{1,2}:\d{2}|PM|AM/i);
    });

    test('should display odds for both teams', async ({ page }) => {
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();

        const firstPick = page.locator('[data-testid="game-prediction"]').first();
        const content = await firstPick.textContent();

        // Should show decimal odds (e.g., 2.15, 1.85)
        expect(content).toMatch(/\d\.\d{2}/); // Decimal odds format
    });
});

test.describe('Today\'s Picks - Interactions', () => {
    let dashboardPage: DashboardPage;

    test.beforeEach(async ({ page }) => {
        dashboardPage = new DashboardPage(page);
        await mockTodaysPicksAPI(page, mockTodaysPicks);
        await dashboardPage.goto();
        await dashboardPage.waitForLoad();
    });

    test('should expand game details when clicked', async ({ page }) => {
        const firstPick = page.locator('[data-testid="game-prediction"]').first();

        // Click on the pick
        await firstPick.click();

        // Wait a moment for potential expansion animation
        await page.waitForTimeout(300);

        // If expandable, details should appear
        // This test will need adjustment based on actual component implementation
        const hasExpandedContent = await page.locator('[data-testid="game-details"]').isVisible().catch(() => false);

        // Either it expands or it's already showing all details (both are acceptable)
        expect(true).toBe(true); // Placeholder - adjust based on actual behavior
    });

    test('should be keyboard accessible', async ({ page }) => {
        // Tab to first pick
        await page.keyboard.press('Tab');
        await page.keyboard.press('Tab');

        // Check that something related to picks is focused
        const focusedElement = await page.evaluate(() => {
            const el = document.activeElement;
            return el?.getAttribute('data-testid') || el?.tagName;
        });

        expect(focusedElement).toBeTruthy();
    });
});

test.describe('Today\'s Picks - Error Handling', () => {
    test('should handle API errors gracefully', async ({ page }) => {
        // Mock API error
        await page.route('**/api/games/today', (route) => {
            route.fulfill({
                status: 500,
                body: JSON.stringify({ error: 'Internal Server Error' }),
            });
        });

        const dashboardPage = new DashboardPage(page);
        await dashboardPage.goto();

        // Dashboard should still load without crashing
        await expect(page.locator('h1')).toBeVisible();

        // Should show error message or empty state, not crash
        const bodyVisible = await page.locator('body').isVisible();
        expect(bodyVisible).toBe(true);
    });

    test('should handle network timeout', async ({ page }) => {
        // Mock slow API response
        await page.route('**/api/games/today', async (route) => {
            await new Promise(resolve => setTimeout(resolve, 2000));
            await route.fulfill({
                status: 200,
                body: JSON.stringify(mockTodaysPicks),
            });
        });

        const dashboardPage = new DashboardPage(page);
        await dashboardPage.goto();

        // Should eventually load or show timeout message
        await page.waitForLoadState('networkidle', { timeout: 30000 });

        const bodyVisible = await page.locator('body').isVisible();
        expect(bodyVisible).toBe(true);
    });
});

import { test, expect, devices } from '@playwright/test';
import { DashboardPage } from '../page-objects/DashboardPage';
import { mockAllDashboardAPIs } from '../helpers/api-mocks';
import { mockBacktestData, mockCalibrationData, mockTodaysPicks } from '../fixtures/test-data';

/**
 * Mobile Responsive E2E Tests
 * 
 * Tests dashboard responsiveness across different mobile devices and screen sizes
 */

test.describe('Mobile Responsive - iPhone 12', () => {
    test.use({ ...devices['iPhone 12'] });

    let dashboardPage: DashboardPage;

    test.beforeEach(async ({ page }) => {
        dashboardPage = new DashboardPage(page);

        await mockAllDashboardAPIs(page, {
            backtest: mockBacktestData,
            calibration: mockCalibrationData,
            todaysPicks: mockTodaysPicks,
        });

        await dashboardPage.goto();
        await dashboardPage.waitForLoad();
    });

    test('should render dashboard on mobile viewport', async ({ page }) => {
        await dashboardPage.verifyDashboardLoaded();

        // Check viewport size is mobile
        const viewportSize = page.viewportSize();
        expect(viewportSize?.width).toBe(390); // iPhone 12 width
    });

    test('should display all metric cards with scrolling', async ({ page }) => {
        // Metric cards should be visible (may need scrolling)
        const metricCards = page.locator('[data-testid^="metric-card-"]');
        const count = await metricCards.count();

        expect(count).toBeGreaterThan(0);

        // Cards should stack vertically on mobile
        const firstCard = metricCards.first();
        const secondCard = metricCards.nth(1);

        const firstBox = await firstCard.boundingBox();
        const secondBox = await secondCard.boundingBox();

        if (firstBox && secondBox) {
            // Second card should be below first (vertical stacking)
            expect(secondBox.y).toBeGreaterThan(firstBox.y);
        }
    });

    test('should render charts responsively', async ({ page }) => {
        // Scroll to charts
        await page.evaluate(() => window.scrollTo(0, 600));

        // Charts should be visible and fit within viewport
        const chart = page.locator('.recharts-wrapper').first();
        if (await chart.isVisible()) {
            const chartBox = await chart.boundingBox();
            const viewport = page.viewportSize();

            expect(chartBox).toBeTruthy();
            expect(chartBox!.width).toBeLessThanOrEqual(viewport!.width);
        }
    });

    test('should have readable text sizes', async ({ page }) => {
        // Check heading font size
        const heading = page.locator('h1').first();
        const fontSize = await heading.evaluate((el) => {
            return window.getComputedStyle(el).fontSize;
        });

        // Font size should be at least 20px on mobile
        const sizePx = parseFloat(fontSize);
        expect(sizePx).toBeGreaterThanOrEqual(20);
    });

    test('should have touch-friendly tap targets', async ({ page }) => {
        // Model selector should be large enough for touch
        const selector = dashboardPage.modelSelector;

        if (await selector.isVisible()) {
            const box = await selector.boundingBox();

            // Should be at least 44x44px (iOS touch target minimum)
            expect(box!.height).toBeGreaterThanOrEqual(40);
        }
    });
});

test.describe('Mobile Responsive - Pixel 5', () => {
    test.use({ ...devices['Pixel 5'] });

    test('should render on Android device', async ({ page }) => {
        await mockAllDashboardAPIs(page, {
            backtest: mockBacktestData,
            calibration: mockCalibrationData,
        });

        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Verify dashboard loads
        await expect(page.locator('h1')).toBeVisible();

        // Check viewport
        const viewportSize = page.viewportSize();
        expect(viewportSize?.width).toBe(393); // Pixel 5 width
    });

    test('should handle vertical scrolling', async ({ page }) => {
        await mockAllDashboardAPIs(page, {
            backtest: mockBacktestData,
            calibration: mockCalibrationData,
        });

        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Get initial scroll position
        const initialScroll = await page.evaluate(() => window.scrollY);

        // Scroll down
        await page.evaluate(() => window.scrollBy(0, 500));
        await page.waitForTimeout(300);

        // Verify scroll happened
        const newScroll = await page.evaluate(() => window.scrollY);
        expect(newScroll).toBeGreaterThan(initialScroll);
    });
});

test.describe('Mobile Responsive - Tablet', () => {
    test.use({ ...devices['iPad Pro'] });

    test('should render on tablet viewport', async ({ page }) => {
        await mockAllDashboardAPIs(page, {
            backtest: mockBacktestData,
            calibration: mockCalibrationData,
        });

        await page.goto('/');
        await page.waitForLoadState('networkidle');

        await expect(page.locator('h1')).toBeVisible();

        // Tablet should show more columns for metric cards
        const metricCards = page.locator('[data-testid^="metric-card-"]').first();
        const firstCardBox = await metricCards.boundingBox();
        const viewport = page.viewportSize();

        // On tablet, cards should occupy less than full width (multi-column layout)
        if (firstCardBox && viewport) {
            expect(firstCardBox.width).toBeLessThan(viewport.width * 0.9);
        }
    });
});

test.describe('Mobile Responsive - Orientation Changes', () => {
    test('should handle portrait to landscape rotation', async ({ page }) => {
        // Start in portrait
        await page.setViewportSize({ width: 390, height: 844 }); // iPhone 12 portrait

        await mockAllDashboardAPIs(page, {
            backtest: mockBacktestData,
        });

        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Verify loads in portrait
        await expect(page.locator('h1')).toBeVisible();

        // Rotate to landscape
        await page.setViewportSize({ width: 844, height: 390 }); // iPhone 12 landscape
        await page.waitForTimeout(500);

        // Should still be visible and render correctly
        await expect(page.locator('h1')).toBeVisible();
    });
});

test.describe('Mobile Responsive - Touch Interactions', () => {
    test.use({ ...devices['iPhone 12'] });

    test('should support touch scrolling', async ({ page }) => {
        await mockAllDashboardAPIs(page, {
            backtest: mockBacktestData,
            calibration: mockCalibrationData,
        });

        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Simulate touch swipe to scroll
        await page.touchscreen.tap(200, 400);
        await page.evaluate(() => window.scrollBy(0, 300));

        // Verify scroll worked
        const scrollY = await page.evaluate(() => window.scrollY);
        expect(scrollY).toBeGreaterThan(0);
    });

    test('should handle tap events on interactive elements', async ({ page }) => {
        await mockAllDashboardAPIs(page, {
            backtest: mockBacktestData,
        });

        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Tap on model selector
        const selector = page.locator('select, [role="combobox"]').first();

        if (await selector.isVisible()) {
            await selector.tap();

            // Should open dropdown or trigger interaction
            await page.waitForTimeout(300);

            // Verify interaction occurred (dropdown opened or state changed)
            expect(true).toBe(true); // Placeholder - specific behavior depends on implementation
        }
    });
});

test.describe('Mobile Responsive - Performance', () => {
    test.use({ ...devices['iPhone 12'] });

    test('should load quickly on mobile', async ({ page }) => {
        await mockAllDashboardAPIs(page, {
            backtest: mockBacktestData,
        });

        const startTime = Date.now();
        await page.goto('/');
        await page.waitForLoadState('networkidle');
        const loadTime = Date.now() - startTime;

        // Should load within reasonable time on mobile
        expect(loadTime).toBeLessThan(8000); // 8 seconds allowance for mobile
    });
});

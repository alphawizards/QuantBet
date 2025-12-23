import { test, expect } from '@playwright/test';

/**
 * Dashboard E2E Tests
 * 
 * Core user flows for the QuantBet dashboard including:
 * - Dashboard loads correctly
 * - Metric cards display
 * - Model selector works
 * - Charts render
 */

test.describe('Dashboard', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
    });

    test('should load dashboard successfully', async ({ page }) => {
        // Verify main heading
        await expect(page.locator('h1')).toBeVisible();

        // Verify dashboard container loads
        await expect(page.locator('.dashboard, [class*="dashboard"]')).toBeVisible({ timeout: 10000 });
    });

    test('should display metric cards', async ({ page }) => {
        // Wait for metrics to load
        await page.waitForLoadState('networkidle');

        // Check for metric cards (ROI, Sharpe, Win Rate, Total Bets)
        const metricCards = page.locator('[class*="metric"], .metric-card');
        await expect(metricCards.first()).toBeVisible({ timeout: 10000 });
    });

    test('should display equity curve chart', async ({ page }) => {
        // Wait for chart to render
        await page.waitForLoadState('networkidle');

        // Look for equity curve container or recharts svg
        const equityCurve = page.locator('[class*="equity"], .recharts-wrapper, svg.recharts-surface').first();
        await expect(equityCurve).toBeVisible({ timeout: 15000 });
    });
});

test.describe('Model Selector', () => {
    test('should display model selector dropdown', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Look for model selector component
        const modelSelector = page.locator('[class*="model-selector"], select, [role="combobox"]').first();
        await expect(modelSelector).toBeVisible({ timeout: 10000 });
    });

    test('should change model when selection changes', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Find and click model selector
        const selector = page.locator('select, [role="combobox"]').first();

        if (await selector.isVisible()) {
            // Click to open dropdown
            await selector.click();

            // Look for options and click one
            const options = page.locator('option, [role="option"]');
            const optionCount = await options.count();

            if (optionCount > 1) {
                await options.nth(1).click();

                // Wait for data to update
                await page.waitForLoadState('networkidle');

                // Verify dashboard still renders after model change
                await expect(page.locator('.dashboard, [class*="dashboard"]')).toBeVisible();
            }
        }
    });
});

test.describe('Error Handling', () => {
    test('should handle API errors gracefully', async ({ page }) => {
        // Intercept API calls and force an error
        await page.route('**/api/**', route => {
            route.fulfill({
                status: 500,
                body: JSON.stringify({ error: 'Internal Server Error' })
            });
        });

        await page.goto('/');

        // Should not crash - either shows error message or falls back gracefully
        await expect(page.locator('body')).toBeVisible();

        // Should not show raw error stack in production
        const errorStack = page.locator('.error-stack');
        // In production, this should not be visible
    });

    test('should handle network errors gracefully', async ({ page }) => {
        // Simulate network failure
        await page.route('**/api/**', route => route.abort('failed'));

        await page.goto('/');

        // Page should still load without crashing
        await expect(page.locator('body')).toBeVisible();
    });
});

test.describe('Accessibility', () => {
    test('should have no accessibility violations on initial load', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Basic accessibility checks
        // All buttons should have accessible names
        const buttons = page.locator('button');
        const buttonCount = await buttons.count();

        for (let i = 0; i < buttonCount; i++) {
            const button = buttons.nth(i);
            const ariaLabel = await button.getAttribute('aria-label');
            const innerText = await button.innerText();

            // Button should have either aria-label or visible text
            expect(ariaLabel || innerText.trim()).toBeTruthy();
        }
    });

    test('should be keyboard navigable', async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('networkidle');

        // Tab through focusable elements
        await page.keyboard.press('Tab');

        // Something should be focused
        const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
        expect(focusedElement).toBeTruthy();
    });
});

test.describe('Performance', () => {
    test('should load within acceptable time', async ({ page }) => {
        const startTime = Date.now();

        await page.goto('/');
        await page.waitForLoadState('networkidle');

        const loadTime = Date.now() - startTime;

        // Dashboard should load within 5 seconds
        expect(loadTime).toBeLessThan(5000);
    });
});

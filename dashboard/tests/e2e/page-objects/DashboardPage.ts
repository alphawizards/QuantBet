/**
 * Dashboard Page Object Model
 * 
 * Provides a clean interface for interacting with the dashboard in E2E tests
 */

import { Page, Locator, expect } from '@playwright/test';

export class DashboardPage {
    readonly page: Page;
    readonly heading: Locator;
    readonly modelSelector: Locator;
    readonly loadingIndicator: Locator;

    constructor(page: Page) {
        this.page = page;
        this.heading = page.locator('h1');
        this.modelSelector = page.locator('select, [role="combobox"]').first();
        this.loadingIndicator = page.locator('text=Loading dashboard...');
    }

    /**
     * Navigate to dashboard
     */
    async goto() {
        await this.page.goto('/');
    }

    /**
     * Wait for dashboard to finish loading
     */
    async waitForLoad() {
        await this.page.waitForLoadState('networkidle');
        // Wait for loading indicator to disappear if it exists
        const hasLoadingIndicator = await this.loadingIndicator.isVisible().catch(() => false);
        if (hasLoadingIndicator) {
            await this.loadingIndicator.waitFor({ state: 'hidden', timeout: 15000 });
        }
    }

    /**
     * Select a betting model
     */
    async selectModel(modelId: string) {
        await this.modelSelector.click();
        await this.page.locator(`option[value="${modelId}"], [role="option"][data-value="${modelId}"]`).click();
        await this.waitForLoad();
    }

    /**
     * Get metric card value by title
     */
    async getMetricCardValue(metricTitle: string): Promise<string> {
        const metricCard = this.page.locator(`[data-testid="metric-card-${metricTitle.toLowerCase().replace(/\s+/g, '-')}"]`);
        const value = await metricCard.locator('[data-testid="metric-value"]').first().textContent();
        return value || '';
    }

    /**
     * Wait for all charts to render
     */
    async waitForChartsToRender() {
        await this.page.waitForSelector('.recharts-wrapper, svg.recharts-surface', { timeout: 15000 });
    }

    /**
     * Check if Today's Picks section is visible
     */
    async isTodaysPicksVisible(): Promise<boolean> {
        return await this.page.locator('[data-testid="todays-picks"]').isVisible().catch(() => false);
    }

    /**
     * Get number of today's picks displayed
     */
    async getTodaysPicksCount(): Promise<number> {
        const picks = this.page.locator('[data-testid="game-prediction"]');
        return await picks.count();
    }

    /**
     * Get calibration chart brier score
     */
    async getCalibrationBrierScore(): Promise<string> {
        const brierScore = this.page.locator('[data-testid="brier-score"]');
        const text = await brierScore.textContent();
        return text || '';
    }

    /**
     * Check if calibration chart is visible
     */
    async isCalibrationChartVisible(): Promise<boolean> {
        return await this.page.locator('[data-testid="calibration-chart"]').isVisible().catch(() => false);
    }

    /**
     * Get all metric card titles
     */
    async getAllMetricTitles(): Promise<string[]> {
        const cards = this.page.locator('[data-testid^="metric-card-"]');
        const count = await cards.count();
        const titles: string[] = [];

        for (let i = 0; i < count; i++) {
            const testId = await cards.nth(i).getAttribute('data-testid');
            if (testId) {
                // Extract title from test ID (e.g., "metric-card-roi" -> "ROI")
                const title = testId.replace('metric-card-', '').replace(/-/g, ' ');
                titles.push(title);
            }
        }

        return titles;
    }

    /**
     * Verify dashboard loaded successfully
     */
    async verifyDashboardLoaded() {
        await expect(this.heading).toBeVisible();
        await expect(this.heading).toContainText('QuantBet');
        await this.waitForLoad();
    }
}

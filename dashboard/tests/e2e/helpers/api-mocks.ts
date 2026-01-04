/**
 * API Mocking Utilities for E2E Tests
 * 
 * Provides utilities to mock API responses in Playwright tests
 */

import { Page } from '@playwright/test';
import type { BacktestResult, CalibrationBin, UpcomingGame, TeamElo, ModelComparison, StrategyComparisonData } from '../../../src/types/api';

/**
 * Mock backtest API endpoint
 */
export async function mockBacktestAPI(page: Page, data: BacktestResult) {
    await page.route('**/api/backtest/**', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(data),
        });
    });
}

/**
 * Mock calibration data API endpoint
 */
export async function mockCalibrationAPI(page: Page, data: CalibrationBin[]) {
    await page.route('**/api/analytics/calibration/**', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(data),
        });
    });
}

/**
 * Mock today's picks API endpoint
 */
export async function mockTodaysPicksAPI(page: Page, picks: UpcomingGame[]) {
    await page.route('**/api/games/today', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(picks),
        });
    });
}

/**
 * Mock ELO rankings API endpoint
 */
export async function mockEloRankingsAPI(page: Page, rankings: TeamElo[]) {
    await page.route('**/api/rankings/elo', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(rankings),
        });
    });
}

/**
 * Mock model comparison API endpoint
 */
export async function mockModelComparisonAPI(page: Page, data: ModelComparison[]) {
    await page.route('**/api/models/comparison/**', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(data),
        });
    });
}

/**
 * Mock strategy comparison API endpoint
 */
export async function mockStrategyComparisonAPI(page: Page, data: StrategyComparisonData) {
    await page.route('**/api/strategies/comparison/**', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(data),
        });
    });
}

/**
 * Mock all dashboard APIs with default data
 */
export async function mockAllDashboardAPIs(
    page: Page,
    options: {
        backtest?: BacktestResult;
        calibration?: CalibrationBin[];
        todaysPicks?: GamePrediction[];
        eloRankings?: TeamElo[];
        modelComparison?: ModelComparison[];
        strategyComparison?: StrategyComparisonData;
    }
) {
    if (options.backtest) {
        await mockBacktestAPI(page, options.backtest);
    }
    if (options.calibration) {
        await mockCalibrationAPI(page, options.calibration);
    }
    if (options.todaysPicks) {
        await mockTodaysPicksAPI(page, options.todaysPicks);
    }
    if (options.eloRankings) {
        await mockEloRankingsAPI(page, options.eloRankings);
    }
    if (options.modelComparison) {
        await mockModelComparisonAPI(page, options.modelComparison);
    }
    if (options.strategyComparison) {
        await mockStrategyComparisonAPI(page, options.strategyComparison);
    }
}

/**
 * Mock API error responses
 */
export async function mockAPIError(page: Page, statusCode: number = 500) {
    await page.route('**/api/**', async (route) => {
        await route.fulfill({
            status: statusCode,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Internal Server Error' }),
        });
    });
}

/**
 * Mock network failure (connection refused)
 */
export async function mockNetworkFailure(page: Page) {
    await page.route('**/api/**', (route) => route.abort('failed'));
}

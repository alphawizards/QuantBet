/**
 * Test Data Fixtures for E2E Tests
 * 
 * Provides consistent mock data for Playwright tests
 */

import type { BacktestResult, CalibrationBin, UpcomingGame, TeamElo } from '../../../src/types/api';

export const mockBacktestData: BacktestResult = {
    id: 'test-backtest-001',
    modelId: 'kelly',
    trainSeasons: ['2021-22', '2022-23'],
    testSeasons: ['2023-24'],
    stakingStrategy: 'Fractional Kelly (0.25)',
    metrics: {
        roi: 0.053,
        sharpeRatio: 1.82,
        sortinoRatio: 2.45,
        calmarRatio: 1.15,
        winRate: 0.547,
        profitLoss: 1247.50,
        totalBets: 156,
        totalStaked: 23550,
        maxDrawdownPct: 0.087,
        brierScore: 0.215,
        calibrationError: 0.042,
    },
    equity: [
        { date: '2023-10-01', bankroll: 10000, cumProfit: 0 },
        { date: '2023-10-15', bankroll: 10250, cumProfit: 250 },
        { date: '2023-11-01', bankroll: 10450, cumProfit: 450 },
        { date: '2023-11-15', bankroll: 10580, cumProfit: 580 },
        { date: '2023-12-01', bankroll: 10720, cumProfit: 720 },
        { date: '2023-12-15', bankroll: 10895, cumProfit: 895 },
        { date: '2024-01-01', bankroll: 11000, cumProfit: 1000 },
    ],
    bets: [
        {
            gameId: 'nbl-2023-001',
            homeTeam: 'Melbourne United',
            awayTeam: 'Sydney Kings',
            date: '2023-10-05',
            prediction: 0.58,
            odds: 2.15,
            stake: 150,
            won: true,
            profit: 172.50,
        },
        {
            gameId: 'nbl-2023-002',
            homeTeam: 'Perth Wildcats',
            awayTeam: 'Adelaide 36ers',
            date: '2023-10-08',
            prediction: 0.62,
            odds: 1.85,
            stake: 200,
            won: true,
            profit: 170,
        },
        {
            gameId: 'nbl-2023-003',
            homeTeam: 'Brisbane Bullets',
            awayTeam: 'Illawarra Hawks',
            date: '2023-10-12',
            prediction: 0.55,
            odds: 2.05,
            stake: 125,
            won: false,
            profit: -125,
        },
    ],
    createdAt: '2024-01-01T00:00:00Z',
};

export const mockCalibrationData: CalibrationBin[] = [
    { bin: '0.1', predicted: 0.1, actual: 0.12, count: 8 },
    { bin: '0.2', predicted: 0.2, actual: 0.19, count: 12 },
    { bin: '0.3', predicted: 0.3, actual: 0.31, count: 15 },
    { bin: '0.4', predicted: 0.4, actual: 0.38, count: 18 },
    { bin: '0.5', predicted: 0.5, actual: 0.51, count: 25 },
    { bin: '0.6', predicted: 0.6, actual: 0.59, count: 22 },
    { bin: '0.7', predicted: 0.7, actual: 0.72, count: 16 },
    { bin: '0.8', predicted: 0.8, actual: 0.78, count: 10 },
    { bin: '0.9', predicted: 0.9, actual: 0.91, count: 6 },
];

export const mockTodaysPicks: UpcomingGame[] = [
    {
        event_id: 'nbl-2024-001',
        home_team: 'Melbourne United',
        away_team: 'Sydney Kings',
        commence_time: '2026-01-03T19:00:00+10:00',
        predicted_home_prob: 0.58,
        predicted_home_prob_lower: 0.53,
        predicted_home_prob_upper: 0.63,
        uncertainty: 0.05,
        home_odds: 2.15,
        away_odds: 1.85,
        best_bookmaker: 'Sportsbet',
        home_edge: 0.0467,
        away_edge: -0.023,
        recommendation: 'BET_HOME',
        kelly_fraction: 0.0234,
        recommended_stake_pct: 0.0234,
        confidence: 'MEDIUM',
        top_factors: ['Home court advantage', 'Recent form'],
    },
    {
        event_id: 'nbl-2024-002',
        home_team: 'Perth Wildcats',
        away_team: 'Adelaide 36ers',
        commence_time: '2026-01-03T20:00:00+10:00',
        predicted_home_prob: 0.62,
        predicted_home_prob_lower: 0.57,
        predicted_home_prob_upper: 0.67,
        uncertainty: 0.05,
        home_odds: 1.92,
        away_odds: 2.10,
        best_bookmaker: 'TAB',
        home_edge: 0.053,
        away_edge: -0.032,
        recommendation: 'BET_HOME',
        kelly_fraction: 0.0318,
        recommended_stake_pct: 0.0318,
        confidence: 'HIGH',
        top_factors: ['Strong home record', 'Opponent injuries'],
    },
];

export const mockEloRankings: TeamElo[] = [
    { rank: 1, team: 'MEL', teamName: 'Melbourne United', elo: 1650, change: 12 },
    { rank: 2, team: 'SYD', teamName: 'Sydney Kings', elo: 1620, change: 5 },
    { rank: 3, team: 'PER', teamName: 'Perth Wildcats', elo: 1595, change: -3 },
    { rank: 4, team: 'TAS', teamName: 'Tasmania JackJumpers', elo: 1580, change: 8 },
    { rank: 5, team: 'ILL', teamName: 'Illawarra Hawks', elo: 1520, change: -7 },
];

export const mockEmptyTodaysPicks: UpcomingGame[] = [];

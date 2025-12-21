import axios from 'axios';
import type {
    BacktestMetrics,
    EquityPoint,
    CalibrationBin,
    BetRecord,
    ModelComparison,
    TeamElo,
    BacktestResult,
    DashboardSummary,
    StrategyEquity,
    StrategyComparisonData,
    STAKING_STRATEGIES,
} from '../types/api';

const API_BASE = '/api';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Dashboard summary
export async function getDashboardSummary(): Promise<DashboardSummary> {
    return {
        latestRoi: 0.082,
        latestSharpe: 1.45,
        totalBets: 142,
        winRate: 0.56,
        todayPredictions: 3,
        pendingBets: 2,
    };
}

// Generate equity curve with specified volatility and trend
function generateEquityCurve(
    startBankroll: number,
    trend: number,
    volatility: number,
    days: number
): EquityPoint[] {
    const equity: EquityPoint[] = [];
    let bankroll = startBankroll;
    const startDate = new Date('2024-01-01');

    for (let i = 0; i < days; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i * 3);
        const change = trend + (Math.random() - 0.5) * volatility;
        bankroll = Math.max(bankroll + change, startBankroll * 0.5);
        equity.push({
            date: date.toISOString().split('T')[0],
            bankroll: Math.round(bankroll * 100) / 100,
            cumProfit: Math.round((bankroll - startBankroll) * 100) / 100,
        });
    }
    return equity;
}

// Strategy comparison data for multi-line chart
export async function getStrategyComparison(
    modelId: string
): Promise<StrategyComparisonData> {
    const startBankroll = 1000;
    const days = 60;

    // Different characteristics per strategy
    const strategyParams: Record<string, { trend: number; volatility: number }> = {
        flat: { trend: 3, volatility: 30 },               // Steady, low variance
        full_kelly: { trend: 8, volatility: 80 },         // High growth, high variance
        fractional_kelly: { trend: 5, volatility: 35 },   // Moderate growth, low variance
        proportional: { trend: 4, volatility: 25 },       // Conservative
        mean_reversion: { trend: 6, volatility: 50 },     // Variable
    };

    // Adjust trends based on model effectiveness
    const modelMultipliers: Record<string, number> = {
        kelly: 1.0,
        poisson: 0.9,
        elo: 0.85,
        mean_reversion: 0.75,
        arbitrage: 1.3,
    };

    const multiplier = modelMultipliers[modelId] || 1.0;

    const strategies: StrategyEquity[] = [
        {
            strategyId: 'flat',
            name: 'Flat Staking',
            color: '#94a3b8',
            equity: generateEquityCurve(
                startBankroll,
                strategyParams.flat.trend * multiplier,
                strategyParams.flat.volatility,
                days
            ),
            finalRoi: 0.045 * multiplier,
            sharpe: 0.82 * multiplier,
            maxDrawdown: 0.065,
        },
        {
            strategyId: 'full_kelly',
            name: 'Full Kelly',
            color: '#ef4444',
            equity: generateEquityCurve(
                startBankroll,
                strategyParams.full_kelly.trend * multiplier,
                strategyParams.full_kelly.volatility,
                days
            ),
            finalRoi: 0.145 * multiplier,
            sharpe: 0.95 * multiplier,
            maxDrawdown: 0.185,
        },
        {
            strategyId: 'fractional_kelly',
            name: 'Fractional Kelly (25%)',
            color: '#3b82f6',
            equity: generateEquityCurve(
                startBankroll,
                strategyParams.fractional_kelly.trend * multiplier,
                strategyParams.fractional_kelly.volatility,
                days
            ),
            finalRoi: 0.082 * multiplier,
            sharpe: 1.45 * multiplier,
            maxDrawdown: 0.086,
        },
        {
            strategyId: 'proportional',
            name: 'Proportional (1%)',
            color: '#10b981',
            equity: generateEquityCurve(
                startBankroll,
                strategyParams.proportional.trend * multiplier,
                strategyParams.proportional.volatility,
                days
            ),
            finalRoi: 0.058 * multiplier,
            sharpe: 1.12 * multiplier,
            maxDrawdown: 0.072,
        },
        {
            strategyId: 'mean_reversion',
            name: 'Mean Reversion',
            color: '#8b5cf6',
            equity: generateEquityCurve(
                startBankroll,
                strategyParams.mean_reversion.trend * multiplier,
                strategyParams.mean_reversion.volatility,
                days
            ),
            finalRoi: 0.068 * multiplier,
            sharpe: 1.05 * multiplier,
            maxDrawdown: 0.098,
        },
    ];

    return {
        modelId,
        strategies,
    };
}

// Backtest results for a specific model
export async function getBacktestForModel(modelId: string): Promise<BacktestResult> {
    const strategyData = await getStrategyComparison(modelId);
    const kellyStrategy = strategyData.strategies.find(s => s.strategyId === 'fractional_kelly');

    const bets: BetRecord[] = Array.from({ length: 20 }, (_, i) => ({
        gameId: `${modelId}_2024_${String(i + 1).padStart(3, '0')}`,
        homeTeam: ['MEL', 'SYD', 'PER', 'BRI', 'ADL'][i % 5],
        awayTeam: ['SYD', 'PER', 'BRI', 'ADL', 'MEL'][i % 5],
        date: new Date(2024, 0, i + 1).toISOString().split('T')[0],
        prediction: 0.55 + Math.random() * 0.1,
        odds: 1.8 + Math.random() * 0.4,
        stake: 20 + Math.random() * 10,
        won: Math.random() > 0.44,
        profit: Math.random() > 0.44 ? 15 + Math.random() * 10 : -(20 + Math.random() * 10),
    }));

    const modelNames: Record<string, string> = {
        kelly: 'Kelly Criterion',
        poisson: 'Poisson Distribution',
        elo: 'ELO Rating',
        mean_reversion: 'Mean Reversion',
        arbitrage: 'Arbitrage',
    };

    return {
        id: `bt_${modelId}_001`,
        modelId,
        trainSeasons: ['2021-22', '2022-23', '2023-24'],
        testSeasons: ['2024-25'],
        stakingStrategy: 'Fractional Kelly (25%)',
        metrics: {
            roi: kellyStrategy?.finalRoi || 0.082,
            sharpeRatio: kellyStrategy?.sharpe || 1.45,
            sortinoRatio: 1.82,
            calmarRatio: 0.95,
            maxDrawdownPct: kellyStrategy?.maxDrawdown || 0.086,
            winRate: 0.56,
            totalBets: 142,
            totalStaked: 4250,
            profitLoss: 348.50,
            brierScore: 0.215,
            calibrationError: 0.032,
        },
        equity: kellyStrategy?.equity || [],
        bets,
        createdAt: new Date().toISOString(),
    };
}

// Calibration data
export async function getCalibrationData(): Promise<CalibrationBin[]> {
    return [
        { bin: '0.0-0.1', predicted: 0.05, actual: 0.04, count: 8 },
        { bin: '0.1-0.2', predicted: 0.15, actual: 0.13, count: 12 },
        { bin: '0.2-0.3', predicted: 0.25, actual: 0.28, count: 18 },
        { bin: '0.3-0.4', predicted: 0.35, actual: 0.34, count: 24 },
        { bin: '0.4-0.5', predicted: 0.45, actual: 0.47, count: 28 },
        { bin: '0.5-0.6', predicted: 0.55, actual: 0.54, count: 26 },
        { bin: '0.6-0.7', predicted: 0.65, actual: 0.63, count: 20 },
        { bin: '0.7-0.8', predicted: 0.75, actual: 0.72, count: 14 },
        { bin: '0.8-0.9', predicted: 0.85, actual: 0.88, count: 10 },
        { bin: '0.9-1.0', predicted: 0.95, actual: 0.92, count: 6 },
    ];
}

// Model comparison
export async function getModelComparison(): Promise<ModelComparison[]> {
    return [
        { name: 'XGBoost', roi: 0.078, sharpe: 1.35, brier: 0.218, winRate: 0.55 },
        { name: 'ELO', roi: 0.045, sharpe: 0.92, brier: 0.235, winRate: 0.52 },
        { name: 'Market', roi: -0.025, sharpe: -0.28, brier: 0.248, winRate: 0.48 },
        { name: 'Ensemble', roi: 0.082, sharpe: 1.45, brier: 0.215, winRate: 0.56 },
    ];
}

// Team ELO rankings
export async function getEloRankings(): Promise<TeamElo[]> {
    return [
        { rank: 1, team: 'MEL', teamName: 'Melbourne United', elo: 1625, change: 15 },
        { rank: 2, team: 'SYD', teamName: 'Sydney Kings', elo: 1598, change: -8 },
        { rank: 3, team: 'PER', teamName: 'Perth Wildcats', elo: 1572, change: 22 },
        { rank: 4, team: 'TAS', teamName: 'Tasmania JackJumpers', elo: 1545, change: 5 },
        { rank: 5, team: 'NZB', teamName: 'NZ Breakers', elo: 1520, change: -12 },
        { rank: 6, team: 'BRI', teamName: 'Brisbane Bullets', elo: 1498, change: 8 },
        { rank: 7, team: 'ADL', teamName: 'Adelaide 36ers', elo: 1475, change: -5 },
        { rank: 8, team: 'ILL', teamName: 'Illawarra Hawks', elo: 1452, change: -18 },
        { rank: 9, team: 'SEM', teamName: 'SE Melbourne Phoenix', elo: 1428, change: 3 },
        { rank: 10, team: 'CAI', teamName: 'Cairns Taipans', elo: 1385, change: -10 },
    ];
}

// Legacy function for backwards compatibility
export async function getLatestBacktest(): Promise<BacktestResult> {
    return getBacktestForModel('kelly');
}

export default api;

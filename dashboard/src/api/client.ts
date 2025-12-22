import axios from 'axios';
import type {
    EquityPoint,
    CalibrationBin,
    BetRecord,
    ModelComparison,
    TeamElo,
    BacktestResult,
    DashboardSummary,
    StrategyEquity,
    StrategyComparisonData,
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

// Cache for backtest results
let backtestCache: Record<string, unknown> | null = null;

async function loadBacktestData(): Promise<Record<string, unknown>> {
    if (backtestCache) return backtestCache;

    try {
        const response = await fetch('/data/backtest_results.json');
        if (!response.ok) throw new Error('Failed to load backtest data');
        backtestCache = await response.json();
        return backtestCache!;
    } catch (error) {
        console.warn('Using mock data - real backtest data not available:', error);
        return {};
    }
}

// Backtest results for a specific model - fetches from real data
export async function getBacktestForModel(modelId: string): Promise<BacktestResult> {
    const data = await loadBacktestData();
    const strategyData = await getStrategyComparison(modelId);
    const kellyStrategy = strategyData.strategies.find(s => s.strategyId === 'fractional_kelly');

    // Map model IDs to data keys
    const modelKeyMap: Record<string, string> = {
        kelly: 'kelly',
        poisson: 'poisson',
        elo: 'elo',
        mean_reversion: 'kelly', // fallback
        arbitrage: 'arbitrage',
    };

    const dataKey = modelKeyMap[modelId] || 'kelly';
    const modelData = (data[dataKey] || data.kelly || {}) as {
        metrics?: {
            roi?: number;
            sharpeRatio?: number;
            sortinoRatio?: number;
            calmarRatio?: number;
            maxDrawdownPct?: number;
            winRate?: number;
            totalBets?: number;
            totalStaked?: number;
            profitLoss?: number;
            brierScore?: number;
            calibrationError?: number;
        };
        equity?: EquityPoint[];
        recentBets?: BetRecord[];
    };

    // Use real metrics if available, otherwise fall back to strategy data
    const metrics = modelData.metrics || {};

    // Convert recentBets to the BetRecord format
    const rawBets = modelData.recentBets || [];
    const bets: BetRecord[] = rawBets.map((bet, i) => ({
        gameId: `${modelId}_2024_${String(i + 1).padStart(3, '0')}`,
        homeTeam: bet.homeTeam || 'MEL',
        awayTeam: bet.awayTeam || 'SYD',
        date: bet.date || new Date().toISOString().split('T')[0],
        prediction: bet.prediction || 0.55,
        odds: bet.odds || 1.9,
        stake: bet.stake || 25,
        won: bet.won || false,
        profit: bet.profit || 0,
    }));

    return {
        id: `bt_${modelId}_001`,
        modelId,
        trainSeasons: ['2021-22', '2022-23', '2023-24'],
        testSeasons: ['2024-25'],
        stakingStrategy: 'Fractional Kelly (25%)',
        metrics: {
            roi: metrics.roi ?? kellyStrategy?.finalRoi ?? 0.082,
            sharpeRatio: metrics.sharpeRatio ?? kellyStrategy?.sharpe ?? 1.45,
            sortinoRatio: metrics.sortinoRatio ?? 1.82,
            calmarRatio: metrics.calmarRatio ?? 0.95,
            maxDrawdownPct: metrics.maxDrawdownPct ?? kellyStrategy?.maxDrawdown ?? 0.086,
            winRate: metrics.winRate ?? 0.56,
            totalBets: metrics.totalBets ?? 142,
            totalStaked: metrics.totalStaked ?? 4250,
            profitLoss: metrics.profitLoss ?? 348.50,
            brierScore: metrics.brierScore ?? 0.215,
            calibrationError: metrics.calibrationError ?? 0.032,
        },
        equity: (modelData.equity as EquityPoint[]) || kellyStrategy?.equity || [],
        bets,
        createdAt: new Date().toISOString(),
    };
}

// Calibration data - fetches from real backtest results
export async function getCalibrationData(modelId: string): Promise<CalibrationBin[]> {
    const data = await loadBacktestData();

    const modelKeyMap: Record<string, string> = {
        kelly: 'kelly',
        poisson: 'poisson',
        elo: 'elo',
        mean_reversion: 'kelly',
        arbitrage: 'arbitrage',
    };

    const dataKey = modelKeyMap[modelId] || 'kelly';
    const modelData = (data[dataKey] || {}) as {
        calibration?: CalibrationBin[];
    };

    // Use real calibration if available
    if (modelData.calibration && modelData.calibration.length > 0) {
        return modelData.calibration;
    }

    // Fallback to synthetic calibration
    const baseCalibration: CalibrationBin[] = [
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

    return baseCalibration;
}

// Model comparison - varies based on selected model's staking strategy
export async function getModelComparison(modelId: string): Promise<ModelComparison[]> {
    // Base model performance data
    const baseModels: ModelComparison[] = [
        { name: 'XGBoost', roi: 0.078, sharpe: 1.35, brier: 0.218, winRate: 0.55 },
        { name: 'ELO', roi: 0.045, sharpe: 0.92, brier: 0.235, winRate: 0.52 },
        { name: 'Market', roi: -0.025, sharpe: -0.28, brier: 0.248, winRate: 0.48 },
        { name: 'Ensemble', roi: 0.082, sharpe: 1.45, brier: 0.215, winRate: 0.56 },
    ];

    // Apply model-specific multipliers to show relative performance
    const modelMultipliers: Record<string, number> = {
        kelly: 1.0,
        poisson: 0.92,
        elo: 0.85,
        mean_reversion: 0.78,
        arbitrage: 1.15,
    };

    const multiplier = modelMultipliers[modelId] || 1.0;

    return baseModels.map(model => ({
        ...model,
        roi: Math.round(model.roi * multiplier * 1000) / 1000,
        sharpe: Math.round(model.sharpe * multiplier * 100) / 100,
        winRate: Math.min(0.65, Math.round(model.winRate * (0.95 + multiplier * 0.05) * 100) / 100),
    }));
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

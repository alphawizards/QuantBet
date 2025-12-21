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
    // Mock data for development - replace with real API when ready
    return {
        latestRoi: 0.082,
        latestSharpe: 1.45,
        totalBets: 142,
        winRate: 0.56,
        todayPredictions: 3,
        pendingBets: 2,
    };
}

// Backtest results
export async function getLatestBacktest(): Promise<BacktestResult> {
    // Mock data - replace with real API
    const equity: EquityPoint[] = [];
    let bankroll = 1000;
    const startDate = new Date('2024-01-01');

    for (let i = 0; i < 60; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i * 3);
        const change = (Math.random() - 0.45) * 40;
        bankroll += change;
        equity.push({
            date: date.toISOString().split('T')[0],
            bankroll: Math.round(bankroll * 100) / 100,
            cumProfit: Math.round((bankroll - 1000) * 100) / 100,
        });
    }

    const bets: BetRecord[] = Array.from({ length: 20 }, (_, i) => ({
        gameId: `nbl_2024_${String(i + 1).padStart(3, '0')}`,
        homeTeam: ['MEL', 'SYD', 'PER', 'BRI', 'ADL'][i % 5],
        awayTeam: ['SYD', 'PER', 'BRI', 'ADL', 'MEL'][i % 5],
        date: new Date(2024, 0, i + 1).toISOString().split('T')[0],
        prediction: 0.55 + Math.random() * 0.1,
        odds: 1.8 + Math.random() * 0.4,
        stake: 20 + Math.random() * 10,
        won: Math.random() > 0.44,
        profit: Math.random() > 0.44 ? 15 + Math.random() * 10 : -(20 + Math.random() * 10),
    }));

    return {
        id: 'bt_001',
        trainSeasons: ['2021-22', '2022-23', '2023-24'],
        testSeasons: ['2024-25'],
        stakingStrategy: 'Fractional Kelly (25%)',
        metrics: {
            roi: 0.082,
            sharpeRatio: 1.45,
            sortinoRatio: 1.82,
            calmarRatio: 0.95,
            maxDrawdownPct: 0.086,
            winRate: 0.56,
            totalBets: 142,
            totalStaked: 4250,
            profitLoss: 348.50,
            brierScore: 0.215,
            calibrationError: 0.032,
        },
        equity,
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

export default api;

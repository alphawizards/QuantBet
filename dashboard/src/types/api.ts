// API Types for QuantBet Dashboard

export interface BacktestMetrics {
    roi: number;
    sharpeRatio: number;
    sortinoRatio: number;
    calmarRatio: number;
    maxDrawdownPct: number;
    winRate: number;
    totalBets: number;
    totalStaked: number;
    profitLoss: number;
    brierScore: number;
    calibrationError: number;
}

export interface EquityPoint {
    date: string;
    bankroll: number;
    cumProfit: number;
}

export interface CalibrationBin {
    bin: string;
    predicted: number;
    actual: number;
    count: number;
}

export interface BetRecord {
    gameId: string;
    homeTeam: string;
    awayTeam: string;
    date: string;
    prediction: number;
    odds: number;
    stake: number;
    won: boolean;
    profit: number;
}

export interface ModelComparison {
    name: string;
    roi: number;
    sharpe: number;
    brier: number;
    winRate: number;
}

export interface TeamElo {
    rank: number;
    team: string;
    teamName: string;
    elo: number;
    change?: number;
}

export interface BacktestResult {
    id: string;
    trainSeasons: string[];
    testSeasons: string[];
    stakingStrategy: string;
    metrics: BacktestMetrics;
    equity: EquityPoint[];
    bets: BetRecord[];
    createdAt: string;
}

export interface DashboardSummary {
    latestRoi: number;
    latestSharpe: number;
    totalBets: number;
    winRate: number;
    todayPredictions: number;
    pendingBets: number;
}

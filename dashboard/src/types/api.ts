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
    modelId: string;
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

// ============================================================================
// Betting Models
// ============================================================================

export interface BettingModel {
    id: string;
    name: string;
    description: string;
    type: 'probability' | 'rating' | 'statistical' | 'arbitrage';
    icon: string;
}

export const BETTING_MODELS: BettingModel[] = [
    {
        id: 'kelly',
        name: 'Kelly Criterion',
        description: 'Optimal bankroll sizing for maximum logarithmic growth',
        type: 'probability',
        icon: '📊',
    },
    {
        id: 'poisson',
        name: 'Poisson Distribution',
        description: 'Score prediction model for Over/Under markets',
        type: 'statistical',
        icon: '📈',
    },
    {
        id: 'elo',
        name: 'ELO Rating',
        description: 'Dynamic team strength ratings (K=20, margin-adjusted)',
        type: 'rating',
        icon: '⚡',
    },
    {
        id: 'mean_reversion',
        name: 'Mean Reversion',
        description: 'Contrarian betting against recent streaks',
        type: 'statistical',
        icon: '🔄',
    },
    {
        id: 'arbitrage',
        name: 'Implicit Probability Arbitrage',
        description: 'Cross-book inefficiency detection for risk-free profit',
        type: 'arbitrage',
        icon: '💰',
    },
];

// ============================================================================
// Staking Strategies
// ============================================================================

export interface StakingStrategy {
    id: string;
    name: string;
    description: string;
    color: string;
    shortName: string;
}

export const STAKING_STRATEGIES: StakingStrategy[] = [
    {
        id: 'flat',
        name: 'Flat Staking',
        description: 'Fixed $20 per bet regardless of edge',
        color: '#94a3b8',  // Gray
        shortName: 'Flat',
    },
    {
        id: 'full_kelly',
        name: 'Full Kelly',
        description: 'Aggressive sizing for maximum growth (high variance)',
        color: '#ef4444',  // Red
        shortName: 'F.Kelly',
    },
    {
        id: 'fractional_kelly',
        name: 'Fractional Kelly (25%)',
        description: 'Conservative Kelly with 75% variance reduction',
        color: '#3b82f6',  // Blue
        shortName: 'Kelly 25%',
    },
    {
        id: 'proportional',
        name: 'Proportional (1%)',
        description: 'Fixed 1% of bankroll per bet',
        color: '#10b981',  // Green
        shortName: '1% Bank',
    },
    {
        id: 'mean_reversion',
        name: 'Mean Reversion Sizing',
        description: 'Larger stakes on streak reversal opportunities',
        color: '#8b5cf6',  // Purple
        shortName: 'MeanRev',
    },
];

// ============================================================================
// Strategy Comparison Data
// ============================================================================

export interface StrategyEquity {
    strategyId: string;
    name: string;
    color: string;
    equity: EquityPoint[];
    finalRoi: number;
    sharpe: number;
    maxDrawdown: number;
}

export interface StrategyComparisonData {
    modelId: string;
    strategies: StrategyEquity[];
}

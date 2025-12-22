import { useEffect, useState, useCallback } from 'react';
import {
    TrendingUp,
    Target,
    Award,
    Activity,
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { EquityCurve } from '../components/EquityCurve';
import { CalibrationChart } from '../components/CalibrationChart';
import { ModelComparisonChart } from '../components/ModelComparison';
import { BetHistoryTable } from '../components/BetHistoryTable';
import { EloRankings } from '../components/EloRankings';
import { ModelSelector } from '../components/ModelSelector';
import { StrategyComparison } from '../components/StrategyComparison';
import { OddsCalculator } from '../components/OddsCalculator';
import TodaysPicks from '../components/TodaysPicks';
import {
    getBacktestForModel,
    getCalibrationData,
    getModelComparison,
    getEloRankings,
    getStrategyComparison,
} from '../api/client';
import {
    BETTING_MODELS,
    type BacktestResult,
    type CalibrationBin,
    type ModelComparison,
    type TeamElo,
    type StrategyComparisonData,
} from '../types/api';

export function Dashboard() {
    const [selectedModel, setSelectedModel] = useState<string>('kelly');
    const [backtest, setBacktest] = useState<BacktestResult | null>(null);
    const [strategyData, setStrategyData] = useState<StrategyComparisonData | null>(null);
    const [calibration, setCalibration] = useState<CalibrationBin[]>([]);
    const [models, setModels] = useState<ModelComparison[]>([]);
    const [rankings, setRankings] = useState<TeamElo[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchData = useCallback(async (modelId: string) => {
        setLoading(true);
        try {
            const [bt, strat, cal, mod, rank] = await Promise.all([
                getBacktestForModel(modelId),
                getStrategyComparison(modelId),
                getCalibrationData(modelId),
                getModelComparison(modelId),
                getEloRankings(),
            ]);
            setBacktest(bt);
            setStrategyData(strat);
            setCalibration(cal);
            setModels(mod);
            setRankings(rank);
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData(selectedModel);
    }, [selectedModel, fetchData]);

    const handleModelChange = (modelId: string) => {
        setSelectedModel(modelId);
    };

    const currentModelInfo = BETTING_MODELS.find(m => m.id === selectedModel);

    if (loading && !backtest) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="animate-pulse text-[var(--muted-foreground)]">
                    Loading dashboard...
                </div>
            </div>
        );
    }

    const metrics = backtest?.metrics;

    return (
        <div className="min-h-screen bg-[var(--background)] p-6">
            {/* Header with Model Selector */}
            <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6 mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-[var(--foreground)]">
                        QuantBet Dashboard
                    </h1>
                    <p className="text-[var(--muted-foreground)] mt-1">
                        NBL/WNBL Betting Model Performance
                    </p>
                </div>

                <div className="w-full lg:w-80">
                    <ModelSelector
                        models={BETTING_MODELS}
                        selected={selectedModel}
                        onSelect={handleModelChange}
                    />
                </div>
            </div>

            {/* TODAY'S PICKS - Primary component for daily use */}
            <TodaysPicks bankroll={1000} />

            {/* Current Model Banner */}
            {currentModelInfo && (
                <div className="bg-[var(--primary)]/10 border border-[var(--primary)]/30 rounded-lg p-4 mb-6">
                    <div className="flex items-center gap-3">
                        <span className="text-3xl">{currentModelInfo.icon}</span>
                        <div>
                            <h2 className="text-lg font-semibold text-[var(--primary)]">
                                {currentModelInfo.name}
                            </h2>
                            <p className="text-sm text-[var(--muted-foreground)]">
                                {currentModelInfo.description}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Metric Cards */}
            {metrics && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                    <MetricCard
                        title="Return on Investment"
                        value={metrics.roi}
                        format="percent"
                        change={0.015}
                        changeLabel="vs last month"
                        icon={<TrendingUp className="w-5 h-5" />}
                    />
                    <MetricCard
                        title="Sharpe Ratio"
                        value={metrics.sharpeRatio}
                        icon={<Activity className="w-5 h-5" />}
                    />
                    <MetricCard
                        title="Win Rate"
                        value={metrics.winRate}
                        format="percent"
                        icon={<Target className="w-5 h-5" />}
                    />
                    <MetricCard
                        title="Total P/L"
                        value={metrics.profitLoss}
                        format="currency"
                        change={metrics.profitLoss > 0 ? 0.08 : -0.05}
                        icon={<Award className="w-5 h-5" />}
                    />
                </div>
            )}

            {/* Secondary Metrics */}
            {metrics && (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
                    <div className="bg-[var(--card)] rounded-lg p-4 border border-[var(--border)]">
                        <p className="text-xs text-[var(--muted-foreground)]">Total Bets</p>
                        <p className="text-xl font-bold">{metrics.totalBets}</p>
                    </div>
                    <div className="bg-[var(--card)] rounded-lg p-4 border border-[var(--border)]">
                        <p className="text-xs text-[var(--muted-foreground)]">Total Staked</p>
                        <p className="text-xl font-bold">${metrics.totalStaked.toLocaleString()}</p>
                    </div>
                    <div className="bg-[var(--card)] rounded-lg p-4 border border-[var(--border)]">
                        <p className="text-xs text-[var(--muted-foreground)]">Sortino Ratio</p>
                        <p className="text-xl font-bold">{metrics.sortinoRatio.toFixed(2)}</p>
                    </div>
                    <div className="bg-[var(--card)] rounded-lg p-4 border border-[var(--border)]">
                        <p className="text-xs text-[var(--muted-foreground)]">Max Drawdown</p>
                        <p className="text-xl font-bold text-red-400">{(metrics.maxDrawdownPct * 100).toFixed(1)}%</p>
                    </div>
                    <div className="bg-[var(--card)] rounded-lg p-4 border border-[var(--border)]">
                        <p className="text-xs text-[var(--muted-foreground)]">Brier Score</p>
                        <p className="text-xl font-bold">{metrics.brierScore.toFixed(3)}</p>
                    </div>
                    <div className="bg-[var(--card)] rounded-lg p-4 border border-[var(--border)]">
                        <p className="text-xs text-[var(--muted-foreground)]">Calibration Error</p>
                        <p className="text-xl font-bold">{metrics.calibrationError.toFixed(3)}</p>
                    </div>
                </div>
            )}

            {/* Strategy Comparison - Full Width */}
            {strategyData && (
                <div className="mb-6">
                    <StrategyComparison strategies={strategyData.strategies} />
                </div>
            )}

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <CalibrationChart data={calibration} />
                <ModelComparisonChart data={models} metric="roi" />
            </div>

            {/* Bottom Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                <EloRankings data={rankings} />
                <OddsCalculator />
                {backtest && <EquityCurve data={backtest.equity} />}
            </div>

            {/* Bet History */}
            {backtest && (
                <BetHistoryTable bets={backtest.bets} limit={15} />
            )}

            {/* Footer */}
            <div className="mt-8 text-center text-sm text-[var(--muted-foreground)]">
                <p>
                    Model: {currentModelInfo?.name}
                    {' | '}
                    Backtest: {backtest?.trainSeasons.join(', ')} → {backtest?.testSeasons.join(', ')}
                    {' | '}Strategy: {backtest?.stakingStrategy}
                </p>
            </div>
        </div>
    );
}

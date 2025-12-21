import { useEffect, useState } from 'react';
import {
    TrendingUp,
    Target,
    Award,
    Percent,
    Activity,
    BarChart3,
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { EquityCurve } from '../components/EquityCurve';
import { CalibrationChart } from '../components/CalibrationChart';
import { ModelComparisonChart } from '../components/ModelComparison';
import { BetHistoryTable } from '../components/BetHistoryTable';
import { EloRankings } from '../components/EloRankings';
import {
    getLatestBacktest,
    getCalibrationData,
    getModelComparison,
    getEloRankings,
} from '../api/client';
import type {
    BacktestResult,
    CalibrationBin,
    ModelComparison,
    TeamElo,
} from '../types/api';

export function Dashboard() {
    const [backtest, setBacktest] = useState<BacktestResult | null>(null);
    const [calibration, setCalibration] = useState<CalibrationBin[]>([]);
    const [models, setModels] = useState<ModelComparison[]>([]);
    const [rankings, setRankings] = useState<TeamElo[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function fetchData() {
            try {
                const [bt, cal, mod, rank] = await Promise.all([
                    getLatestBacktest(),
                    getCalibrationData(),
                    getModelComparison(),
                    getEloRankings(),
                ]);
                setBacktest(bt);
                setCalibration(cal);
                setModels(mod);
                setRankings(rank);
            } catch (error) {
                console.error('Failed to fetch dashboard data:', error);
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, []);

    if (loading) {
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
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-[var(--foreground)]">
                    QuantBet Dashboard
                </h1>
                <p className="text-[var(--muted-foreground)] mt-1">
                    NBL/WNBL Betting Model Performance
                </p>
            </div>

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
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
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

            {/* Charts Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                {backtest && <EquityCurve data={backtest.equity} />}
                <CalibrationChart data={calibration} />
            </div>

            {/* Charts Row 2 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <ModelComparisonChart data={models} metric="roi" />
                <EloRankings data={rankings} />
            </div>

            {/* Bet History */}
            {backtest && (
                <BetHistoryTable bets={backtest.bets} limit={15} />
            )}

            {/* Footer */}
            <div className="mt-8 text-center text-sm text-[var(--muted-foreground)]">
                <p>
                    Backtest: {backtest?.trainSeasons.join(', ')} → {backtest?.testSeasons.join(', ')}
                    {' | '}Strategy: {backtest?.stakingStrategy}
                </p>
            </div>
        </div>
    );
}

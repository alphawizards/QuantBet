import { useState } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import type { StrategyEquity } from '../types/api';

interface StrategyComparisonProps {
    strategies: StrategyEquity[];
    initialBankroll?: number;
}

export function StrategyComparison({
    strategies,
    initialBankroll = 1000
}: StrategyComparisonProps) {
    const [visibleStrategies, setVisibleStrategies] = useState<Set<string>>(
        new Set(strategies.map(s => s.strategyId))
    );

    // Combine all equity curves into unified data points
    const combinedData: Array<{ date: string;[key: string]: number | string }> = [];

    if (strategies.length > 0 && strategies[0].equity.length > 0) {
        const dates = strategies[0].equity.map(e => e.date);

        dates.forEach((date, idx) => {
            const point: { date: string;[key: string]: number | string } = { date };

            strategies.forEach(strategy => {
                if (strategy.equity[idx]) {
                    point[strategy.strategyId] = strategy.equity[idx].bankroll;
                }
            });

            combinedData.push(point);
        });
    }

    const toggleStrategy = (strategyId: string) => {
        setVisibleStrategies(prev => {
            const next = new Set(prev);
            if (next.has(strategyId)) {
                // Don't allow hiding all strategies
                if (next.size > 1) {
                    next.delete(strategyId);
                }
            } else {
                next.add(strategyId);
            }
            return next;
        });
    };

    const formatCurrency = (value: number) => `$${value.toLocaleString()}`;

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 shadow-lg">
                    <p className="text-[var(--muted-foreground)] text-sm mb-2">{label}</p>
                    <div className="space-y-1">
                        {payload
                            .sort((a: any, b: any) => b.value - a.value)
                            .map((entry: any) => {
                                const strategy = strategies.find(s => s.strategyId === entry.dataKey);
                                const profit = entry.value - initialBankroll;
                                const pctChange = ((entry.value - initialBankroll) / initialBankroll) * 100;

                                return (
                                    <div key={entry.dataKey} className="flex items-center gap-2">
                                        <div
                                            className="w-3 h-3 rounded-full"
                                            style={{ backgroundColor: entry.color }}
                                        />
                                        <span className="text-sm text-[var(--foreground)] font-medium">
                                            {strategy?.name}:
                                        </span>
                                        <span className="text-sm text-[var(--foreground)]">
                                            {formatCurrency(entry.value)}
                                        </span>
                                        <span className={`text-xs ${profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                            ({pctChange >= 0 ? '+' : ''}{pctChange.toFixed(1)}%)
                                        </span>
                                    </div>
                                );
                            })}
                    </div>
                </div>
            );
        }
        return null;
    };

    // Calculate Y-axis domain
    const allValues = combinedData.flatMap(d =>
        strategies
            .filter(s => visibleStrategies.has(s.strategyId))
            .map(s => d[s.strategyId] as number)
            .filter(v => v !== undefined)
    );
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const yDomain = [
        Math.floor(minVal * 0.95),
        Math.ceil(maxVal * 1.05),
    ];

    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-[var(--foreground)]">
                    Strategy Comparison
                </h3>

                {/* Strategy toggles */}
                <div className="flex flex-wrap gap-2">
                    {strategies.map(strategy => (
                        <button
                            key={strategy.strategyId}
                            onClick={() => toggleStrategy(strategy.strategyId)}
                            className={`px-3 py-1 text-xs rounded-full border transition-all ${visibleStrategies.has(strategy.strategyId)
                                ? 'border-transparent text-white'
                                : 'border-[var(--border)] text-[var(--muted-foreground)] bg-transparent'
                                }`}
                            style={{
                                backgroundColor: visibleStrategies.has(strategy.strategyId)
                                    ? strategy.color
                                    : 'transparent',
                            }}
                        >
                            {strategy.name.split(' ')[0]}
                        </button>
                    ))}
                </div>
            </div>

            <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={combinedData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />

                        <XAxis
                            dataKey="date"
                            stroke="var(--muted-foreground)"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                            tickFormatter={(val) => val.slice(5)}
                            interval="preserveStartEnd"
                        />

                        <YAxis
                            domain={yDomain}
                            stroke="var(--muted-foreground)"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                            tickFormatter={formatCurrency}
                            width={75}
                        />

                        <Tooltip content={<CustomTooltip />} />

                        <ReferenceLine
                            y={initialBankroll}
                            stroke="var(--muted-foreground)"
                            strokeDasharray="5 5"
                        />

                        {strategies.map(strategy => (
                            visibleStrategies.has(strategy.strategyId) && (
                                <Line
                                    key={strategy.strategyId}
                                    type="monotone"
                                    dataKey={strategy.strategyId}
                                    name={strategy.name}
                                    stroke={strategy.color}
                                    strokeWidth={2}
                                    dot={false}
                                    activeDot={{ r: 4, strokeWidth: 0 }}
                                />
                            )
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Strategy summary table */}
            <div className="mt-4 grid grid-cols-5 gap-2 text-center">
                {strategies.map(strategy => (
                    <div
                        key={strategy.strategyId}
                        className={`p-2 rounded-lg border border-[var(--border)] transition-opacity ${visibleStrategies.has(strategy.strategyId) ? 'opacity-100' : 'opacity-40'
                            }`}
                    >
                        <div
                            className="w-3 h-3 rounded-full mx-auto mb-1"
                            style={{ backgroundColor: strategy.color }}
                        />
                        <p className="text-xs text-[var(--muted-foreground)]">{strategy.name.split(' ')[0]}</p>
                        <p className={`text-sm font-bold ${strategy.finalRoi >= 0 ? 'text-green-400' : 'text-red-400'
                            }`}>
                            {(strategy.finalRoi * 100).toFixed(1)}%
                        </p>
                        <p className="text-xs text-[var(--muted-foreground)]">
                            Sharpe: {strategy.sharpe.toFixed(2)}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
}

import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from 'recharts';
import type { ModelComparison } from '../types/api';

interface ModelComparisonChartProps {
    data: ModelComparison[];
    metric?: 'roi' | 'sharpe' | 'brier' | 'winRate';
}

export function ModelComparisonChart({
    data,
    metric = 'roi'
}: ModelComparisonChartProps) {
    const metricConfig = {
        roi: { label: 'ROI', format: (v: number) => `${(v * 100).toFixed(1)}%` },
        sharpe: { label: 'Sharpe Ratio', format: (v: number) => v.toFixed(2) },
        brier: { label: 'Brier Score', format: (v: number) => v.toFixed(3) },
        winRate: { label: 'Win Rate', format: (v: number) => `${(v * 100).toFixed(1)}%` },
    };

    const config = metricConfig[metric];

    const getBarColor = (value: number, metricKey: string) => {
        if (metricKey === 'brier') {
            // Lower is better for Brier
            return value < 0.22 ? '#10b981' : value < 0.24 ? '#f59e0b' : '#ef4444';
        }
        // Higher is better for others
        return value > 0 ? '#10b981' : value > -0.05 ? '#f59e0b' : '#ef4444';
    };

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length > 0) {
            const model = payload[0].payload;

            return (
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 shadow-lg">
                    <p className="text-[var(--foreground)] font-semibold mb-2">
                        {model.name}
                    </p>
                    <div className="space-y-1 text-sm">
                        <p><span className="text-[var(--muted-foreground)]">ROI:</span> {(model.roi * 100).toFixed(1)}%</p>
                        <p><span className="text-[var(--muted-foreground)]">Sharpe:</span> {model.sharpe.toFixed(2)}</p>
                        <p><span className="text-[var(--muted-foreground)]">Brier:</span> {model.brier.toFixed(3)}</p>
                        <p><span className="text-[var(--muted-foreground)]">Win Rate:</span> {(model.winRate * 100).toFixed(1)}%</p>
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <h3 className="text-lg font-semibold text-[var(--foreground)] mb-4">
                Model Comparison - {config.label}
            </h3>

            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />

                        <XAxis
                            dataKey="name"
                            stroke="var(--muted-foreground)"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                        />

                        <YAxis
                            stroke="var(--muted-foreground)"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                            tickFormatter={config.format}
                        />

                        <Tooltip content={<CustomTooltip />} />

                        <Bar dataKey={metric} radius={[6, 6, 0, 0]} barSize={50}>
                            {data.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={getBarColor(entry[metric], metric)}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

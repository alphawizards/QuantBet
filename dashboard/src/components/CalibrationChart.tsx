import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    Line,
    ComposedChart,
} from 'recharts';
import type { CalibrationBin } from '../types/api';

interface CalibrationChartProps {
    data: CalibrationBin[];
}

export function CalibrationChart({ data }: CalibrationChartProps) {
    // Add perfect calibration line
    const chartData = data.map(d => ({
        ...d,
        perfect: d.predicted,
    }));

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length > 0) {
            const item = payload[0].payload;
            const error = Math.abs(item.predicted - item.actual);

            return (
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 shadow-lg">
                    <p className="text-[var(--muted-foreground)] text-sm mb-2">
                        Bin: {item.bin}
                    </p>
                    <div className="space-y-1">
                        <p className="text-[var(--primary)] text-sm">
                            Predicted: {(item.predicted * 100).toFixed(1)}%
                        </p>
                        <p className="text-[var(--accent)] text-sm">
                            Actual: {(item.actual * 100).toFixed(1)}%
                        </p>
                        <p className="text-[var(--muted-foreground)] text-sm">
                            Error: {(error * 100).toFixed(1)}%
                        </p>
                        <p className="text-[var(--muted-foreground)] text-sm">
                            Samples: {item.count}
                        </p>
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-[var(--foreground)]">
                    Calibration Plot
                </h3>
                <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded bg-[var(--primary)]" />
                        <span className="text-[var(--muted-foreground)]">Predicted</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded bg-[var(--accent)]" />
                        <span className="text-[var(--muted-foreground)]">Actual</span>
                    </div>
                </div>
            </div>

            <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />

                        <XAxis
                            dataKey="bin"
                            stroke="var(--muted-foreground)"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                            angle={-45}
                            textAnchor="end"
                            height={60}
                        />

                        <YAxis
                            domain={[0, 1]}
                            stroke="var(--muted-foreground)"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                            tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                        />

                        <Tooltip content={<CustomTooltip />} />

                        {/* Perfect calibration line */}
                        <Line
                            type="linear"
                            dataKey="perfect"
                            stroke="var(--muted-foreground)"
                            strokeDasharray="5 5"
                            strokeWidth={1}
                            dot={false}
                        />

                        <Bar
                            dataKey="predicted"
                            fill="var(--primary)"
                            opacity={0.8}
                            radius={[4, 4, 0, 0]}
                            barSize={20}
                        />

                        <Bar
                            dataKey="actual"
                            fill="var(--accent)"
                            opacity={0.8}
                            radius={[4, 4, 0, 0]}
                            barSize={20}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            <p className="text-[var(--muted-foreground)] text-xs mt-4 text-center">
                Dashed line = perfect calibration. Bars should align for well-calibrated predictions.
            </p>
        </div>
    );
}

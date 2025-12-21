import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    Area,
    AreaChart,
} from 'recharts';
import type { EquityPoint } from '../types/api';

interface EquityCurveProps {
    data: EquityPoint[];
    initialBankroll?: number;
}

export function EquityCurve({ data, initialBankroll = 1000 }: EquityCurveProps) {
    const formatCurrency = (value: number) => `$${value.toLocaleString()}`;

    const minBankroll = Math.min(...data.map(d => d.bankroll));
    const maxBankroll = Math.max(...data.map(d => d.bankroll));
    const yDomain = [
        Math.floor(minBankroll * 0.95),
        Math.ceil(maxBankroll * 1.05),
    ];

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            const value = payload[0].value;
            const profit = value - initialBankroll;
            const profitPct = ((value - initialBankroll) / initialBankroll) * 100;

            return (
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 shadow-lg">
                    <p className="text-[var(--muted-foreground)] text-sm">{label}</p>
                    <p className="text-[var(--foreground)] font-bold text-lg">
                        {formatCurrency(value)}
                    </p>
                    <p className={`text-sm ${profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {profit >= 0 ? '+' : ''}{formatCurrency(profit)} ({profitPct.toFixed(1)}%)
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <h3 className="text-lg font-semibold text-[var(--foreground)] mb-4">
                Equity Curve
            </h3>

            <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                        <defs>
                            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                        </defs>

                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />

                        <XAxis
                            dataKey="date"
                            stroke="var(--muted-foreground)"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                            tickFormatter={(val) => val.slice(5)} // Show MM-DD
                            interval="preserveStartEnd"
                        />

                        <YAxis
                            domain={yDomain}
                            stroke="var(--muted-foreground)"
                            tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                            tickFormatter={formatCurrency}
                            width={80}
                        />

                        <Tooltip content={<CustomTooltip />} />

                        <ReferenceLine
                            y={initialBankroll}
                            stroke="var(--muted-foreground)"
                            strokeDasharray="5 5"
                            label={{
                                value: 'Initial',
                                position: 'right',
                                fill: 'var(--muted-foreground)',
                                fontSize: 12,
                            }}
                        />

                        <Area
                            type="monotone"
                            dataKey="bankroll"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#equityGradient)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

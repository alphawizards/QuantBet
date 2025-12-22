import type { ReactNode } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricCardProps {
    title: string;
    value: string | number;
    change?: number;
    changeLabel?: string;
    icon?: ReactNode;
    format?: 'percent' | 'number' | 'currency';
}

export function MetricCard({
    title,
    value,
    change,
    changeLabel,
    icon,
    format = 'number',
}: MetricCardProps) {
    const formatValue = (val: string | number): string => {
        if (typeof val === 'string') return val;

        switch (format) {
            case 'percent':
                return `${(val * 100).toFixed(1)}%`;
            case 'currency':
                return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2 })}`;
            default:
                return val.toLocaleString(undefined, { maximumFractionDigits: 2 });
        }
    };

    const getTrendIcon = () => {
        if (change === undefined) return null;
        if (change > 0) return <TrendingUp className="w-4 h-4 text-green-400" />;
        if (change < 0) return <TrendingDown className="w-4 h-4 text-red-400" />;
        return <Minus className="w-4 h-4 text-gray-400" />;
    };

    const getTrendColor = () => {
        if (change === undefined) return 'text-gray-400';
        if (change > 0) return 'text-green-400';
        if (change < 0) return 'text-red-400';
        return 'text-gray-400';
    };

    return (
        <div className="metric-card bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <div className="flex items-center justify-between mb-4">
                <span className="text-[var(--muted-foreground)] text-sm font-medium">
                    {title}
                </span>
                {icon && (
                    <div className="w-10 h-10 rounded-lg bg-[var(--primary)]/10 flex items-center justify-center text-[var(--primary)]">
                        {icon}
                    </div>
                )}
            </div>

            <div className="flex items-end justify-between">
                <div>
                    <p className="text-3xl font-bold text-[var(--foreground)]">
                        {formatValue(value)}
                    </p>

                    {change !== undefined && (
                        <div className={`flex items-center gap-1 mt-2 ${getTrendColor()}`}>
                            {getTrendIcon()}
                            <span className="text-sm font-medium">
                                {change > 0 ? '+' : ''}{(change * 100).toFixed(1)}%
                            </span>
                            {changeLabel && (
                                <span className="text-xs text-[var(--muted-foreground)] ml-1">
                                    {changeLabel}
                                </span>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

import { Check, X } from 'lucide-react';
import type { BetRecord } from '../types/api';

interface BetHistoryTableProps {
    bets: BetRecord[];
    limit?: number;
}

export function BetHistoryTable({ bets, limit = 10 }: BetHistoryTableProps) {
    const displayBets = bets.slice(0, limit);

    const formatCurrency = (value: number) => {
        const prefix = value >= 0 ? '+' : '';
        return `${prefix}$${Math.abs(value).toFixed(2)}`;
    };

    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-lg font-semibold text-[var(--foreground)]">
                        Backtest Performance
                    </h3>
                    <p className="text-xs text-[var(--muted-foreground)] mt-0.5">
                        Historical model validation data • 2011-2012 season
                    </p>
                </div>
                <span className="text-sm text-[var(--muted-foreground)]">
                    Showing {displayBets.length} of {bets.length}
                </span>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead>
                        <tr className="border-b border-[var(--border)]">
                            <th className="text-left py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                Date
                            </th>
                            <th className="text-left py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                Game
                            </th>
                            <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                Pred
                            </th>
                            <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                Odds
                            </th>
                            <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                Stake
                            </th>
                            <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                Result
                            </th>
                            <th className="text-right py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                P/L
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {displayBets.map((bet, index) => (
                            <tr
                                key={bet.gameId}
                                className={`border-b border-[var(--border)] hover:bg-[var(--muted)]/30 transition-colors ${index % 2 === 0 ? 'bg-transparent' : 'bg-[var(--muted)]/10'
                                    }`}
                            >
                                <td className="py-3 px-2 text-sm text-[var(--muted-foreground)]">
                                    {bet.date}
                                </td>
                                <td className="py-3 px-2 text-sm text-[var(--foreground)]">
                                    <span className="font-medium">{bet.homeTeam}</span>
                                    <span className="text-[var(--muted-foreground)]"> vs </span>
                                    <span>{bet.awayTeam}</span>
                                </td>
                                <td className="py-3 px-2 text-sm text-center text-[var(--foreground)]">
                                    {(bet.prediction * 100).toFixed(0)}%
                                </td>
                                <td className="py-3 px-2 text-sm text-center text-[var(--muted-foreground)]">
                                    {bet.odds.toFixed(2)}
                                </td>
                                <td className="py-3 px-2 text-sm text-center text-[var(--foreground)]">
                                    ${bet.stake.toFixed(2)}
                                </td>
                                <td className="py-3 px-2 text-center">
                                    {bet.won ? (
                                        <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-green-500/20">
                                            <Check className="w-4 h-4 text-green-400" />
                                        </span>
                                    ) : (
                                        <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-red-500/20">
                                            <X className="w-4 h-4 text-red-400" />
                                        </span>
                                    )}
                                </td>
                                <td className={`py-3 px-2 text-sm text-right font-medium ${bet.profit >= 0 ? 'text-green-400' : 'text-red-400'
                                    }`}>
                                    {formatCurrency(bet.profit)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

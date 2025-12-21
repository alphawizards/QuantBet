import { TrendingUp, TrendingDown } from 'lucide-react';
import type { TeamElo } from '../types/api';

interface EloRankingsProps {
    data: TeamElo[];
}

export function EloRankings({ data }: EloRankingsProps) {
    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <h3 className="text-lg font-semibold text-[var(--foreground)] mb-4">
                Team ELO Rankings
            </h3>

            <div className="space-y-2">
                {data.map((team) => (
                    <div
                        key={team.team}
                        className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-[var(--muted)]/30 transition-colors"
                    >
                        <div className="flex items-center gap-3">
                            <span className={`w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold ${team.rank <= 3
                                    ? 'bg-[var(--primary)]/20 text-[var(--primary)]'
                                    : 'bg-[var(--muted)] text-[var(--muted-foreground)]'
                                }`}>
                                {team.rank}
                            </span>

                            <div>
                                <p className="text-sm font-medium text-[var(--foreground)]">
                                    {team.teamName}
                                </p>
                                <p className="text-xs text-[var(--muted-foreground)]">
                                    {team.team}
                                </p>
                            </div>
                        </div>

                        <div className="flex items-center gap-4">
                            <span className="text-sm font-mono font-medium text-[var(--foreground)]">
                                {team.elo.toFixed(0)}
                            </span>

                            {team.change !== undefined && (
                                <div className={`flex items-center gap-1 text-xs ${team.change > 0 ? 'text-green-400' : team.change < 0 ? 'text-red-400' : 'text-gray-400'
                                    }`}>
                                    {team.change > 0 ? (
                                        <TrendingUp className="w-3 h-3" />
                                    ) : team.change < 0 ? (
                                        <TrendingDown className="w-3 h-3" />
                                    ) : null}
                                    <span>{team.change > 0 ? '+' : ''}{team.change}</span>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

import { Clock, TrendingUp, AlertCircle } from 'lucide-react';
import type { UpcomingGame } from '../types/api';

interface PendingBetsTableProps {
    games: UpcomingGame[];
    bankroll?: number;
    limit?: number;
}

export function PendingBetsTable({ games, bankroll = 1000, limit = 10 }: PendingBetsTableProps) {
    const displayGames = games.slice(0, limit);

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        const today = new Date();
        const tomorrow = new Date(today);
        tomorrow.setDate(tomorrow.getDate() + 1);

        if (date.toDateString() === today.toDateString()) {
            return `Today ${date.toLocaleTimeString('en-AU', { hour: '2-digit', minute: '2-digit' })}`;
        } else if (date.toDateString() === tomorrow.toDateString()) {
            return `Tomorrow ${date.toLocaleTimeString('en-AU', { hour: '2-digit', minute: '2-digit' })}`;
        } else {
            return date.toLocaleDateString('en-AU', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        }
    };

    const calculateStake = (kellyFraction: number) => {
        return (bankroll * kellyFraction).toFixed(2);
    };

    const getConfidenceBadge = (confidence: string) => {
        const colors = {
            HIGH: 'bg-green-500/20 text-green-400 border-green-500/30',
            MEDIUM: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
            LOW: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
        };
        return colors[confidence as keyof typeof colors] || colors.LOW;
    };

    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-lg font-semibold text-[var(--foreground)] flex items-center gap-2">
                        <Clock className="w-5 h-5 text-blue-400" />
                        Pending Bets
                    </h3>
                    <p className="text-xs text-[var(--muted-foreground)] mt-0.5">
                        Upcoming games with positive edge • Ready to place
                    </p>
                </div>
                <span className="text-sm text-[var(--muted-foreground)]">
                    {displayGames.length} opportunities
                </span>
            </div>

            {displayGames.length === 0 ? (
                <div className="text-center py-8 text-[var(--muted-foreground)]">
                    <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No upcoming games with positive edge</p>
                    <p className="text-xs mt-1">Check back later for new opportunities</p>
                </div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-[var(--border)]">
                                <th className="text-left py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                    When
                                </th>
                                <th className="text-left py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                    Game
                                </th>
                                <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                    Bet On
                                </th>
                                <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                    Odds
                                </th>
                                <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                    Edge
                                </th>
                                <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                    Stake
                                </th>
                                <th className="text-center py-3 px-2 text-sm font-medium text-[var(--muted-foreground)]">
                                    Confidence
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {displayGames.map((game, index) => {
                                const betOn = game.recommendation === 'BET_HOME' ? game.home_team : game.away_team;
                                const odds = game.recommendation === 'BET_HOME' ? game.home_odds : game.away_odds;
                                const edge = game.recommendation === 'BET_HOME' ? game.home_edge : game.away_edge;

                                return (
                                    <tr
                                        key={game.event_id}
                                        className={`border-b border-[var(--border)] hover:bg-[var(--muted)]/30 transition-colors ${index % 2 === 0 ? 'bg-transparent' : 'bg-[var(--muted)]/10'
                                            }`}
                                    >
                                        <td className="py-3 px-2 text-sm text-[var(--muted-foreground)] whitespace-nowrap">
                                            {formatDate(game.commence_time)}
                                        </td>
                                        <td className="py-3 px-2 text-sm text-[var(--foreground)]">
                                            <div className="flex flex-col">
                                                <span className="font-medium">{game.home_team}</span>
                                                <span className="text-[var(--muted-foreground)] text-xs">vs {game.away_team}</span>
                                            </div>
                                        </td>
                                        <td className="py-3 px-2 text-sm text-center">
                                            <span className="font-medium text-blue-400">{betOn}</span>
                                        </td>
                                        <td className="py-3 px-2 text-sm text-center text-[var(--foreground)]">
                                            {odds.toFixed(2)}
                                        </td>
                                        <td className="py-3 px-2 text-center">
                                            <span className="inline-flex items-center gap-1 text-sm font-medium text-green-400">
                                                <TrendingUp className="w-3 h-3" />
                                                {(edge * 100).toFixed(1)}%
                                            </span>
                                        </td>
                                        <td className="py-3 px-2 text-sm text-center text-[var(--foreground)] font-medium">
                                            ${calculateStake(game.recommended_stake_pct)}
                                            <span className="text-xs text-[var(--muted-foreground)] ml-1">
                                                ({(game.recommended_stake_pct * 100).toFixed(1)}%)
                                            </span>
                                        </td>
                                        <td className="py-3 px-2 text-center">
                                            <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full border ${getConfidenceBadge(game.confidence)}`}>
                                                {game.confidence}
                                            </span>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

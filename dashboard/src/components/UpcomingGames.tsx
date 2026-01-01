import { useEffect, useState } from 'react';
import { Calendar, TrendingUp, AlertCircle, Clock, DollarSign } from 'lucide-react';
import { getUpcomingGames } from '../api/client';
import type { UpcomingGame } from '../types/api';

interface UpcomingGamesProps {
    days?: number;
    bankroll?: number;
}

export function UpcomingGames({ days = 7, bankroll = 1000 }: UpcomingGamesProps) {
    const [games, setGames] = useState<UpcomingGame[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedDays, setSelectedDays] = useState(days);

    useEffect(() => {
        fetchGames();
    }, [selectedDays, bankroll]);

    const fetchGames = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getUpcomingGames(selectedDays, bankroll);
            setGames(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch games');
            console.error('Error fetching upcoming games:', err);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateStr: string) => {
        try {
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-AU', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            });
        } catch {
            return dateStr;
        }
    };

    const getConfidenceBadge = (confidence: string) => {
        const colors = {
            HIGH: 'bg-green-500/20 text-green-400 border-green-500/30',
            MEDIUM: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
            LOW: 'bg-red-500/20 text-red-400 border-red-500/30',
        };
        return colors[confidence as keyof typeof colors] || colors.MEDIUM;
    };

    const getRecommendationBadge = (rec: string) => {
        if (rec === 'BET_HOME') return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
        if (rec === 'BET_AWAY') return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    };

    const getRecommendationText = (rec: string, homeTeam: string, awayTeam: string) => {
        if (rec === 'BET_HOME') return `Bet ${homeTeam}`;
        if (rec === 'BET_AWAY') return `Bet ${awayTeam}`;
        return 'Skip';
    };

    if (loading) {
        return (
            <div className="bg-[var(--card)] rounded-lg p-8 border border-[var(--border)]">
                <div className="flex items-center justify-center">
                    <div className="animate-pulse text-[var(--muted-foreground)]">
                        Loading upcoming games...
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-[var(--card)] rounded-lg p-6 border border-red-500/30">
                <div className="flex items-center gap-3 text-red-400">
                    <AlertCircle className="w-5 h-5" />
                    <div>
                        <p className="font-semibold">Error loading games</p>
                        <p className="text-sm text-[var(--muted-foreground)] mt-1">{error}</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-[var(--card)] rounded-lg border border-[var(--border)] mb-6">
            {/* Header */}
            <div className="p-6 border-b border-[var(--border)]">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Calendar className="w-6 h-6 text-[var(--primary)]" />
                        <div>
                            <h2 className="text-2xl font-bold text-[var(--foreground)]">
                                Upcoming Games
                            </h2>
                            <p className="text-sm text-[var(--muted-foreground)] mt-1">
                                NBL predictions for the next {selectedDays} days
                            </p>
                        </div>
                    </div>

                    {/* Day selector */}
                    <div className="flex items-center gap-2">
                        <label className="text-sm text-[var(--muted-foreground)]">
                            Days ahead:
                        </label>
                        <select
                            value={selectedDays}
                            onChange={(e) => setSelectedDays(Number(e.target.value))}
                            className="bg-[var(--background)] border border-[var(--border)] rounded px-3 py-1.5 text-sm"
                        >
                            <option value={3}>3 days</option>
                            <option value={7}>7 days</option>
                            <option value={14}>14 days</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Games List */}
            <div className="p-6">
                {games.length === 0 ? (
                    <div className="text-center py-8 text-[var(--muted-foreground)]">
                        <Calendar className="w-12 h-12 mx-auto mb-3 opacity-30" />
                        <p>No games scheduled in the next {selectedDays} days</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {games.map((game, idx) => (
                            <div
                                key={game.event_id || idx}
                                className="border border-[var(--border)] rounded-lg p-4 hover:border-[var(--primary)]/50 transition-colors"
                            >
                                {/* Game Header */}
                                <div className="flex items-start justify-between mb-3">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-3 mb-2">
                                            <div className="text-lg font-bold">
                                                {game.home_team}
                                                <span className="text-[var(--muted-foreground)] mx-2">vs</span>
                                                {game.away_team}
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-4 text-sm text-[var(--muted-foreground)]">
                                            <div className="flex items-center gap-1">
                                                <Clock className="w-4 h-4" />
                                                {formatDate(game.commence_time)}
                                            </div>
                                            {game.best_bookmaker !== 'N/A' && (
                                                <div className="text-xs">
                                                    Best odds: {game.best_bookmaker}
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Recommendation Badge */}
                                    <div className="flex flex-col items-end gap-2">
                                        <span
                                            className={`px-3 py-1 rounded-full text-xs font-semibold border ${getRecommendationBadge(
                                                game.recommendation
                                            )}`}
                                        >
                                            {getRecommendationText(game.recommendation, game.home_team, game.away_team)}
                                        </span>
                                        <span
                                            className={`px-2 py-0.5 rounded text-xs border ${getConfidenceBadge(
                                                game.confidence
                                            )}`}
                                        >
                                            {game.confidence}
                                        </span>
                                    </div>
                                </div>

                                {/* Stats Grid */}
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3 p-3 bg-[var(--background)] rounded border border-[var(--border)]">
                                    <div>
                                        <p className="text-xs text-[var(--muted-foreground)] mb-1">Home Win</p>
                                        <p className="text-lg font-bold text-[var(--primary)]">
                                            {(game.predicted_home_prob * 100).toFixed(1)}%
                                        </p>
                                        <p className="text-xs text-[var(--muted-foreground)]">
                                            {game.predicted_home_prob_lower
                                                ? `CI: ${(game.predicted_home_prob_lower * 100).toFixed(0)}-${(
                                                    game.predicted_home_prob_upper * 100
                                                ).toFixed(0)}%`
                                                : ''}
                                        </p>
                                    </div>

                                    <div>
                                        <p className="text-xs text-[var(--muted-foreground)] mb-1">Odds</p>
                                        <p className="text-lg font-bold">
                                            {game.home_odds.toFixed(2)} / {game.away_odds.toFixed(2)}
                                        </p>
                                        <p className="text-xs text-[var(--muted-foreground)]">
                                            Home / Away
                                        </p>
                                    </div>

                                    <div>
                                        <p className="text-xs text-[var(--muted-foreground)] mb-1">Edge</p>
                                        <p
                                            className={`text-lg font-bold ${Math.max(game.home_edge, game.away_edge) > 0
                                                    ? 'text-green-400'
                                                    : 'text-red-400'
                                                }`}
                                        >
                                            {(Math.max(game.home_edge, game.away_edge) * 100).toFixed(1)}%
                                        </p>
                                        <p className="text-xs text-[var(--muted-foreground)]">
                                            {game.home_edge > game.away_edge ? 'Home' : 'Away'}
                                        </p>
                                    </div>

                                    <div>
                                        <p className="text-xs text-[var(--muted-foreground)] mb-1">Kelly Stake</p>
                                        <p className="text-lg font-bold flex items-center gap-1">
                                            <DollarSign className="w-4 h-4" />
                                            {(game.recommended_stake_pct * bankroll / 100).toFixed(0)}
                                        </p>
                                        <p className="text-xs text-[var(--muted-foreground)]">
                                            {game.recommended_stake_pct.toFixed(2)}% of bankroll
                                        </p>
                                    </div>
                                </div>

                                {/* Key Factors */}
                                {game.top_factors && game.top_factors.length > 0 && (
                                    <div className="pt-3 border-t border-[var(--border)]">
                                        <p className="text-xs text-[var(--muted-foreground)] mb-2 flex items-center gap-1">
                                            <TrendingUp className="w-3 h-3" />
                                            Key Factors
                                        </p>
                                        <ul className="space-y-1">
                                            {game.top_factors.map((factor, i) => (
                                                <li
                                                    key={i}
                                                    className="text-xs text-[var(--muted-foreground)] flex items-start gap-2"
                                                >
                                                    <span className="text-[var(--primary)] mt-0.5">•</span>
                                                    <span>{factor}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

import { useEffect, useState } from 'react';
import { Calendar, TrendingUp, DollarSign, Target } from 'lucide-react';

interface ModelPrediction {
    model_name: string;
    predicted_home_prob: number;
    recommended_bet: 'BET_HOME' | 'BET_AWAY' | 'SKIP';
    kelly_stake_pct: number;
    edge: number;
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
}

interface MultiModelGame {
    event_id: string;
    home_team: string;
    away_team: string;
    commence_time: string;
    home_odds: number;
    away_odds: number;
    best_bookmaker: string;
    model_predictions: ModelPrediction[];
}

interface WeeklyPredictionsProps {
    bankroll?: number;
    startDate?: string; // Format: "2026-01-05"
}

export function WeeklyPredictionsTable({ bankroll = 1000, startDate = "2026-01-05" }: WeeklyPredictionsProps) {
    const [games, setGames] = useState<MultiModelGame[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchWeeklyPredictions();
    }, [startDate, bankroll]);

    const fetchWeeklyPredictions = async () => {
        setLoading(true);
        setError(null);
        try {
            // Calculate days from start date to get one week window
            const start = new Date(startDate);
            const now = new Date();
            const daysUntilStart = Math.ceil((start.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
            const days = Math.max(1, daysUntilStart + 7); // Start date + 7 days

            const response = await fetch(
                `http://localhost:8000/games/multi-model-predictions?days=${days}&bankroll=${bankroll}`
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data: MultiModelGame[] = await response.json();

            // Filter to only show games in the target week
            const filtered = data.filter(game => {
                const gameDate = new Date(game.commence_time);
                const weekEnd = new Date(start);
                weekEnd.setDate(weekEnd.getDate() + 7);
                return gameDate >= start && gameDate < weekEnd;
            });

            setGames(filtered);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch predictions');
            console.error('Error fetching weekly predictions:', err);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-AU', {
            weekday: 'short',
            day: 'numeric',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    const getConfidenceColor = (confidence: string) => {
        if (confidence === 'HIGH') return 'text-green-400';
        if (confidence === 'MEDIUM') return 'text-yellow-400';
        return 'text-red-400';
    };

    const getRecommendationBadge = (rec: string) => {
        if (rec === 'BET_HOME') return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
        if (rec === 'BET_AWAY') return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    };

    if (loading) {
        return (
            <div className="bg-[var(--card)] rounded-lg p-8 border border-[var(--border)]">
                <div className="flex items-center justify-center">
                    <div className="animate-pulse text-[var(--muted-foreground)]">
                        Loading weekly predictions...
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-[var(--card)] rounded-lg p-6 border border-red-500/30">
                <p className="text-red-400">Error: {error}</p>
            </div>
        );
    }

    return (
        <div className="bg-[var(--card)] rounded-lg border border-[var(--border)] mb-6">
            {/* Header */}
            <div className="p-6 border-b border-[var(--border)]">
                <div className="flex items-center gap-3">
                    <Calendar className="w-6 h-6 text-[var(--primary)]" />
                    <div>
                        <h2 className="text-2xl font-bold text-[var(--foreground)]">
                            Weekly Multi-Model Predictions
                        </h2>
                        <p className="text-sm text-[var(--muted-foreground)] mt-1">
                            Week starting {new Date(startDate).toLocaleDateString('en-AU', { month: 'long', day: 'numeric', year: 'numeric' })}
                        </p>
                    </div>
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
                {games.length === 0 ? (
                    <div className="text-center py-8 text-[var(--muted-foreground)]">
                        <Calendar className="w-12 h-12 mx-auto mb-3 opacity-30" />
                        <p>No games scheduled for this week</p>
                    </div>
                ) : (
                    <table className="w-full">
                        <thead className="bg-[var(--background)] border-b border-[var(--border)]">
                            <tr>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
                                    Game
                                </th>
                                <th className="px-4 py-4 text-center text-xs font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
                                    Odds
                                </th>
                                <th className="px-4 py-4 text-center text-xs font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
                                    Bayesian ELO
                                </th>
                                <th className="px-4 py-4 text-center text-xs font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
                                    Market Implied
                                </th>
                                <th className="px-4 py-4 text-center text-xs font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
                                    Ensemble
                                </th>
                                <th className="px-4 py-4 text-center text-xs font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">
                                    Classic ELO
                                </th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-[var(--border)]">
                            {games.map((game) => (
                                <tr key={game.event_id} className="hover:bg-[var(--background)] transition-colors">
                                    {/* Game Info */}
                                    <td className="px-6 py-4">
                                        <div className="flex flex-col gap-1">
                                            <div className="font-semibold text-[var(--foreground)]">
                                                {game.home_team} <span className="text-[var(--muted-foreground)] font-normal">vs</span> {game.away_team}
                                            </div>
                                            <div className="text-xs text-[var(--muted-foreground)]">
                                                {formatDate(game.commence_time)}
                                            </div>
                                            <div className="text-xs text-[var(--muted-foreground)]">
                                                {game.best_bookmaker}
                                            </div>
                                        </div>
                                    </td>

                                    {/* Odds */}
                                    <td className="px-4 py-4">
                                        <div className="text-center">
                                            <div className="text-sm font-semibold">{game.home_odds.toFixed(2)}</div>
                                            <div className="text-xs text-[var(--muted-foreground)]">/</div>
                                            <div className="text-sm font-semibold">{game.away_odds.toFixed(2)}</div>
                                        </div>
                                    </td>

                                    {/* Model Predictions */}
                                    {['Bayesian ELO', 'Market Implied', 'Ensemble', 'ELO'].map((modelName) => {
                                        const prediction = game.model_predictions.find(p => p.model_name === modelName);
                                        if (!prediction) {
                                            return (
                                                <td key={modelName} className="px-4 py-4 text-center text-[var(--muted-foreground)]">
                                                    N/A
                                                </td>
                                            );
                                        }

                                        return (
                                            <td key={modelName} className="px-4 py-4">
                                                <div className="flex flex-col items-center gap-2">
                                                    {/* Win Probability */}
                                                    <div className="text-sm font-semibold text-[var(--primary)]">
                                                        {(prediction.predicted_home_prob * 100).toFixed(1)}%
                                                    </div>

                                                    {/* Recommendation */}
                                                    <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${getRecommendationBadge(prediction.recommended_bet)}`}>
                                                        {prediction.recommended_bet === 'BET_HOME' ? '↑ Home' :
                                                            prediction.recommended_bet === 'BET_AWAY' ? '↓ Away' : 'Skip'}
                                                    </span>

                                                    {/* Kelly Stake */}
                                                    {prediction.kelly_stake_pct > 0 && (
                                                        <div className="flex items-center gap-1 text-xs">
                                                            <DollarSign className="w-3 h-3" />
                                                            <span className="font-mono">
                                                                ${Math.round(prediction.kelly_stake_pct * bankroll / 100)}
                                                            </span>
                                                        </div>
                                                    )}

                                                    {/* Edge */}
                                                    <div className={`text-xs ${prediction.edge > 0 ? 'text-green-400' : 'text-gray-400'}`}>
                                                        {prediction.edge > 0 ? '+' : ''}{(prediction.edge * 100).toFixed(1)}%
                                                    </div>

                                                    {/* Confidence */}
                                                    <div className={`text-xs ${getConfidenceColor(prediction.confidence)}`}>
                                                        {prediction.confidence}
                                                    </div>
                                                </div>
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Legend */}
            <div className="px-6 py-4 border-t border-[var(--border)] bg-[var(--background)]/50">
                <div className="flex flex-wrap gap-6 text-xs text-[var(--muted-foreground)]">
                    <div className="flex items-center gap-2">
                        <Target className="w-4 h-4" />
                        <span>Win Probability = Model's predicted home team win chance</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <DollarSign className="w-4 h-4" />
                        <span>Stake = Kelly criterion recommended bet size (${bankroll} bankroll)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" />
                        <span>Edge = Model probability vs market probability difference</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

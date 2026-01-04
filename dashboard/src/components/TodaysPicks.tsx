/**
 * Today's Picks Component
 * 
 * Shows daily NBL betting predictions with:
 * - Game matchups with live odds
 * - Model probability with uncertainty
 * - Kelly stake recommendations
 * - Color-coded bet suggestions
 */

import { useState, useEffect } from 'react';
import { trackBet } from '../api/client';

interface TodayPrediction {
    event_id: string;
    home_team: string;
    away_team: string;
    commence_time: string;
    predicted_home_prob: number;
    predicted_home_prob_lower: number;
    predicted_home_prob_upper: number;
    uncertainty: number;
    home_odds: number;
    away_odds: number;
    best_bookmaker: string;
    home_edge: number;
    away_edge: number;
    recommendation: 'BET_HOME' | 'BET_AWAY' | 'SKIP';
    kelly_fraction: number;
    recommended_stake_pct: number;
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
    top_factors: string[];
}

interface TodaysPicksProps {
    bankroll?: number;
}

const formatTime = (isoString: string): string => {
    try {
        const date = new Date(isoString);
        return date.toLocaleTimeString('en-AU', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        });
    } catch {
        return isoString;
    }
};

const formatDate = (isoString: string): string => {
    try {
        const date = new Date(isoString);
        return date.toLocaleDateString('en-AU', {
            weekday: 'short',
            month: 'short',
            day: 'numeric'
        });
    } catch {
        return '';
    }
};

const getRecommendationColor = (rec: string): string => {
    switch (rec) {
        case 'BET_HOME':
        case 'BET_AWAY':
            return 'bg-green-500/20 border-green-500 text-green-400';
        case 'SKIP':
        default:
            return 'bg-gray-500/20 border-gray-500 text-gray-400';
    }
};

const getConfidenceBadge = (conf: string): string => {
    switch (conf) {
        case 'HIGH':
            return 'bg-green-600 text-white';
        case 'MEDIUM':
            return 'bg-yellow-600 text-white';
        case 'LOW':
        default:
            return 'bg-red-600 text-white';
    }
};

export default function TodaysPicks({ bankroll = 1000 }: TodaysPicksProps) {
    const [predictions, setPredictions] = useState<TodayPrediction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
    const [tracking, setTracking] = useState<string | null>(null); // Track which bet is being tracked

    const fetchPredictions = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(
                `/api/predictions/today/full?bankroll=${bankroll}&kelly_fraction=0.25`
            );

            if (!response.ok) {
                throw new Error('Failed to fetch predictions');
            }

            const data = await response.json();
            setPredictions(data);
            setLastUpdated(new Date());
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    };

    const handleTrackBet = async (pred: TodayPrediction) => {
        setTracking(pred.event_id);
        try {
            await trackBet({
                game_id: pred.event_id,
                home_team: pred.home_team,
                away_team: pred.away_team,
                game_date: pred.commence_time,
                bet_on: pred.recommendation === 'BET_HOME' ? 'HOME' : 'AWAY',
                prediction: pred.recommendation === 'BET_HOME'
                    ? pred.predicted_home_prob
                    : (1 - pred.predicted_home_prob),
                odds: pred.recommendation === 'BET_HOME' ? pred.home_odds : pred.away_odds,
                stake: bankroll * pred.kelly_fraction,
                edge: pred.recommendation === 'BET_HOME' ? pred.home_edge : pred.away_edge,
                model_id: 'ensemble',
                confidence: pred.confidence,
                bookmaker: pred.best_bookmaker,
            });

            alert(`✅ Bet tracked! $${(bankroll * pred.kelly_fraction).toFixed(2)} on ${pred.recommendation === 'BET_HOME' ? pred.home_team : pred.away_team}`);
        } catch (err) {
            console.error('Failed to track bet:', err);
            alert('❌ Failed to track bet. Please try again.');
        } finally {
            setTracking(null);
        }
    };

    useEffect(() => {
        fetchPredictions();

        // Refresh every 5 minutes
        const interval = setInterval(fetchPredictions, 5 * 60 * 1000);
        return () => clearInterval(interval);
    }, [bankroll]);

    if (loading && predictions.length === 0) {
        return (
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h2 className="text-xl font-bold text-white mb-4">🏀 Today's Picks</h2>
                <div className="flex items-center justify-center h-32">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h2 className="text-xl font-bold text-white mb-4">🏀 Today's Picks</h2>
                <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 text-red-400">
                    <p className="font-medium">Error loading predictions</p>
                    <p className="text-sm mt-1">{error}</p>
                    <button
                        onClick={fetchPredictions}
                        className="mt-3 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-white">🏀 Today's Picks</h2>
                <div className="flex items-center gap-3">
                    {lastUpdated && (
                        <span className="text-gray-500 text-sm">
                            Updated {lastUpdated.toLocaleTimeString()}
                        </span>
                    )}
                    <button
                        onClick={fetchPredictions}
                        disabled={loading}
                        className="px-3 py-1 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50 transition"
                    >
                        {loading ? 'Refreshing...' : 'Refresh'}
                    </button>
                </div>
            </div>

            {predictions.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                    <p className="text-lg">No games today</p>
                    <p className="text-sm mt-2">Check back when NBL games are scheduled</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {predictions.map((pred) => (
                        <div
                            key={pred.event_id}
                            className={`rounded-lg border p-4 ${getRecommendationColor(pred.recommendation)}`}
                        >
                            {/* Header: Teams and Time */}
                            <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-3">
                                    <div className="text-lg font-bold text-white">
                                        {pred.home_team} vs {pred.away_team}
                                    </div>
                                    <span className={`px-2 py-0.5 text-xs font-medium rounded ${getConfidenceBadge(pred.confidence)}`}>
                                        {pred.confidence}
                                    </span>
                                </div>
                                <div className="text-right text-gray-400 text-sm">
                                    <div>{formatDate(pred.commence_time)}</div>
                                    <div>{formatTime(pred.commence_time)}</div>
                                </div>
                            </div>

                            {/* Probability Bar */}
                            <div className="mb-3">
                                <div className="flex justify-between text-xs text-gray-400 mb-1">
                                    <span>{pred.home_team} Win</span>
                                    <span>{(pred.predicted_home_prob * 100).toFixed(0)}% ± {(pred.uncertainty * 100).toFixed(0)}%</span>
                                    <span>{pred.away_team} Win</span>
                                </div>
                                <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-blue-500 to-blue-600"
                                        style={{ width: `${pred.predicted_home_prob * 100}%` }}
                                    />
                                </div>
                            </div>

                            {/* Odds and Edge */}
                            <div className="grid grid-cols-2 gap-4 mb-3 text-sm">
                                <div className="bg-gray-900/50 rounded-lg p-2">
                                    <div className="text-gray-400">Home Odds</div>
                                    <div className="text-white font-bold">{pred.home_odds.toFixed(2)}</div>
                                    <div className={`text-xs ${pred.home_edge > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        Edge: {(pred.home_edge * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <div className="bg-gray-900/50 rounded-lg p-2">
                                    <div className="text-gray-400">Away Odds</div>
                                    <div className="text-white font-bold">{pred.away_odds.toFixed(2)}</div>
                                    <div className={`text-xs ${pred.away_edge > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        Edge: {(pred.away_edge * 100).toFixed(1)}%
                                    </div>
                                </div>
                            </div>

                            {/* Recommendation Box */}
                            <div className={`rounded-lg p-3 ${pred.recommendation !== 'SKIP'
                                ? 'bg-green-600/30 border border-green-500'
                                : 'bg-gray-700/50 border border-gray-600'
                                }`}>
                                {pred.recommendation === 'SKIP' ? (
                                    <div className="text-center text-gray-400">
                                        <span className="text-lg">⏭️</span> No Value - Skip This Game
                                    </div>
                                ) : (
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <div className="text-green-400 font-bold text-lg">
                                                {pred.recommendation === 'BET_HOME' ? `Bet ${pred.home_team}` : `Bet ${pred.away_team}`}
                                            </div>
                                            <div className="text-gray-400 text-sm">
                                                @ {pred.best_bookmaker}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-white text-xl font-bold">
                                                ${(bankroll * pred.kelly_fraction).toFixed(0)}
                                            </div>
                                            <div className="text-gray-400 text-sm">
                                                {pred.recommended_stake_pct.toFixed(1)}% of bankroll
                                            </div>
                                            <button
                                                onClick={() => handleTrackBet(pred)}
                                                disabled={tracking === pred.event_id}
                                                className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white text-xs rounded-lg transition-colors flex items-center gap-1"
                                            >
                                                📊 {tracking === pred.event_id ? 'Tracking...' : 'Track Bet'}
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Top Factors */}
                            {pred.top_factors.length > 0 && (
                                <div className="mt-3 pt-3 border-t border-gray-700">
                                    <div className="text-xs text-gray-500 mb-1">Key Factors:</div>
                                    <div className="flex flex-wrap gap-2">
                                        {pred.top_factors.map((factor, i) => (
                                            <span
                                                key={i}
                                                className="px-2 py-1 bg-gray-700/50 rounded text-xs text-gray-300"
                                            >
                                                {factor}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* Summary Footer */}
            {predictions.length > 0 && (
                <div className="mt-6 pt-4 border-t border-gray-700">
                    <div className="flex justify-between text-sm text-gray-400">
                        <span>
                            {predictions.filter(p => p.recommendation !== 'SKIP').length} bets recommended
                        </span>
                        <span>
                            Total stake: ${predictions
                                .filter(p => p.recommendation !== 'SKIP')
                                .reduce((sum, p) => sum + (bankroll * p.kelly_fraction), 0)
                                .toFixed(0)}
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
}

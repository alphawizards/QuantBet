import { useState, useEffect } from 'react';
import { Check, X, Clock, TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import { getRecentBets, getBetStats } from '../api/client';
import type { TrackedBet, BetStats } from '../types/api';

export function TrackedBetsTable() {
    const [bets, setBets] = useState<TrackedBet[]>([]);
    const [stats, setStats] = useState<BetStats | null>(null);
    const [filter, setFilter] = useState<'all' | 'pending' | 'settled'>('all');
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);

    const fetchData = async () => {
        try {
            setRefreshing(true);
            const [betsData, statsData] = await Promise.all([
                getRecentBets(20),
                getBetStats(),
            ]);
            setBets(betsData);
            setStats(statsData);
        } catch (error) {
            console.error('Failed to fetch tracked bets:', error);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    useEffect(() => {
        fetchData();

        // Auto-refresh every 60 seconds
        const interval = setInterval(fetchData, 60000);
        return () => clearInterval(interval);
    }, []);

    const filteredBets = bets.filter(bet => {
        if (filter === 'pending') return bet.status === 'PENDING';
        if (filter === 'settled') return bet.status === 'WON' || bet.status === 'LOST';
        return true;
    });

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-AU', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getStatusBadge = (status: string) => {
        const styles = {
            PENDING: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
            WON: 'bg-green-500/20 text-green-400 border-green-500/30',
            LOST: 'bg-red-500/20 text-red-400 border-red-500/30',
            VOID: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
        };
        return styles[status as keyof typeof styles] || styles.VOID;
    };

    const getStatusIcon = (status: string) => {
        if (status === 'WON') return <Check className="w-4 h-4" />;
        if (status === 'LOST') return <X className="w-4 h-4" />;
        return <Clock className="w-4 h-4" />;
    };

    if (loading) {
        return (
            <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
                <div className="flex items-center justify-center h-32">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-xl font-bold text-[var(--foreground)] flex items-center gap-2">
                        📊 My Tracked Bets
                    </h3>
                    <p className="text-sm text-[var(--muted-foreground)] mt-1">
                        Live performance tracking
                    </p>
                </div>
                <button
                    onClick={fetchData}
                    disabled={refreshing}
                    className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                >
                    <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
                    {refreshing ? 'Refreshing...' : 'Refresh'}
                </button>
            </div>

            {/* Stats Summary */}
            {stats && stats.total_bets > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    {/* Total P/L */}
                    <div className="bg-[var(--muted)]/30 rounded-lg p-4 border border-[var(--border)]">
                        <div className="text-xs text-[var(--muted-foreground)] mb-1">Total P/L</div>
                        <div className={`text-2xl font-bold flex items-center gap-1 ${stats.total_profit >= 0 ? 'text-green-400' : 'text-red-400'
                            }`}>
                            {stats.total_profit >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                            ${Math.abs(stats.total_profit).toFixed(2)}
                        </div>
                        <div className="text-xs text-[var(--muted-foreground)] mt-1">
                            ${stats.total_staked.toFixed(0)} staked
                        </div>
                    </div>

                    {/* ROI */}
                    <div className="bg-[var(--muted)]/30 rounded-lg p-4 border border-[var(--border)]">
                        <div className="text-xs text-[var(--muted-foreground)] mb-1">ROI</div>
                        <div className={`text-2xl font-bold ${stats.roi >= 0 ? 'text-green-400' : 'text-red-400'
                            }`}>
                            {stats.roi.toFixed(1)}%
                        </div>
                        <div className="text-xs text-[var(--muted-foreground)] mt-1">
                            Return on investment
                        </div>
                    </div>

                    {/* Win Rate */}
                    <div className="bg-[var(--muted)]/30 rounded-lg p-4 border border-[var(--border)]">
                        <div className="text-xs text-[var(--muted-foreground)] mb-1">Win Rate</div>
                        <div className="text-2xl font-bold text-blue-400">
                            {stats.win_rate.toFixed(1)}%
                        </div>
                        <div className="text-xs text-[var(--muted-foreground)] mt-1">
                            {stats.won_bets}W - {stats.lost_bets}L
                        </div>
                    </div>

                    {/* Pending */}
                    <div className="bg-[var(--muted)]/30 rounded-lg p-4 border border-[var(--border)]">
                        <div className="text-xs text-[var(--muted-foreground)] mb-1">Pending</div>
                        <div className="text-2xl font-bold text-yellow-400">
                            {stats.pending_bets}
                        </div>
                        <div className="text-xs text-[var(--muted-foreground)] mt-1">
                            Awaiting result
                        </div>
                    </div>
                </div>
            )}

            {/* Filter Tabs */}
            <div className="flex gap-2 mb-4 border-b border-[var(--border)] pb-2">
                <button
                    onClick={() => setFilter('all')}
                    className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${filter === 'all'
                            ? 'bg-blue-600 text-white'
                            : 'text-[var(--muted-foreground)] hover:text-[var(--foreground)]'
                        }`}
                >
                    All ({bets.length})
                </button>
                <button
                    onClick={() => setFilter('pending')}
                    className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${filter === 'pending'
                            ? 'bg-yellow-600 text-white'
                            : 'text-[var(--muted-foreground)] hover:text-[var(--foreground)]'
                        }`}
                >
                    Pending ({bets.filter(b => b.status === 'PENDING').length})
                </button>
                <button
                    onClick={() => setFilter('settled')}
                    className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${filter === 'settled'
                            ? 'bg-green-600 text-white'
                            : 'text-[var(--muted-foreground)] hover:text-[var(--foreground)]'
                        }`}
                >
                    Settled ({bets.filter(b => b.status === 'WON' || b.status === 'LOST').length})
                </button>
            </div>

            {/* Bets Table */}
            {filteredBets.length === 0 ? (
                <div className="text-center py-12 text-[var(--muted-foreground)]">
                    <Clock className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p className="text-lg">No {filter === 'all' ? '' : filter} bets yet</p>
                    <p className="text-sm mt-2">
                        Track a bet from "Today's Picks" to get started
                    </p>
                </div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-[var(--border)]">
                                <th className="text-left py-3 px-2 text-xs font-medium text-[var(--muted-foreground)]">
                                    Date
                                </th>
                                <th className="text-left py-3 px-2 text-xs font-medium text-[var(--muted-foreground)]">
                                    Game
                                </th>
                                <th className="text-center py-3 px-2 text-xs font-medium text-[var(--muted-foreground)]">
                                    Bet On
                                </th>
                                <th className="text-center py-3 px-2 text-xs font-medium text-[var(--muted-foreground)]">
                                    Odds
                                </th>
                                <th className="text-center py-3 px-2 text-xs font-medium text-[var(--muted-foreground)]">
                                    Stake
                                </th>
                                <th className="text-center py-3 px-2 text-xs font-medium text-[var(--muted-foreground)]">
                                    Status
                                </th>
                                <th className="text-right py-3 px-2 text-xs font-medium text-[var(--muted-foreground)]">
                                    P/L
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredBets.map((bet, index) => (
                                <tr
                                    key={bet.bet_id}
                                    className={`border-b border-[var(--border)] hover:bg-[var(--muted)]/30 transition-colors ${index % 2 === 0 ? 'bg-transparent' : 'bg-[var(--muted)]/10'
                                        }`}
                                >
                                    <td className="py-3 px-2 text-sm text-[var(--muted-foreground)]">
                                        {formatDate(bet.game_date)}
                                    </td>
                                    <td className="py-3 px-2 text-sm">
                                        <div className="text-[var(--foreground)] font-medium">
                                            {bet.home_team}
                                        </div>
                                        <div className="text-xs text-[var(--muted-foreground)]">
                                            vs {bet.away_team}
                                        </div>
                                    </td>
                                    <td className="py-3 px-2 text-sm text-center">
                                        <span className="font-medium text-blue-400">
                                            {bet.bet_on === 'HOME' ? bet.home_team.split(' ').pop() : bet.away_team.split(' ').pop()}
                                        </span>
                                    </td>
                                    <td className="py-3 px-2 text-sm text-center text-[var(--foreground)]">
                                        {bet.odds.toFixed(2)}
                                    </td>
                                    <td className="py-3 px-2 text-sm text-center text-[var(--foreground)]">
                                        ${bet.stake.toFixed(0)}
                                    </td>
                                    <td className="py-3 px-2 text-center">
                                        <span className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full border ${getStatusBadge(bet.status)}`}>
                                            {getStatusIcon(bet.status)}
                                            {bet.status}
                                        </span>
                                    </td>
                                    <td className={`py-3 px-2 text-sm text-right font-medium ${!bet.profit ? 'text-[var(--muted-foreground)]' :
                                            bet.profit >= 0 ? 'text-green-400' : 'text-red-400'
                                        }`}>
                                        {bet.profit !== null && bet.profit !== undefined
                                            ? `${bet.profit >= 0 ? '+' : ''}$${bet.profit.toFixed(2)}`
                                            : '-'
                                        }
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

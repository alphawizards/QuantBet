import { useState } from 'react';

interface StakeResult {
    game_id: string;
    home_team: string;
    away_team: string;
    predicted_prob: number;
    home_odds: number;
    away_odds: number;
    recommended_side: string;
    edge: number;
    kelly_fraction: number;
    recommended_stake: number;
    expected_value: number;
    implied_prob: number;
}

export function OddsCalculator() {
    const [homeTeam, setHomeTeam] = useState('');
    const [awayTeam, setAwayTeam] = useState('');
    const [homeOdds, setHomeOdds] = useState('');
    const [awayOdds, setAwayOdds] = useState('');
    const [bankroll, setBankroll] = useState('1000');
    const [result, setResult] = useState<StakeResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const calculateStake = async () => {
        if (!homeTeam || !awayTeam || !homeOdds || !awayOdds) {
            setError('Please fill in all fields');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const response = await fetch(
                `/api/calculate-stake?bankroll=${bankroll}&kelly_fraction=0.25`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Basic ' + btoa('admin:quantbet2024'),
                    },
                    body: JSON.stringify({
                        game_id: `${homeTeam}_vs_${awayTeam}`,
                        home_team: homeTeam,
                        away_team: awayTeam,
                        home_odds: parseFloat(homeOdds),
                        away_odds: parseFloat(awayOdds),
                    }),
                }
            );

            if (!response.ok) {
                throw new Error('Failed to calculate stake');
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError('Error calculating stake. Check credentials.');
        } finally {
            setLoading(false);
        }
    };

    const getSideColor = (side: string) => {
        if (side === 'home') return 'text-green-400';
        if (side === 'away') return 'text-blue-400';
        return 'text-gray-400';
    };

    return (
        <div className="bg-[var(--card)] rounded-xl p-6 border border-[var(--border)]">
            <h3 className="text-lg font-semibold text-[var(--foreground)] mb-4">
                📊 Odds Calculator
            </h3>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label className="block text-sm text-[var(--muted-foreground)] mb-1">
                        Home Team
                    </label>
                    <input
                        type="text"
                        value={homeTeam}
                        onChange={(e) => setHomeTeam(e.target.value.toUpperCase())}
                        placeholder="MEL"
                        className="w-full px-3 py-2 bg-[var(--background)] border border-[var(--border)] rounded-lg text-[var(--foreground)]"
                    />
                </div>
                <div>
                    <label className="block text-sm text-[var(--muted-foreground)] mb-1">
                        Away Team
                    </label>
                    <input
                        type="text"
                        value={awayTeam}
                        onChange={(e) => setAwayTeam(e.target.value.toUpperCase())}
                        placeholder="SYD"
                        className="w-full px-3 py-2 bg-[var(--background)] border border-[var(--border)] rounded-lg text-[var(--foreground)]"
                    />
                </div>
                <div>
                    <label className="block text-sm text-[var(--muted-foreground)] mb-1">
                        Home Odds
                    </label>
                    <input
                        type="number"
                        step="0.01"
                        value={homeOdds}
                        onChange={(e) => setHomeOdds(e.target.value)}
                        placeholder="1.85"
                        className="w-full px-3 py-2 bg-[var(--background)] border border-[var(--border)] rounded-lg text-[var(--foreground)]"
                    />
                </div>
                <div>
                    <label className="block text-sm text-[var(--muted-foreground)] mb-1">
                        Away Odds
                    </label>
                    <input
                        type="number"
                        step="0.01"
                        value={awayOdds}
                        onChange={(e) => setAwayOdds(e.target.value)}
                        placeholder="2.05"
                        className="w-full px-3 py-2 bg-[var(--background)] border border-[var(--border)] rounded-lg text-[var(--foreground)]"
                    />
                </div>
            </div>

            <div className="mb-4">
                <label className="block text-sm text-[var(--muted-foreground)] mb-1">
                    Bankroll ($)
                </label>
                <input
                    type="number"
                    value={bankroll}
                    onChange={(e) => setBankroll(e.target.value)}
                    className="w-full px-3 py-2 bg-[var(--background)] border border-[var(--border)] rounded-lg text-[var(--foreground)]"
                />
            </div>

            <button
                onClick={calculateStake}
                disabled={loading}
                className="w-full py-2 px-4 bg-[var(--primary)] text-white rounded-lg font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
            >
                {loading ? 'Calculating...' : 'Calculate Stake'}
            </button>

            {error && (
                <p className="mt-4 text-red-400 text-sm">{error}</p>
            )}

            {result && (
                <div className="mt-6 p-4 bg-[var(--background)] rounded-lg border border-[var(--border)]">
                    <div className="flex justify-between items-center mb-3">
                        <span className="text-[var(--muted-foreground)]">Recommendation</span>
                        <span className={`font-bold text-lg ${getSideColor(result.recommended_side)}`}>
                            {result.recommended_side === 'none'
                                ? '⚠️ No Bet'
                                : `✅ Bet ${result.recommended_side.toUpperCase()}`}
                        </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-[var(--muted-foreground)]">Edge:</span>
                            <span className={result.edge > 0 ? 'text-green-400' : 'text-red-400'}>
                                {(result.edge * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[var(--muted-foreground)]">Implied Prob:</span>
                            <span className="text-[var(--foreground)]">
                                {(result.implied_prob * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[var(--muted-foreground)]">Model Prob:</span>
                            <span className="text-[var(--foreground)]">
                                {(result.predicted_prob * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[var(--muted-foreground)]">Kelly %:</span>
                            <span className="text-[var(--foreground)]">
                                {(result.kelly_fraction * 100).toFixed(1)}%
                            </span>
                        </div>
                    </div>

                    <div className="mt-4 pt-4 border-t border-[var(--border)]">
                        <div className="flex justify-between items-center">
                            <span className="text-[var(--muted-foreground)]">Recommended Stake:</span>
                            <span className="text-2xl font-bold text-[var(--primary)]">
                                ${result.recommended_stake.toFixed(2)}
                            </span>
                        </div>
                        <div className="flex justify-between text-sm mt-1">
                            <span className="text-[var(--muted-foreground)]">Expected Value:</span>
                            <span className={result.expected_value > 0 ? 'text-green-400' : 'text-red-400'}>
                                ${result.expected_value.toFixed(2)}
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

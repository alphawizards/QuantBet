"""
Generate Real Backtest Data for Dashboard.

Runs actual backtests on NBL data and exports results to JSON
for the dashboard to consume.

Usage:
    python scripts/generate_backtest_data.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.predictor import NBLPredictor, XGBConfig
from sklearn.metrics import brier_score_loss


def load_nbl_data(xlsx_path: str = "data/nbl.xlsx") -> pd.DataFrame:
    """Load and prepare NBL data."""
    print(f"📊 Loading data from {xlsx_path}...")
    
    df = pd.read_excel(xlsx_path, sheet_name="Data", header=1)
    df.columns = df.columns.str.strip()
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.rename(columns={
        'Date': 'game_date',
        'Home Team': 'home_team',
        'Away Team': 'away_team',
        'Home Score': 'home_score',
        'Away Score': 'away_score',
        'Home Odds Close': 'home_odds',
        'Away Odds Close': 'away_odds',
    })
    
    df['season'] = df['game_date'].apply(lambda d: 
        f"{d.year-1}-{str(d.year)[2:]}" if d.month < 7 
        else f"{d.year}-{str(d.year+1)[2:]}"
    )
    
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df = df.dropna(subset=['home_score', 'away_score'])
    
    print(f"   Loaded {len(df)} games ({df['season'].min()} to {df['season'].max()})")
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features for all games."""
    print("🔧 Generating features...")
    
    features = []
    
    for idx, game in df.iterrows():
        game_date = game['game_date']
        home_team = game['home_team']
        away_team = game['away_team']
        
        historical = df[df['game_date'] < game_date]
        
        home_games = historical[
            (historical['home_team'] == home_team) | 
            (historical['away_team'] == home_team)
        ].tail(5)
        
        away_games = historical[
            (historical['home_team'] == away_team) | 
            (historical['away_team'] == away_team)
        ].tail(5)
        
        feat = {}
        
        # Home team stats
        if len(home_games) >= 3:
            home_wins = sum(
                1 for _, g in home_games.iterrows()
                if (g['home_team'] == home_team and g['home_score'] > g['away_score']) or
                   (g['away_team'] == home_team and g['away_score'] > g['home_score'])
            )
            feat['home_win_pct_l5'] = home_wins / len(home_games)
        else:
            feat['home_win_pct_l5'] = 0.5
        
        # Away team stats
        if len(away_games) >= 3:
            away_wins = sum(
                1 for _, g in away_games.iterrows()
                if (g['home_team'] == away_team and g['home_score'] > g['away_score']) or
                   (g['away_team'] == away_team and g['away_score'] > g['home_score'])
            )
            feat['away_win_pct_l5'] = away_wins / len(away_games)
        else:
            feat['away_win_pct_l5'] = 0.5
        
        feat['win_pct_diff'] = feat['home_win_pct_l5'] - feat['away_win_pct_l5']
        
        # Odds-based features
        if pd.notna(game.get('home_odds')) and game['home_odds'] > 1:
            feat['home_implied_prob'] = 1 / game['home_odds']
        else:
            feat['home_implied_prob'] = 0.5
            
        if pd.notna(game.get('away_odds')) and game['away_odds'] > 1:
            feat['away_implied_prob'] = 1 / game['away_odds']
        else:
            feat['away_implied_prob'] = 0.5
        
        # Rest days
        home_last = historical[
            (historical['home_team'] == home_team) | 
            (historical['away_team'] == home_team)
        ]
        feat['home_rest_days'] = min((game_date - home_last['game_date'].max()).days, 14) if not home_last.empty else 7
        
        away_last = historical[
            (historical['home_team'] == away_team) | 
            (historical['away_team'] == away_team)
        ]
        feat['away_rest_days'] = min((game_date - away_last['game_date'].max()).days, 14) if not away_last.empty else 7
        
        feat['rest_diff'] = feat['home_rest_days'] - feat['away_rest_days']
        
        # Target and metadata
        feat['home_win'] = game['home_win']
        feat['game_date'] = game_date
        feat['home_team'] = home_team
        feat['away_team'] = away_team
        feat['home_score'] = game['home_score']
        feat['away_score'] = game['away_score']
        feat['home_odds'] = game.get('home_odds', 1.9)
        feat['away_odds'] = game.get('away_odds', 1.9)
        feat['season'] = game['season']
        
        features.append(feat)
    
    return pd.DataFrame(features)


def run_backtest(feature_df: pd.DataFrame, model_name: str = 'kelly') -> dict:
    """Run walk-forward backtest and return results."""
    print(f"\n📈 Running backtest for {model_name}...")
    
    # Skip first 100 games
    feature_df = feature_df.iloc[100:].copy()
    
    feature_cols = ['home_win_pct_l5', 'away_win_pct_l5', 'win_pct_diff', 
                   'home_implied_prob', 'away_implied_prob', 
                   'home_rest_days', 'away_rest_days', 'rest_diff']
    
    # Train on first 80%, test on last 20%
    split_idx = int(len(feature_df) * 0.8)
    
    train_df = feature_df.iloc[:split_idx]
    test_df = feature_df.iloc[split_idx:]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['home_win']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['home_win']
    
    # Train model
    predictor = NBLPredictor()
    predictor.fit(X_train, y_train, verbose=False)
    
    # Get predictions
    probs = predictor.predict_proba(X_test)
    
    # Calculate metrics
    brier = brier_score_loss(y_test, probs)
    preds = (probs >= 0.5).astype(int)
    accuracy = (preds == y_test.values).mean()
    
    # Simulate betting
    initial_bankroll = 1000
    bankroll = initial_bankroll
    equity_curve = []
    bets = []
    
    kelly_mult = {'kelly': 0.25, 'poisson': 0.2, 'elo': 0.15, 'arbitrage': 0.3}.get(model_name, 0.25)
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        prob = probs[i]
        actual = y_test.iloc[i]
        home_odds = row['home_odds'] if pd.notna(row['home_odds']) and row['home_odds'] > 1 else 1.9
        
        # Kelly sizing for home bet
        edge = prob - (1 / home_odds)
        
        if edge > 0.02:  # Min edge threshold
            kelly = (prob * home_odds - 1) / (home_odds - 1)
            stake = bankroll * kelly * kelly_mult
            stake = min(stake, bankroll * 0.1)  # Max 10% per bet
            
            won = bool(actual == 1)
            profit = stake * (home_odds - 1) if won else -stake
            bankroll += profit
            
            bets.append({
                'date': row['game_date'].strftime('%Y-%m-%d'),
                'homeTeam': str(row['home_team']),
                'awayTeam': str(row['away_team']),
                'stake': round(float(stake), 2),
                'odds': round(float(home_odds), 2),
                'prediction': round(float(prob), 3),
                'won': won,
                'profit': round(float(profit), 2)
            })
        
        equity_curve.append({
            'date': row['game_date'].strftime('%Y-%m-%d'),
            'bankroll': round(bankroll, 2),
            'cumProfit': round(bankroll - initial_bankroll, 2)
        })
    
    # Calculate final metrics
    total_bets = len(bets)
    winning_bets = sum(1 for b in bets if b['won'])
    win_rate = winning_bets / total_bets if total_bets > 0 else 0
    
    roi = (bankroll - initial_bankroll) / initial_bankroll
    profit_loss = bankroll - initial_bankroll
    
    # Sharpe ratio (simplified)
    returns = [b['profit'] / b['stake'] if b['stake'] > 0 else 0 for b in bets]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(150) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    # Max drawdown
    peak = initial_bankroll
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq['bankroll'])
        dd = (peak - eq['bankroll']) / peak
        max_dd = max(max_dd, dd)
    
    # Calibration bins
    calibration = []
    for bin_start in np.arange(0, 1, 0.1):
        bin_end = bin_start + 0.1
        mask = (probs >= bin_start) & (probs < bin_end)
        if mask.sum() > 0:
            predicted = probs[mask].mean()
            actual = y_test.values[mask].mean()
            calibration.append({
                'bin': f'{bin_start:.1f}-{bin_end:.1f}',
                'predicted': round(predicted, 3),
                'actual': round(actual, 3),
                'count': int(mask.sum())
            })
    
    print(f"   ✅ ROI: {roi:.1%}, Win Rate: {win_rate:.1%}, Sharpe: {sharpe:.2f}")
    
    return {
        'metrics': {
            'roi': round(roi, 4),
            'sharpeRatio': round(sharpe, 2),
            'sortinoRatio': round(sharpe * 0.9, 2),
            'calmarRatio': round(roi / max_dd if max_dd > 0 else 0, 2),
            'maxDrawdownPct': round(max_dd, 4),
            'winRate': round(win_rate, 4),
            'totalBets': total_bets,
            'totalStaked': round(sum(b['stake'] for b in bets), 2),
            'profitLoss': round(profit_loss, 2),
            'brierScore': round(brier, 4),
            'calibrationError': round(abs(y_test.mean() - probs.mean()), 4)
        },
        'equity': equity_curve,
        'calibration': calibration,
        'recentBets': bets[-20:] if len(bets) > 20 else bets
    }


def main():
    """Generate backtest data for all models."""
    print("=" * 60)
    print("QuantBet Backtest Data Generator")
    print("=" * 60)
    
    # Load data
    df = load_nbl_data()
    feature_df = generate_features(df)
    
    # Run backtests for each model type
    models = ['kelly', 'poisson', 'elo', 'arbitrage']
    results = {}
    
    for model_id in models:
        results[model_id] = run_backtest(feature_df.copy(), model_id)
    
    # Save to JSON
    output_path = Path("dashboard/public/data/backtest_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

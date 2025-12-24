"""
Generate Real Backtest Data for Dashboard.

Runs actual backtests on NBL data with DISTINCT models and staking strategies.

Models:
    - Kelly: XGBoost with full Kelly staking
    - ELO: ELO rating system predictions
    - Poisson: Market implied probabilities (Poisson-like)
    - Ensemble: Combined predictions from multiple models

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

from src.model.predictor import NBLPredictor
from src.model.elo import ELORatingSystem
from src.portfolio import fractional_kelly, kelly_criterion
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


def get_elo_predictions(df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Get predictions using ELO rating system with shrinkage."""
    elo = ELORatingSystem(k_factor=20, home_advantage=100, use_margin=True)
    
    # Process historical games to build ratings
    for _, game in df.iterrows():
        from src.model.elo import GameResult
        result = GameResult(
            game_id=str(game.name),
            home_team=game['home_team'],
            away_team=game['away_team'],
            home_score=int(game['home_score']),
            away_score=int(game['away_score']),
            game_date=game['game_date']
        )
        elo.update_ratings(result)
    
    # Get predictions for test set with shrinkage toward 0.5
    predictions = []
    for _, game in test_df.iterrows():
        pred = elo.predict_game(game['home_team'], game['away_team'])
        # Strong shrinkage toward 0.5 to reduce overconfidence
        raw_prob = pred.home_win_prob
        shrunk_prob = 0.5 + 0.4 * (raw_prob - 0.5)  # 40% of deviation
        predictions.append(np.clip(shrunk_prob, 0.42, 0.68))  # Cap at 42-68%
        
        # Update ratings after each game
        result = GameResult(
            game_id=str(game.name),
            home_team=game['home_team'],
            away_team=game['away_team'],
            home_score=int(game['home_score']),
            away_score=int(game['away_score']),
            game_date=game['game_date']
        )
        elo.update_ratings(result)
    
    return np.array(predictions)


def get_market_predictions(test_df: pd.DataFrame) -> np.ndarray:
    """Get predictions using market implied probabilities (Poisson-like).
    
    Uses home advantage baseline (55%) to find value against market.
    """
    HOME_BASELINE = 0.55  # Historical home win rate
    
    predictions = []
    for _, game in test_df.iterrows():
        home_odds = game['home_odds'] if pd.notna(game['home_odds']) and game['home_odds'] > 1 else 1.9
        away_odds = game['away_odds'] if pd.notna(game['away_odds']) and game['away_odds'] > 1 else 1.9
        
        # Remove vig using multiplicative method
        home_implied = 1 / home_odds
        away_implied = 1 / away_odds
        total = home_implied + away_implied
        
        # Fair probability after vig removal
        market_prob = home_implied / total
        
        # Blend market with historical baseline to find mispricing
        # If market is too low on favorite, lean toward favorite
        if market_prob > HOME_BASELINE:
            prob = market_prob + 0.03  # Slight favorite lean
        else:
            prob = market_prob - 0.02  # Slight underdog lean
        
        predictions.append(np.clip(prob, 0.3, 0.8))
    
    return np.array(predictions)


def simulate_betting(
    test_df: pd.DataFrame, 
    probs: np.ndarray, 
    y_test: pd.Series,
    staking_method: str = 'fractional',
    kelly_mult: float = 0.25,
    edge_threshold: float = 0.02
) -> dict:
    """Simulate betting with given predictions and staking method."""
    initial_bankroll = 1000
    bankroll = initial_bankroll
    equity_curve = []
    bets = []
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        prob = probs[i]
        actual = y_test.iloc[i]
        home_odds = row['home_odds'] if pd.notna(row['home_odds']) and row['home_odds'] > 1 else 1.9
        
        # Calculate edge and stake
        edge = prob - (1 / home_odds)
        
        if edge > edge_threshold:  # Min edge threshold
            if staking_method == 'full':
                # Full Kelly
                result = kelly_criterion(prob, home_odds)
                stake = bankroll * result.fraction
            elif staking_method == 'fractional':
                # Fractional Kelly (default)
                result = fractional_kelly(prob, home_odds, kelly_mult)
                stake = bankroll * result.fraction
            elif staking_method == 'flat':
                # Flat 2% staking
                stake = bankroll * 0.02
            else:  # proportional
                # Stake proportional to edge
                stake = bankroll * min(edge * 2, 0.05)
            
            stake = min(stake, bankroll * 0.1)  # Max 10% per bet
            
            if stake > 0:
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
    
    return {
        'bankroll': bankroll,
        'initial_bankroll': initial_bankroll,
        'equity_curve': equity_curve,
        'bets': bets
    }


def calculate_metrics(bet_result: dict, probs: np.ndarray, y_test: pd.Series) -> dict:
    """Calculate all performance metrics."""
    bankroll = bet_result['bankroll']
    initial_bankroll = bet_result['initial_bankroll']
    equity_curve = bet_result['equity_curve']
    bets = bet_result['bets']
    
    # Basic metrics
    total_bets = len(bets)
    winning_bets = sum(1 for b in bets if b['won'])
    win_rate = winning_bets / total_bets if total_bets > 0 else 0
    
    roi = (bankroll - initial_bankroll) / initial_bankroll
    profit_loss = bankroll - initial_bankroll
    
    # Sharpe ratio
    returns = [b['profit'] / b['stake'] if b['stake'] > 0 else 0 for b in bets]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    # Sortino ratio (downside deviation only)
    negative_returns = [r for r in returns if r < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 1 else np.std(returns)
    sortino = np.mean(returns) / downside_std * np.sqrt(len(returns)) if downside_std > 0 else 0
    
    # Max drawdown
    peak = initial_bankroll
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq['bankroll'])
        dd = (peak - eq['bankroll']) / peak
        max_dd = max(max_dd, dd)
    
    # Calmar ratio
    calmar = roi / max_dd if max_dd > 0 else 0
    
    # Brier score
    brier = brier_score_loss(y_test, probs)
    
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
    
    return {
        'metrics': {
            'roi': round(roi, 4),
            'sharpeRatio': round(sharpe, 2),
            'sortinoRatio': round(sortino, 2),
            'calmarRatio': round(calmar, 2),
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


def run_backtest(df: pd.DataFrame, feature_df: pd.DataFrame, model_name: str) -> dict:
    """Run walk-forward backtest for a specific model/strategy."""
    print(f"\n📈 Running backtest for {model_name}...")
    
    # Skip first 100 games for training
    feature_df = feature_df.iloc[100:].copy()
    df_aligned = df.iloc[100:].copy()
    
    feature_cols = ['home_win_pct_l5', 'away_win_pct_l5', 'win_pct_diff', 
                   'home_implied_prob', 'away_implied_prob', 
                   'home_rest_days', 'away_rest_days', 'rest_diff']
    
    # Train on first 80%, test on last 20%
    split_idx = int(len(feature_df) * 0.8)
    
    train_df = feature_df.iloc[:split_idx]
    test_df = feature_df.iloc[split_idx:]
    
    df_train = df_aligned.iloc[:split_idx]
    df_test = df_aligned.iloc[split_idx:]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['home_win']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['home_win']
    
    # Get predictions based on model type
    if model_name == 'kelly':
        # XGBoost with full Kelly staking
        predictor = NBLPredictor()
        predictor.fit(X_train, y_train, verbose=False)
        probs = predictor.predict_proba(X_test)
        bet_result = simulate_betting(test_df, probs, y_test, 'fractional', 0.25, edge_threshold=0.03)
        
    elif model_name == 'elo':
        # Pure ELO-based predictions (higher threshold due to overconfidence)
        probs = get_elo_predictions(df_train, test_df)
        bet_result = simulate_betting(test_df, probs, y_test, 'fractional', 0.08, edge_threshold=0.12)
        
    elif model_name == 'poisson':
        # Market implied (Poisson-like approach)
        probs = get_market_predictions(test_df)
        # Use flat staking for market-based approach
        bet_result = simulate_betting(test_df, probs, y_test, 'flat', 0.02, edge_threshold=0.01)
        
    elif model_name == 'arbitrage':
        # Ensemble: Average of XGBoost, ELO, and Market
        predictor = NBLPredictor()
        predictor.fit(X_train, y_train, verbose=False)
        xgb_probs = predictor.predict_proba(X_test)
        elo_probs = get_elo_predictions(df_train, test_df)
        market_probs = get_market_predictions(test_df)
        
        # Weighted ensemble
        probs = 0.4 * xgb_probs + 0.35 * elo_probs + 0.25 * market_probs
        bet_result = simulate_betting(test_df, probs, y_test, 'proportional', 0.15, edge_threshold=0.03)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Calculate metrics
    result = calculate_metrics(bet_result, probs, y_test)
    
    print(f"   ✅ ROI: {result['metrics']['roi']:.1%}, Win Rate: {result['metrics']['winRate']:.1%}, Sharpe: {result['metrics']['sharpeRatio']:.2f}")
    
    return result


def main():
    """Generate backtest data for all models."""
    print("=" * 60)
    print("QuantBet Backtest Data Generator (Fixed)")
    print("=" * 60)
    
    # Load data
    df = load_nbl_data()
    feature_df = generate_features(df)
    
    # Run backtests for each model type
    models = ['kelly', 'elo', 'poisson', 'arbitrage']
    results = {}
    
    for model_id in models:
        results[model_id] = run_backtest(df.copy(), feature_df.copy(), model_id)
    
    # Save to JSON
    output_path = Path("dashboard/public/data/backtest_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also copy to dist
    dist_path = Path("dashboard/dist/data/backtest_results.json")
    if dist_path.parent.exists():
        with open(dist_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    print("=" * 60)
    
    # Print comparison table
    print("\n📊 Strategy Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<12} {'ROI':>8} {'Sharpe':>8} {'Win Rate':>10} {'Max DD':>8} {'Bets':>6}")
    print("-" * 80)
    for model_id, data in results.items():
        m = data['metrics']
        print(f"{model_id.title():<12} {m['roi']*100:>7.1f}% {m['sharpeRatio']:>8.2f} {m['winRate']*100:>9.1f}% {m['maxDrawdownPct']*100:>7.1f}% {m['totalBets']:>6}")
    print("-" * 80)
    
    return results


if __name__ == "__main__":
    main()

"""
Hyperparameter Tuning Test Script for QuantBet.

Loads NBL data from Excel, generates features, and runs Optuna tuning.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.tuning import HyperparameterTuner, quick_tune
from src.model.predictor import NBLPredictor, XGBConfig


def load_nbl_excel(xlsx_path: str = "data/nbl.xlsx") -> pd.DataFrame:
    """Load and prepare NBL data from Excel file."""
    print(f"Loading data from {xlsx_path}...")
    
    df = pd.read_excel(xlsx_path, sheet_name="Data", header=1)
    df.columns = df.columns.str.strip()
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'Date': 'game_date',
        'Home Team': 'home_team',
        'Away Team': 'away_team',
        'Home Score': 'home_score',
        'Away Score': 'away_score',
        'Home Odds Close': 'home_odds',
        'Away Odds Close': 'away_odds',
    })
    
    # Add season
    df['season'] = df['game_date'].apply(lambda d: 
        f"{d.year-1}-{str(d.year)[2:]}" if d.month < 7 
        else f"{d.year}-{str(d.year+1)[2:]}"
    )
    
    # Add target: home_win
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    # Filter to completed games only
    df = df.dropna(subset=['home_score', 'away_score'])
    
    print(f"Loaded {len(df)} games from {df['season'].min()} to {df['season'].max()}")
    return df


def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic features from game data for tuning test.
    
    Uses simplified features that can be computed directly from the Excel data.
    """
    print("\nGenerating features...")
    
    features = []
    
    for idx, game in df.iterrows():
        game_date = game['game_date']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Get historical games for this matchup
        historical = df[df['game_date'] < game_date]
        
        # Home team last 5 games
        home_games = historical[
            (historical['home_team'] == home_team) | 
            (historical['away_team'] == home_team)
        ].tail(5)
        
        # Away team last 5 games
        away_games = historical[
            (historical['home_team'] == away_team) | 
            (historical['away_team'] == away_team)
        ].tail(5)
        
        # Calculate features
        feat = {}
        
        # Home team rolling stats
        if len(home_games) >= 3:
            home_wins = 0
            home_pts_for = 0
            home_pts_against = 0
            for _, g in home_games.iterrows():
                if g['home_team'] == home_team:
                    home_pts_for += g['home_score']
                    home_pts_against += g['away_score']
                    if g['home_score'] > g['away_score']:
                        home_wins += 1
                else:
                    home_pts_for += g['away_score']
                    home_pts_against += g['home_score']
                    if g['away_score'] > g['home_score']:
                        home_wins += 1
            
            feat['home_win_pct_l5'] = home_wins / len(home_games)
            feat['home_pts_for_l5'] = home_pts_for / len(home_games)
            feat['home_pts_against_l5'] = home_pts_against / len(home_games)
            feat['home_net_rtg_l5'] = feat['home_pts_for_l5'] - feat['home_pts_against_l5']
        else:
            feat['home_win_pct_l5'] = 0.5
            feat['home_pts_for_l5'] = 85
            feat['home_pts_against_l5'] = 85
            feat['home_net_rtg_l5'] = 0
        
        # Away team rolling stats
        if len(away_games) >= 3:
            away_wins = 0
            away_pts_for = 0
            away_pts_against = 0
            for _, g in away_games.iterrows():
                if g['home_team'] == away_team:
                    away_pts_for += g['home_score']
                    away_pts_against += g['away_score']
                    if g['home_score'] > g['away_score']:
                        away_wins += 1
                else:
                    away_pts_for += g['away_score']
                    away_pts_against += g['home_score']
                    if g['away_score'] > g['home_score']:
                        away_wins += 1
            
            feat['away_win_pct_l5'] = away_wins / len(away_games)
            feat['away_pts_for_l5'] = away_pts_for / len(away_games)
            feat['away_pts_against_l5'] = away_pts_against / len(away_games)
            feat['away_net_rtg_l5'] = feat['away_pts_for_l5'] - feat['away_pts_against_l5']
        else:
            feat['away_win_pct_l5'] = 0.5
            feat['away_pts_for_l5'] = 85
            feat['away_pts_against_l5'] = 85
            feat['away_net_rtg_l5'] = 0
        
        # Differential features
        feat['win_pct_diff'] = feat['home_win_pct_l5'] - feat['away_win_pct_l5']
        feat['net_rtg_diff'] = feat['home_net_rtg_l5'] - feat['away_net_rtg_l5']
        
        # Odds-based features (if available)
        if pd.notna(game.get('home_odds')) and game['home_odds'] > 0:
            feat['home_implied_prob'] = 1 / game['home_odds']
            feat['away_implied_prob'] = 1 / game['away_odds'] if game['away_odds'] > 0 else 0.5
        else:
            feat['home_implied_prob'] = 0.5
            feat['away_implied_prob'] = 0.5
        
        # Rest days (simplified - just based on days since team last played)
        home_last = historical[
            (historical['home_team'] == home_team) | 
            (historical['away_team'] == home_team)
        ]
        if not home_last.empty:
            days_rest_home = (game_date - home_last['game_date'].max()).days
            feat['home_rest_days'] = min(days_rest_home, 14)
        else:
            feat['home_rest_days'] = 7
            
        away_last = historical[
            (historical['home_team'] == away_team) | 
            (historical['away_team'] == away_team)
        ]
        if not away_last.empty:
            days_rest_away = (game_date - away_last['game_date'].max()).days
            feat['away_rest_days'] = min(days_rest_away, 14)
        else:
            feat['away_rest_days'] = 7
        
        feat['rest_diff'] = feat['home_rest_days'] - feat['away_rest_days']
        feat['home_b2b'] = 1 if feat['home_rest_days'] <= 1 else 0
        feat['away_b2b'] = 1 if feat['away_rest_days'] <= 1 else 0
        
        # Store target
        feat['home_win'] = game['home_win']
        feat['game_date'] = game_date
        
        features.append(feat)
    
    feature_df = pd.DataFrame(features)
    print(f"Generated {len(feature_df.columns) - 2} features for {len(feature_df)} games")
    
    return feature_df


def run_tuning_test(n_trials: int = 20, timeout: int = 300):
    """Run hyperparameter tuning test."""
    print("=" * 60)
    print("QuantBet Hyperparameter Tuning Test")
    print("=" * 60)
    
    # Load data
    df = load_nbl_excel()
    
    # Generate features
    feature_df = create_basic_features(df)
    
    # Split features and target
    feature_cols = [c for c in feature_df.columns if c not in ['home_win', 'game_date']]
    X = feature_df[feature_cols]
    y = feature_df['home_win']
    
    print(f"\nFeature columns: {feature_cols}")
    print(f"\nDataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Target distribution: {y.mean():.2%} home wins")
    
    # Skip first 100 games (not enough history)
    X = X.iloc[100:]
    y = y.iloc[100:]
    print(f"After filtering: {len(X)} samples with sufficient history")
    
    # Run tuning
    print(f"\n{'=' * 60}")
    print(f"Starting Optuna hyperparameter tuning...")
    print(f"  Trials: {n_trials}")
    print(f"  Timeout: {timeout}s")
    print(f"  CV Splits: 3")
    print(f"{'=' * 60}\n")
    
    tuner = HyperparameterTuner(n_cv_splits=3, calibrate=True)
    result = tuner.tune(
        X, y, 
        n_trials=n_trials, 
        timeout=timeout,
        show_progress=True
    )
    
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    print(result)
    
    # Get parameter importances
    try:
        importances = tuner.get_param_importances()
        print("\nParameter Importances:")
        for param, imp in sorted(importances.items(), key=lambda x: -x[1])[:5]:
            print(f"  {param}: {imp:.3f}")
    except Exception as e:
        print(f"\nCould not compute importances: {e}")
    
    # Compare with default config
    print("\n" + "=" * 60)
    print("COMPARISON: Tuned vs Default")
    print("=" * 60)
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, brier_score_loss
    import xgboost as xgb
    
    # Fill NaN values
    X_clean = X.fillna(X.median())
    
    # Default XGBoost
    default_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    
    # Tuned XGBoost
    tuned_params = {
        **result.best_params,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic'
    }
    tuned_model = xgb.XGBClassifier(**tuned_params)
    
    # Brier score (lower is better)
    brier_scorer = make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
    
    print("\nCross-validation (5-fold):")
    
    default_scores = cross_val_score(default_model, X_clean, y, cv=5, scoring=brier_scorer)
    print(f"  Default Brier: {-default_scores.mean():.4f} ± {default_scores.std():.4f}")
    
    tuned_scores = cross_val_score(tuned_model, X_clean, y, cv=5, scoring=brier_scorer)
    print(f"  Tuned Brier:   {-tuned_scores.mean():.4f} ± {tuned_scores.std():.4f}")
    
    improvement = (-default_scores.mean()) - (-tuned_scores.mean())
    pct_improvement = (improvement / -default_scores.mean()) * 100
    
    if improvement > 0:
        print(f"\n  ✅ Improvement: {improvement:.4f} ({pct_improvement:.1f}% better)")
    else:
        print(f"\n  ⚠️ No improvement over default config")
    
    # Save tuned config
    config = result.to_xgb_config()
    print(f"\n{'=' * 60}")
    print("Best XGBoost Config:")
    print(f"{'=' * 60}")
    for key, val in result.best_params.items():
        print(f"  {key}: {val}")
    
    return result


if __name__ == "__main__":
    # Run with 20 trials (quick test) 
    # Increase to 100+ for better results
    result = run_tuning_test(n_trials=20, timeout=300)

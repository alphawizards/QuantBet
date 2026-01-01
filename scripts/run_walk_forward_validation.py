"""
Run Walk-Forward Validation on QuantBet Models.

Tests models using temporal validation to prevent overfitting.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.walk_forward import WalkForwardValidator
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_historical_data():
    """Load historical NBL game data."""
    # This would load from your database or CSV
    # For now, creating sample data structure
    logger.info("Loading historical game data...")
    
    data_path = Path("data/games.csv")
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} games from {data_path}")
        return df
    else:
        logger.warning("No historical data found. Creating sample data...")
        # Create sample data for demonstration
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='2D')
        n_games = len(dates)
        
        df = pd.DataFrame({
            'game_date': dates,
            'home_team': np.random.choice(['MEL', 'SYD', 'PER', 'BRI'], n_games),
            'away_team': np.random.choice(['ADL', 'NZB', 'PHX', 'SEM'], n_games),
            'home_score': np.random.randint(70, 110, n_games),
            'away_score': np.random.randint(70, 110, n_games),
            'home_win': np.random.randint(0, 2, n_games),
            # Features
            'delta_efg': np.random.randn(n_games) * 0.05,
            'delta_tov': np.random.randn(n_games) * 3,
            'delta_orb': np.random.randn(n_games) * 0.05,
            'delta_ftr': np.random.randn(n_games) * 0.05,
            'home_elo': 1500 + np.random.randn(n_games) * 100,
            'away_elo': 1500 + np.random.randn(n_games) * 100,
        })
        
        logger.info(f"Created {len(df)} sample games")
        return df


def create_simple_model():
    """Factory function to create a simple model."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )


def create_logistic_model():
    """Factory function to create logistic regression model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])


def run_validation():
    """Run walk-forward validation on models."""
    logger.info("=" * 60)
    logger.info("QUANTBET WALK-FORWARD VALIDATION")
    logger.info("=" * 60)
    
    # Load data
    data = load_historical_data()
    
    # Ensure date column is datetime
    data['game_date'] = pd.to_datetime(data['game_date'])
    
    # Create validator
    validator = WalkForwardValidator(
        train_window_days=180,  # 6 months training
        test_window_days=30,     # 1 month testing
        step_days=30,            # Step forward 1 month
        min_train_samples=50
    )
    
    logger.info("\nConfiguration:")
    logger.info(f"  Train window: 180 days")
    logger.info(f"  Test window: 30 days")
    logger.info(f"  Step size: 30 days")
    
    # Test Model 1: Random Forest
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 1: Random Forest Classifier")
    logger.info("=" * 60)
    
    try:
        rf_results = validator.validate(
            model_factory=create_simple_model,
            data=data,
            target_column='home_win',
            date_column='game_date'
        )
        
        logger.info("\nRandom Forest Results:")
        logger.info(f"  Windows: {rf_results['n_windows']}")
        logger.info(f"  Mean Brier Score: {rf_results['mean_brier']:.4f} ± {rf_results['std_brier']:.4f}")
        logger.info(f"  Mean Accuracy: {rf_results['mean_accuracy']:.2%} ± {rf_results['std_accuracy']:.2%}")
        
        print("\n" + rf_results['summary'])
        
    except Exception as e:
        logger.error(f"Random Forest validation failed: {e}")
    
    # Test Model 2: Logistic Regression
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 2: Logistic Regression")
    logger.info("=" * 60)
    
    try:
        lr_results = validator.validate(
            model_factory=create_logistic_model,
            data=data,
            target_column='home_win',
            date_column='game_date'
        )
        
        logger.info("\nLogistic Regression Results:")
        logger.info(f"  Windows: {lr_results['n_windows']}")
        logger.info(f"  Mean Brier Score: {lr_results['mean_brier']:.4f} ± {lr_results['std_brier']:.4f}")
        logger.info(f"  Mean Accuracy: {lr_results['mean_accuracy']:.2%} ± {lr_results['std_accuracy']:.2%}")
        
        print("\n" + lr_results['summary'])
        
    except Exception as e:
        logger.error(f"Logistic Regression validation failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)
    
    # Save results
    results_path = Path("data/validation/walk_forward_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nResults would be saved to: {results_path}")
    logger.info("\nRecommendations:")
    logger.info("  1. Review Brier score stability across windows")
    logger.info("  2. If std > 0.05, model may be unstable")
    logger.info("  3. Compare multiple models to find best performer")
    logger.info("  4. Retrain on full dataset if validation passes")


if __name__ == "__main__":
    run_validation()

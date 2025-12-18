import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, Dict

class NBLModel:
    """
    XGBoost Regression Model for NBL Margin of Victory.
    """

    def __init__(self, params: Dict = None):
        """
        Args:
            params: XGBoost hyperparameters.
        """
        self.params = params if params else {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8
        }
        self.model = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into Walk-Forward Validation sets.
        Train: 2021-2023 seasons.
        Test: 2024 season.
        """
        # Assuming df has a 'season' or 'date' column
        # Converting date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        train_mask = (df['date'] >= '2021-01-01') & (df['date'] < '2024-01-01')
        test_mask = (df['date'] >= '2024-01-01')

        train_df = df[train_mask]
        test_df = df[test_mask]

        # Define Feature columns (excluding metadata and target)
        # Placeholder feature names based on our engineering
        feature_cols = [
            'home_rolling_off_rtg', 'home_rolling_def_rtg',
            'away_rolling_off_rtg', 'away_rolling_def_rtg',
            'home_travel_fatigue', 'away_travel_fatigue',
            'home_import_availability', 'away_import_availability'
        ]

        # Target: Margin of Victory (Home Score - Away Score)
        target_col = 'margin'

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        return X_train, y_train, X_test, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the XGBoost regressor.
        """
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train)

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts point margin.
        Positive = Home Win, Negative = Away Win.
        """
        if not self.model:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)

    def predict_spread_probability(self, predicted_margin: float, spread: float, std_dev: float = 11.5) -> float:
        """
        Converts predicted margin to probability of covering the spread.

        Args:
            predicted_margin: The model's predicted margin (Home - Away).
            spread: The bookmaker's spread (e.g., -5.5 for Home favorite).
                    If spread is -5.5, Home needs to win by 6+.
                    We compare (Predicted - Spread).
            std_dev: Standard deviation of NBL margins.
                     NBL standard deviation is typically around 10-13 points.

        Returns:
            Probability (0.0 - 1.0) that Home covers the spread.
        """
        # Z-score calculation
        # We want P(Actual Margin > -Spread)
        # Note: Spreads are usually denoted as "-5.5" for favorite.
        # So covering means Margin > 5.5.
        # Equivalent to Margin + Spread > 0 if Spread is negative?
        # Let's standardize: Spread is usually "Line".
        # If Home Line is -5.5, Home must win by > 5.5.
        # Probability = P(N(pred, std) > -spread)
        # Wait, if line is -5.5, we need Margin > 5.5.
        # Z = (Threshold - Mean) / StdDev
        # We want P(X > -Spread) where X ~ N(PredictedMargin, StdDev)
        # Z = (-Spread - PredictedMargin) / StdDev
        # Prob = 1 - CDF(Z)

        # Example: Pred = 10, Spread = -5.5. Home must win by 5.5.
        # We want P(X > 5.5).
        # Z = (5.5 - 10) / 11.5 = -4.5 / 11.5 = -0.39
        # P(Z > -0.39) = 1 - CDF(-0.39) = CDF(0.39) > 0.5. Makes sense.

        threshold = -spread
        z_score = (threshold - predicted_margin) / std_dev
        prob_cover = 1 - norm.cdf(z_score)

        return prob_cover

# Usage Example:
if __name__ == "__main__":
    # Mock Data Generation
    dates = pd.date_range(start='2021-01-01', end='2024-04-01', freq='W')
    data = []
    for d in dates:
        data.append({
            'date': d,
            'home_rolling_off_rtg': 110 + np.random.normal(0, 5),
            'home_rolling_def_rtg': 108 + np.random.normal(0, 5),
            'away_rolling_off_rtg': 105 + np.random.normal(0, 5),
            'away_rolling_def_rtg': 112 + np.random.normal(0, 5),
            'home_travel_fatigue': np.random.randint(0, 10),
            'away_travel_fatigue': np.random.randint(0, 10),
            'home_import_availability': np.random.choice([0.5, 1.0]),
            'away_import_availability': np.random.choice([0.5, 1.0]),
            'margin': np.random.normal(0, 12)
        })

    df = pd.DataFrame(data)

    model_sys = NBLModel()
    X_train, y_train, X_test, y_test = model_sys.prepare_data(df)

    if not X_train.empty:
        model_sys.train(X_train, y_train)

        # Test Prediction
        if not X_test.empty:
            sample_row = X_test.iloc[0:1]
            pred_margin = model_sys.predict_margin(sample_row)[0]

            # Example Spread: Home -3.5
            spread = -3.5
            prob = model_sys.predict_spread_probability(pred_margin, spread)

            print(f"Predicted Margin: {pred_margin:.2f}")
            print(f"Spread: {spread}")
            print(f"Probability of Cover: {prob:.4f}")
    else:
        print("Not enough data to train.")

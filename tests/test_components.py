import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.features import NBLFeatureEngineer
from src.model import NBLModel
from src.staking import calculate_bet_size

class TestNBLSystem(unittest.TestCase):

    def setUp(self):
        # Mock Data for Features
        self.games_data = pd.DataFrame({
            'game_id': [1, 2, 3, 4, 5],
            'date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22', '2023-01-29']),
            'home_team_id': [1, 2, 1, 2, 1],
            'away_team_id': [2, 1, 2, 1, 2],
            'venue': ['PER', 'MEL', 'PER', 'MEL', 'PER'],
            'home_score': [90, 85, 92, 88, 95],
            'away_score': [85, 90, 80, 95, 85]
        })

        self.player_stats = pd.DataFrame({
            'game_id': [1, 1],
            'player_id': [101, 102],
            'team_id': [1, 1],
            'is_import': [True, False],
            'dnp': [False, False]
        })

        self.team_metadata = {1: {'name': 'Perth'}, 2: {'name': 'Melbourne'}}

        self.fe = NBLFeatureEngineer(self.games_data, self.player_stats, self.team_metadata)

    def test_rolling_efficiency(self):
        df = self.fe.calculate_rolling_efficiency(team_id=1, window=2)
        self.assertFalse(df.empty)
        self.assertIn('rolling_off_rtg', df.columns)
        # First entry should be NaN for rolling
        self.assertTrue(pd.isna(df.iloc[0]['rolling_off_rtg']))

    def test_travel_fatigue(self):
        # PER to NZL
        score = self.fe.calculate_travel_fatigue(
            team_id=1,
            current_game_date=pd.Timestamp('2023-02-05'),
            prev_game_venue='PER',
            current_venue='NZL'
        )
        # 10 * 1.5 = 15. Rest = 2023-02-05 - 2023-01-29 = 7 days.
        # Fatigue = 15 - (7-1) = 15 - 6 = 9
        self.assertEqual(score, 9)

        # PER to PER (Home stand)
        score_home = self.fe.calculate_travel_fatigue(
            team_id=1,
            current_game_date=pd.Timestamp('2023-02-05'),
            prev_game_venue='PER',
            current_venue='PER'
        )
        self.assertEqual(score_home, 0)

    def test_import_availability(self):
        score = self.fe.assess_import_availability(game_id=1, team_id=1)
        self.assertEqual(score, 1.0) # 1 active import out of 1 import

    def test_bet_sizing(self):
        # 0.55 prob, 1.91 odds. Edge = (0.91 * 0.55 - 0.45) / 0.91 = (0.5005 - 0.45) / 0.91 = 0.0505 / 0.91 = 0.055
        # 1/4 Kelly = 0.0138
        # Bankroll 10000 -> ~138
        bet = calculate_bet_size(10000, 0.55, 1.91, kelly_fraction=0.25)
        self.assertGreater(bet, 130)
        self.assertLess(bet, 150)

        # Negative EV
        bet_loss = calculate_bet_size(10000, 0.40, 1.91)
        self.assertEqual(bet_loss, 0.0)

    def test_model_pipeline(self):
        model = NBLModel()
        df = pd.DataFrame({
            'date': pd.to_datetime(['2022-01-01', '2024-02-01']),
            'home_rolling_off_rtg': [110, 110],
            'home_rolling_def_rtg': [100, 100],
            'away_rolling_off_rtg': [100, 100],
            'away_rolling_def_rtg': [110, 110],
            'home_travel_fatigue': [0, 0],
            'away_travel_fatigue': [5, 5],
            'home_import_availability': [1, 1],
            'away_import_availability': [1, 1],
            'margin': [10, 10]
        })
        X_train, y_train, X_test, y_test = model.prepare_data(df)

        self.assertEqual(len(X_train), 1)
        self.assertEqual(len(X_test), 1)

        model.train(X_train, y_train)
        preds = model.predict_margin(X_test)
        self.assertEqual(len(preds), 1)

        prob = model.predict_spread_probability(10.0, -5.5)
        self.assertGreater(prob, 0.5)

if __name__ == '__main__':
    unittest.main()

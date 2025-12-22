"""
Tests for momentum feature calculations.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.momentum import (
    calculate_rsi,
    calculate_ma_crossover,
    calculate_volatility,
    calculate_momentum_score,
    calculate_momentum_features,
    get_team_margins
)


class TestRSI:
    """Tests for RSI calculation."""
    
    def test_all_wins_returns_high_rsi(self):
        """All positive margins should give high RSI."""
        margins = np.array([10, 5, 15, 8, 12])
        rsi = calculate_rsi(margins)
        assert rsi > 80
    
    def test_all_losses_returns_low_rsi(self):
        """All negative margins should give low RSI."""
        margins = np.array([-10, -5, -15, -8, -12])
        rsi = calculate_rsi(margins)
        assert rsi < 20
    
    def test_mixed_results_returns_mid_rsi(self):
        """Mixed margins should give RSI near 50."""
        margins = np.array([10, -10, 5, -5, 3])
        rsi = calculate_rsi(margins)
        assert 30 < rsi < 70
    
    def test_insufficient_data_returns_neutral(self):
        """Single game should return neutral RSI."""
        margins = np.array([5])
        rsi = calculate_rsi(margins)
        assert rsi == 50.0


class TestMACrossover:
    """Tests for moving average crossover."""
    
    def test_uptrend_returns_positive(self):
        """Improving recent form should give positive crossover."""
        # Last few games are wins
        margins = np.array([-10, -5, 0, 5, 10, 15, 20])
        crossover = calculate_ma_crossover(margins, short_period=3, long_period=7)
        assert crossover > 0
    
    def test_downtrend_returns_negative(self):
        """Declining recent form should give negative crossover."""
        margins = np.array([20, 15, 10, 5, 0, -5, -10])
        crossover = calculate_ma_crossover(margins, short_period=3, long_period=7)
        assert crossover < 0
    
    def test_insufficient_data_returns_zero(self):
        """Insufficient data should return zero."""
        margins = np.array([5, 10])
        crossover = calculate_ma_crossover(margins, short_period=3, long_period=7)
        assert crossover == 0.0


class TestMomentumScore:
    """Tests for composite momentum score."""
    
    def test_hot_streak_returns_positive(self):
        """Team on hot streak should have positive momentum."""
        margins = np.array([5, 10, 8, 12, 15])
        score = calculate_momentum_score(margins)
        assert score > 20
    
    def test_cold_streak_returns_negative(self):
        """Team on cold streak should have negative momentum."""
        margins = np.array([-5, -10, -8, -12, -15])
        score = calculate_momentum_score(margins)
        assert score < -20
    
    def test_score_in_valid_range(self):
        """Score should always be between -100 and 100."""
        for _ in range(100):
            margins = np.random.randn(10) * 15
            score = calculate_momentum_score(margins)
            assert -100 <= score <= 100


class TestMomentumFeatures:
    """Tests for full momentum features calculation."""
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i*3) for i in range(10)]
        
        data = []
        # Melbourne wins first 5, loses last 5
        for i, date in enumerate(dates):
            data.append({
                'game_date': date,
                'home_team': 'MEL' if i % 2 == 0 else 'SYD',
                'away_team': 'SYD' if i % 2 == 0 else 'MEL',
                'home_score': 90 + (5 if i < 5 else -5),
                'away_score': 85
            })
        
        return pd.DataFrame(data)
    
    def test_features_have_expected_keys(self, sample_historical_data):
        """Features should have all expected keys."""
        features = calculate_momentum_features(
            'MEL',
            pd.Timestamp('2024-02-01'),
            sample_historical_data
        )
        
        expected_keys = [
            'momentum_rsi',
            'momentum_trend',
            'momentum_volatility',
            'momentum_score',
            'momentum_last3_wpct'
        ]
        
        for key in expected_keys:
            assert key in features
    
    def test_features_have_valid_values(self, sample_historical_data):
        """Features should have valid numeric values."""
        features = calculate_momentum_features(
            'MEL',
            pd.Timestamp('2024-02-01'),
            sample_historical_data
        )
        
        assert 0 <= features['momentum_rsi'] <= 100
        assert -100 <= features['momentum_score'] <= 100
        assert features['momentum_volatility'] >= 0
        assert 0 <= features['momentum_last3_wpct'] <= 1


class TestGetTeamMargins:
    """Tests for extracting team margins."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample game data."""
        return pd.DataFrame([
            {'game_date': pd.Timestamp('2024-01-01'), 'home_team': 'MEL', 
             'away_team': 'SYD', 'home_score': 95, 'away_score': 85},
            {'game_date': pd.Timestamp('2024-01-04'), 'home_team': 'PER', 
             'away_team': 'MEL', 'home_score': 88, 'away_score': 92},
            {'game_date': pd.Timestamp('2024-01-07'), 'home_team': 'MEL', 
             'away_team': 'BRI', 'home_score': 100, 'away_score': 90},
        ])
    
    def test_extracts_correct_margins(self, sample_data):
        """Should extract margins from both home and away games."""
        margins = get_team_margins(
            'MEL',
            pd.Timestamp('2024-01-10'),
            sample_data,
            n_games=10
        )
        
        # Should have 3 games in chronological order
        assert len(margins) == 3
        assert margins[0] == 10   # Jan 1: MEL beats SYD by 10
        assert margins[1] == 4    # Jan 4: MEL beats PER by 4 (away)
        assert margins[2] == 10   # Jan 7: MEL beats BRI by 10
    
    def test_respects_date_filter(self, sample_data):
        """Should only include games before the date."""
        margins = get_team_margins(
            'MEL',
            pd.Timestamp('2024-01-05'),
            sample_data,
            n_games=10
        )
        
        # Games on Jan 1 and Jan 4 should be included (both < Jan 5)
        assert len(margins) == 2
        assert margins[0] == 10  # Jan 1: MEL beats SYD by 10
        assert margins[1] == 4   # Jan 4: MEL beats PER by 4

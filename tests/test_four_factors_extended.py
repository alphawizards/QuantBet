"""
Extended tests for Four Factors components.

Tests for:
    - PaceCalculator: NBL pace normalization
    - RollingFourFactors: Rolling differential calculation
    - PaceMetrics: Pace analysis metrics
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.features.four_factors import (
    FourFactors,
    FourFactorsDifferential,
    FourFactorsCalculator,
    PaceCalculator,
    PaceMetrics,
    RollingFourFactors,
)


class TestPaceCalculator:
    """Tests for NBL pace calculations."""
    
    @pytest.fixture
    def calculator(self):
        return PaceCalculator()
    
    def test_estimate_possessions_formula(self, calculator):
        """Test standard possession formula."""
        # Poss = FGA - OREB + TO + 0.44*FTA
        poss = calculator.estimate_possessions(
            fga=80, oreb=10, turnovers=15, fta=20
        )
        expected = 80 - 10 + 15 + (0.44 * 20)
        assert poss == pytest.approx(expected)
    
    def test_estimate_possessions_zero_inputs(self, calculator):
        """Zero stats should give zero possessions."""
        poss = calculator.estimate_possessions(0, 0, 0, 0)
        assert poss == 0.0
    
    def test_calculate_game_pace_40_minutes(self, calculator):
        """Pace for 40-minute FIBA game."""
        # 75 possessions in 40 minutes = pace 75
        metrics = calculator.calculate_game_pace(
            fga=80, oreb=10, turnovers=15, fta=20
        )
        assert isinstance(metrics, PaceMetrics)
        assert metrics.pace > 0
        assert metrics.pace_48 > metrics.pace  # 48-min > 40-min
    
    def test_pace_fiba_to_nba_conversion(self, calculator):
        """Pace_48 should be 1.2x pace_40."""
        metrics = calculator.calculate_game_pace(
            fga=80, oreb=10, turnovers=15, fta=20
        )
        ratio = metrics.pace_48 / metrics.pace
        assert ratio == pytest.approx(48 / 40)
    
    def test_calculate_game_average_pace(self, calculator):
        """Game pace is average of both teams."""
        metrics = calculator.calculate_game_average_pace(
            home_fga=80, home_oreb=10, home_to=15, home_fta=20,
            away_fga=78, away_oreb=12, away_to=14, away_fta=22
        )
        assert isinstance(metrics, PaceMetrics)
        assert 65 < metrics.pace < 100  # Reasonable NBL pace range
    
    def test_nbl_pace_impact_analysis(self, calculator):
        """Pace impact analysis returns expected keys."""
        result = calculator.calculate_nbl_pace_impact(
            team_pace=78.0,
            opponent_pace=72.0
        )
        assert 'pace_differential' in result
        assert 'expected_game_pace' in result
        assert 'is_pace_advantage' in result
        assert 'pace_mismatch' in result
        assert result['pace_differential'] == 6.0
        assert result['is_pace_advantage'] is True
        assert result['pace_mismatch'] is True  # >3.0 difference


class TestPaceMetrics:
    """Tests for PaceMetrics dataclass."""
    
    def test_pace_relative_to_league(self):
        """Test league-relative pace calculation."""
        metrics = PaceMetrics(possessions=75, pace=75, pace_48=90)
        assert metrics.pace_relative_to_league == 1.0  # Exactly average
        
        fast_metrics = PaceMetrics(possessions=80, pace=80, pace_48=96)
        assert fast_metrics.pace_relative_to_league > 1.0
    
    def test_is_high_pace(self):
        """Test high pace detection."""
        # NBL_AVG_PACE = 75, high threshold = 78.75
        slow = PaceMetrics(possessions=72, pace=72, pace_48=86.4)
        assert slow.is_high_pace is False
        
        fast = PaceMetrics(possessions=80, pace=80, pace_48=96)
        assert fast.is_high_pace is True


class TestRollingFourFactors:
    """Tests for rolling Four Factors calculations."""
    
    @pytest.fixture
    def sample_games(self):
        """Create sample game data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=20, freq='3D')
        
        games = []
        for i, date in enumerate(dates):
            games.append({
                'game_date': date,
                'season': '2023-2024',
                'home_team': 'MEL' if i % 2 == 0 else 'SYD',
                'away_team': 'SYD' if i % 2 == 0 else 'MEL',
                'home_score': 90 + np.random.randint(-10, 10),
                'away_score': 88 + np.random.randint(-10, 10),
                'home_fgm': 32 + np.random.randint(-5, 5),
                'home_fga': 75 + np.random.randint(-5, 5),
                'home_fg3m': 10 + np.random.randint(-3, 3),
                'home_ftm': 15 + np.random.randint(-5, 5),
                'home_fta': 20 + np.random.randint(-5, 5),
                'home_turnovers': 12 + np.random.randint(-3, 3),
                'home_orb': 10 + np.random.randint(-3, 3),
                'home_drb': 28 + np.random.randint(-3, 3),
                'away_fgm': 31 + np.random.randint(-5, 5),
                'away_fga': 73 + np.random.randint(-5, 5),
                'away_fg3m': 9 + np.random.randint(-3, 3),
                'away_ftm': 14 + np.random.randint(-5, 5),
                'away_fta': 18 + np.random.randint(-5, 5),
                'away_turnovers': 13 + np.random.randint(-3, 3),
                'away_orb': 9 + np.random.randint(-3, 3),
                'away_drb': 27 + np.random.randint(-3, 3),
            })
        
        return pd.DataFrame(games)
    
    def test_initialization(self):
        """Test RollingFourFactors initialization."""
        rolling = RollingFourFactors(window=5)
        assert rolling.window == 5
        assert rolling.calculator is not None
    
    def test_calculate_team_rolling_factors(self, sample_games):
        """Test rolling factors calculation for a team."""
        rolling = RollingFourFactors(window=3)
        
        # Calculate factors for MEL using games before game 15
        factors = rolling.calculate_team_rolling_factors(
            sample_games,
            team_code='MEL',
            as_of_date=pd.Timestamp('2024-02-15')
        )
        
        assert factors is not None
        assert isinstance(factors, FourFactors)
        assert 0.3 < factors.efg_pct < 0.7  # Reasonable range
        assert 5 < factors.tov_pct < 25  # Reasonable range
    
    def test_calculate_team_rolling_factors_insufficient_data(self, sample_games):
        """Return None when insufficient game history."""
        rolling = RollingFourFactors(window=10)
        
        # Early date - not enough games
        factors = rolling.calculate_team_rolling_factors(
            sample_games,
            team_code='MEL',
            as_of_date=pd.Timestamp('2024-01-05')
        )
        
        assert factors is None
    
    def test_calculate_rolling_differentials(self, sample_games):
        """Test differential calculation between teams."""
        rolling = RollingFourFactors(window=3)
        
        diffs = rolling.calculate_rolling_differentials(
            sample_games,
            home_team='MEL',
            away_team='SYD',
            game_date=pd.Timestamp('2024-02-15')
        )
        
        assert diffs is not None
        assert 'delta_efg' in diffs
        assert 'delta_tov' in diffs
        assert 'delta_orb' in diffs
        assert 'delta_ftr' in diffs
        assert 'weighted_score' in diffs
    
    def test_add_rolling_features_to_df(self, sample_games):
        """Test adding rolling features to DataFrame."""
        rolling = RollingFourFactors(window=3)
        
        result = rolling.add_rolling_features_to_df(sample_games)
        
        assert 'roll_delta_efg' in result.columns
        assert 'roll_delta_tov' in result.columns
        assert 'roll_weighted_score' in result.columns
        
        # First N games should be NaN
        assert result['roll_delta_efg'].iloc[:3].isna().all()
        
        # Some later games should have values
        assert not result['roll_delta_efg'].iloc[10:].isna().all()


class TestFourFactorsDifferential:
    """Tests for Four Factors differential calculations."""
    
    def test_weighted_score_calculation(self):
        """Test Oliver's weighted score formula."""
        diff = FourFactorsDifferential(
            delta_efg=0.05,  # 5% better shooting
            delta_tov=-2.0,  # 2% fewer turnovers (good)
            delta_orb=0.03,  # 3% more offensive rebounds
            delta_ftr=0.02   # 2% better FT rate
        )
        
        # Weights: eFG 40%, TOV 25%, ORB 20%, FT 15%
        # TOV is inverted (negative is good, so -2 * -0.25 = +0.5)
        score = diff.weighted_score
        
        # Should be positive (favoring the team)
        assert score > 0
    
    def test_weighted_score_negative(self):
        """Test negative weighted score for worse team."""
        diff = FourFactorsDifferential(
            delta_efg=-0.08,  # 8% worse shooting
            delta_tov=3.0,    # 3% more turnovers (bad)
            delta_orb=-0.02,  # 2% fewer offensive rebounds
            delta_ftr=-0.01   # 1% worse FT rate
        )
        
        score = diff.weighted_score
        assert score < 0


class TestFourFactorsCalculator:
    """Additional tests for FourFactorsCalculator."""
    
    @pytest.fixture
    def calculator(self):
        return FourFactorsCalculator()
    
    def test_calculate_differential(self, calculator):
        """Test differential calculation."""
        home = FourFactors(efg_pct=0.55, tov_pct=12.0, orb_pct=0.28, ft_rate=0.22)
        away = FourFactors(efg_pct=0.50, tov_pct=14.0, orb_pct=0.25, ft_rate=0.20)
        
        diff = calculator.calculate_differential(home, away)
        
        assert isinstance(diff, FourFactorsDifferential)
        assert diff.delta_efg == pytest.approx(0.05)
        assert diff.delta_tov == pytest.approx(-2.0)
        assert diff.delta_orb == pytest.approx(0.03)
        assert diff.delta_ftr == pytest.approx(0.02)

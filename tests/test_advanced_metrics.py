"""
Unit tests for Advanced Metrics Calculator.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.advanced_metrics import (
    AdvancedMetricsCalculator,
    PlayerBPM,
    TeamAdvancedMetrics,
)


class TestAdvancedMetricsCalculator:
    """Tests for BPM and advanced metrics calculations."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return AdvancedMetricsCalculator()
    
    @pytest.fixture
    def sample_player_stats(self):
        """Sample player stats for testing."""
        return pd.Series({
            'pts': 20.0,
            'trb': 5.0,
            'ast': 4.0,
            'stl': 1.5,
            'blk': 0.5,
            'tov': 2.0,
            'pf': 2.5,
            'fgm': 7.0,
            'fga': 15.0,
            'fta': 6.0,
            'fg3m': 2.0,
            'mp': 32.0,
        })
    
    @pytest.fixture
    def sample_team_results(self):
        """Sample team results DataFrame."""
        return pd.DataFrame({
            'home_team': ['MEL', 'SYD', 'PER', 'MEL', 'SYD'],
            'away_team': ['SYD', 'PER', 'MEL', 'PER', 'MEL'],
            'home_score': [95, 88, 102, 91, 99],
            'away_score': [91, 92, 99, 88, 95],
        })
    
    def test_initialization_defaults(self, calculator):
        """Calculator should initialize with default values."""
        assert calculator.min_minutes == 10.0
        assert 'pts_per_100' in calculator.league_avg
    
    def test_bpm_returns_float(self, calculator, sample_player_stats):
        """BPM calculation should return a float."""
        bpm = calculator.calculate_player_bpm(sample_player_stats)
        assert isinstance(bpm, float)
    
    def test_bpm_reasonable_range(self, calculator, sample_player_stats):
        """BPM should be within reasonable bounds (-15 to +15)."""
        bpm = calculator.calculate_player_bpm(sample_player_stats)
        assert -15.0 <= bpm <= 15.0
    
    def test_bpm_low_minutes_returns_zero(self, calculator):
        """Players with low minutes should return 0 BPM."""
        low_min_stats = pd.Series({
            'pts': 5.0, 'trb': 2.0, 'ast': 1.0,
            'stl': 0.0, 'blk': 0.0, 'tov': 1.0,
            'pf': 1.0, 'fgm': 2.0, 'fga': 4.0,
            'fta': 2.0, 'fg3m': 0.0, 'mp': 5.0,  # Below threshold
        })
        bpm = calculator.calculate_player_bpm(low_min_stats)
        assert bpm == 0.0
    
    def test_bpm_star_player_positive(self, calculator):
        """Star player stats should produce positive BPM."""
        star_stats = pd.Series({
            'pts': 28.0, 'trb': 8.0, 'ast': 7.0,
            'stl': 1.5, 'blk': 0.8, 'tov': 2.5,
            'pf': 2.0, 'fgm': 10.0, 'fga': 18.0,
            'fta': 8.0, 'fg3m': 3.0, 'mp': 36.0,
        })
        bpm = calculator.calculate_player_bpm(star_stats)
        assert bpm > 0.0
    
    def test_bpm_inefficient_player_lower(self, calculator):
        """Inefficient player should have lower BPM than efficient one."""
        efficient_stats = pd.Series({
            'pts': 15.0, 'trb': 4.0, 'ast': 3.0,
            'stl': 1.0, 'blk': 0.5, 'tov': 1.0,
            'pf': 2.0, 'fgm': 6.0, 'fga': 10.0,  # 60% FG
            'fta': 3.0, 'fg3m': 1.0, 'mp': 28.0,
        })
        
        inefficient_stats = pd.Series({
            'pts': 15.0, 'trb': 4.0, 'ast': 3.0,
            'stl': 1.0, 'blk': 0.5, 'tov': 3.0,  # More turnovers
            'pf': 2.0, 'fgm': 6.0, 'fga': 18.0,  # 33% FG
            'fta': 3.0, 'fg3m': 1.0, 'mp': 28.0,
        })
        
        efficient_bpm = calculator.calculate_player_bpm(efficient_stats)
        inefficient_bpm = calculator.calculate_player_bpm(inefficient_stats)
        
        assert efficient_bpm > inefficient_bpm
    
    def test_bpm_differential(self, calculator):
        """BPM differential should be home minus away."""
        home_bpm = 3.5
        away_bpm = -1.2
        
        diff = calculator.calculate_bpm_differential(home_bpm, away_bpm)
        
        assert diff == pytest.approx(4.7, abs=0.01)
    
    def test_sos_returns_bounded_value(self, calculator, sample_team_results):
        """SOS should be between -1 and 1."""
        team_ratings = {'MEL': 0.2, 'SYD': -0.1, 'PER': 0.3}
        
        sos = calculator.calculate_sos(sample_team_results, 'MEL', team_ratings)
        
        assert -1.0 <= sos <= 1.0
    
    def test_sos_hard_schedule_positive(self, calculator):
        """Playing strong opponents should yield positive SOS."""
        results = pd.DataFrame({
            'home_team': ['WEAK', 'STRONG1', 'STRONG2'],
            'away_team': ['STRONG1', 'WEAK', 'WEAK'],
            'home_score': [80, 100, 98],
            'away_score': [95, 85, 82],
        })
        ratings = {'WEAK': -0.5, 'STRONG1': 0.6, 'STRONG2': 0.7}
        
        sos = calculator.calculate_sos(results, 'WEAK', ratings)
        
        assert sos > 0.0  # Played stronger teams
    
    def test_sos_adjusted_win_pct(self, calculator):
        """SOS adjustment should increase win % for hard schedule."""
        win_pct = 0.50
        hard_sos = 0.3
        easy_sos = -0.3
        
        hard_adj = calculator.calculate_sos_adjusted_win_pct(win_pct, hard_sos)
        easy_adj = calculator.calculate_sos_adjusted_win_pct(win_pct, easy_sos)
        
        assert hard_adj > win_pct
        assert easy_adj < win_pct
    
    def test_sos_adjusted_win_pct_bounded(self, calculator):
        """Adjusted win % should be clamped to [0, 1]."""
        result_high = calculator.calculate_sos_adjusted_win_pct(0.95, 0.5)
        result_low = calculator.calculate_sos_adjusted_win_pct(0.05, -0.5)
        
        assert result_high <= 1.0
        assert result_low >= 0.0
    
    def test_expected_wins_formula(self, calculator):
        """Expected wins should follow Pythagorean expectation."""
        # Equal points = 50% expected
        exp = calculator.calculate_expected_wins(1000, 1000, 10)
        assert exp == pytest.approx(5.0, abs=0.1)
        
        # Better offense = more expected wins
        exp_better = calculator.calculate_expected_wins(1100, 1000, 10)
        assert exp_better > 5.0
    
    def test_expected_wins_handles_zero(self, calculator):
        """Expected wins should handle edge cases."""
        assert calculator.calculate_expected_wins(100, 0, 10) == 0.0
        assert calculator.calculate_expected_wins(100, 100, 0) == 0.0
    
    def test_team_advanced_metrics_structure(self, calculator, sample_team_results):
        """Team metrics should return all expected fields."""
        metrics = calculator.calculate_team_advanced_metrics(
            team_code='MEL',
            team_results=sample_team_results,
        )
        
        assert isinstance(metrics, TeamAdvancedMetrics)
        assert metrics.team_code == 'MEL'
        assert hasattr(metrics, 'bpm_avg')
        assert hasattr(metrics, 'sos')
        assert hasattr(metrics, 'sos_adjusted_win_pct')
        assert hasattr(metrics, 'expected_wins')
    
    def test_team_metrics_as_dict(self, calculator, sample_team_results):
        """Team metrics should convert to dict for DataFrame use."""
        metrics = calculator.calculate_team_advanced_metrics(
            team_code='MEL',
            team_results=sample_team_results,
        )
        
        metrics_dict = metrics.as_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'bpm_avg' in metrics_dict
        assert 'sos' in metrics_dict


class TestTrueShootingCalculation:
    """Tests for True Shooting Percentage helper."""
    
    @pytest.fixture
    def calculator(self):
        return AdvancedMetricsCalculator()
    
    def test_ts_perfect_shooting(self, calculator):
        """Perfect shooting should yield high TS%."""
        # 20 pts on 10 FGA, 0 FTA = excellent efficiency
        ts = calculator._calculate_ts_pct(20, 10, 0)
        assert ts == 1.0
    
    def test_ts_zero_attempts(self, calculator):
        """Zero attempts should return 0."""
        ts = calculator._calculate_ts_pct(0, 0, 0)
        assert ts == 0.0
    
    def test_ts_realistic_range(self, calculator):
        """Realistic stats should produce TS% between 0.4 and 0.7."""
        # 18 pts on 12 FGA, 6 FTA (typical)
        ts = calculator._calculate_ts_pct(18, 12, 6)
        assert 0.4 <= ts <= 0.7


class TestPositionAdjustments:
    """Tests for position-based BPM adjustments."""
    
    @pytest.fixture
    def calculator(self):
        return AdvancedMetricsCalculator()
    
    @pytest.fixture
    def rebounding_stats(self):
        """Stats with many rebounds."""
        return pd.Series({
            'pts': 12.0, 'trb': 12.0, 'ast': 1.0,
            'stl': 0.5, 'blk': 2.0, 'tov': 1.0,
            'pf': 2.5, 'fgm': 5.0, 'fga': 10.0,
            'fta': 2.0, 'fg3m': 0.0, 'mp': 30.0,
        })
    
    def test_center_gets_rebound_credit(self, calculator, rebounding_stats):
        """Centers should get more credit for rebounds than guards."""
        center_bpm = calculator.calculate_player_bpm(rebounding_stats, position='C')
        guard_bpm = calculator.calculate_player_bpm(rebounding_stats, position='G')
        
        assert center_bpm > guard_bpm

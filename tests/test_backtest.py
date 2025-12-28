"""
Unit tests for Backtesting Framework.

Tests cover:
    - BacktestMetrics calculation
    - BetSimulator execution
    - StakingStrategy implementations
    - LeakageValidator detection
    - BacktestEngine walk-forward validation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import numpy as np
import pandas as pd

from src.backtest.metrics import (
    BacktestMetrics,
    calculate_metrics,
    calculate_equity_curve,
    calculate_drawdown_metrics,
    calculate_sharpe_ratio,
    calculate_brier_score,
    calculate_calibration_error
)
from src.backtest.simulator import (
    BetSimulator,
    BetResult,
    SessionResult,
    BetOutcome,
    FlatStake,
    PercentageStake,
    FractionalKellyStake,
    BoundedKellyStake
)
from src.backtest.validator import (
    LeakageValidator,
    LeakageWarning,
    LeakageSeverity,
    ValidationReport
)
from src.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult
)
from src.strategies.kelly import BetOpportunity


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_bet_results():
    """Sample bet results DataFrame."""
    return pd.DataFrame({
        'stake': [10, 10, 10, 10, 10],
        'odds': [2.0, 2.5, 1.8, 2.2, 1.9],
        'won': [True, False, True, False, True],
        'profit': [10, -10, 8, -10, 9]
    })


@pytest.fixture
def sample_predictions():
    """Sample probability predictions."""
    return np.array([0.55, 0.40, 0.60, 0.45, 0.52])


@pytest.fixture
def sample_actuals():
    """Sample actual outcomes."""
    return np.array([1, 0, 1, 0, 1])


@pytest.fixture
def sample_game_data():
    """Sample game data for backtesting."""
    dates = pd.date_range('2021-01-01', periods=200, freq='D')
    
    return pd.DataFrame({
        'game_date': dates,
        'season': ['2021-22'] * 50 + ['2022-23'] * 50 + ['2023-24'] * 50 + ['2024-25'] * 50,
        'home_team': ['MEL'] * 200,
        'away_team': ['SYD'] * 200,
        'home_win': np.random.randint(0, 2, 200),
        'home_odds': np.random.uniform(1.5, 3.0, 200),
        'feature_1': np.random.randn(200),
        'feature_2': np.random.randn(200),
        'feature_3': np.random.randn(200),
    })


# ============================================================================
# Metrics Tests
# ============================================================================

class TestBacktestMetrics:
    """Tests for BacktestMetrics."""
    
    def test_calculate_metrics_basic(self, sample_bet_results, sample_predictions, sample_actuals):
        """Test basic metrics calculation."""
        metrics = calculate_metrics(
            bet_results=sample_bet_results,
            predictions=sample_predictions,
            actuals=sample_actuals
        )
        
        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_bets == 5
        assert metrics.win_rate == 0.6  # 3 wins out of 5
        assert metrics.total_staked == 50
    
    def test_calculate_metrics_roi(self, sample_bet_results):
        """Test ROI calculation."""
        metrics = calculate_metrics(sample_bet_results)
        
        expected_profit = sum(sample_bet_results['profit'])
        expected_roi = expected_profit / 50  # Total staked
        
        assert metrics.roi == pytest.approx(expected_roi, rel=0.01)
    
    def test_calculate_metrics_empty(self):
        """Test with empty results."""
        empty_df = pd.DataFrame(columns=['stake', 'odds', 'won', 'profit'])
        metrics = calculate_metrics(empty_df)
        
        assert metrics.total_bets == 0
        assert metrics.roi == 0
    
    def test_equity_curve(self, sample_bet_results):
        """Test equity curve calculation."""
        curve = calculate_equity_curve(sample_bet_results, initial_bankroll=1000)
        
        assert len(curve) == 5
        assert curve.iloc[-1] == 1000 + sum(sample_bet_results['profit'])
    
    def test_drawdown_metrics(self):
        """Test drawdown calculation."""
        equity = pd.Series([1000, 1050, 1020, 1000, 980, 1010, 1050])
        max_dd, max_dd_pct, max_duration = calculate_drawdown_metrics(equity)
        
        assert max_dd > 0
        assert 0 <= max_dd_pct <= 1
        assert max_duration >= 0
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.05, -0.02, 0.03, 0.01, -0.01, 0.04])
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
    
    def test_brier_score(self):
        """Test Brier score calculation."""
        predictions = np.array([0.7, 0.3, 0.6, 0.4])
        actuals = np.array([1, 0, 1, 0])
        
        brier = calculate_brier_score(predictions, actuals)
        
        # Perfect = 0, random = 0.25
        assert 0 <= brier <= 0.5


# ============================================================================
# Simulator Tests
# ============================================================================

class TestBetSimulator:
    """Tests for BetSimulator."""
    
    def test_simulate_winning_bet(self):
        """Test winning bet simulation."""
        simulator = BetSimulator(slippage_rate=0, random_seed=42)
        
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0)
        result = simulator.simulate_bet(bet, stake=10, actual_outcome=True, bankroll=100)
        
        assert result.won
        assert result.profit > 0
        assert result.post_bankroll > result.pre_bankroll
    
    def test_simulate_losing_bet(self):
        """Test losing bet simulation."""
        simulator = BetSimulator(slippage_rate=0, random_seed=42)
        
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0)
        result = simulator.simulate_bet(bet, stake=10, actual_outcome=False, bankroll=100)
        
        assert not result.won
        assert result.profit == -10
        assert result.post_bankroll == 90
    
    def test_slippage_applied(self):
        """Test that slippage reduces effective odds."""
        simulator = BetSimulator(slippage_rate=0.05, random_seed=42)
        
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0)
        result = simulator.simulate_bet(bet, stake=10, actual_outcome=True, bankroll=100)
        
        assert result.odds <= 2.0
        assert result.slippage_applied >= 0
    
    def test_minimum_stake_rejection(self):
        """Test that bets below minimum are rejected."""
        simulator = BetSimulator(min_stake=5)
        
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0)
        result = simulator.simulate_bet(bet, stake=2, actual_outcome=True, bankroll=100)
        
        assert result.outcome == BetOutcome.VOID
        assert result.stake == 0
    
    def test_session_simulation(self):
        """Test full session simulation."""
        simulator = BetSimulator(random_seed=42)
        staking = FlatStake(stake_amount=10)
        
        bets = [
            (BetOpportunity(prob=0.55, decimal_odds=2.0), True),
            (BetOpportunity(prob=0.45, decimal_odds=2.2), False),
            (BetOpportunity(prob=0.60, decimal_odds=1.8), True),
        ]
        
        result = simulator.simulate_session(bets, staking, initial_bankroll=100)
        
        assert isinstance(result, SessionResult)
        assert result.initial_bankroll == 100
        assert result.num_bets > 0


# ============================================================================
# Staking Strategy Tests
# ============================================================================

class TestStakingStrategies:
    """Tests for staking strategies."""
    
    def test_flat_stake(self):
        """Test flat staking."""
        strategy = FlatStake(stake_amount=10)
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0)
        
        stake = strategy.calculate_stake(bet, bankroll=1000)
        
        assert stake == 10
    
    def test_flat_stake_negative_ev(self):
        """Test flat staking rejects negative EV."""
        strategy = FlatStake(stake_amount=10)
        bet = BetOpportunity(prob=0.40, decimal_odds=2.0)  # Negative EV
        
        stake = strategy.calculate_stake(bet, bankroll=1000)
        
        assert stake == 0
    
    def test_percentage_stake(self):
        """Test percentage staking."""
        strategy = PercentageStake(percentage=0.02)
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0)
        
        stake = strategy.calculate_stake(bet, bankroll=1000)
        
        assert stake == 20  # 2% of 1000
    
    def test_fractional_kelly(self):
        """Test fractional Kelly staking."""
        strategy = FractionalKellyStake(kelly_fraction=0.25)
        bet = BetOpportunity(prob=0.60, decimal_odds=2.0)
        
        stake = strategy.calculate_stake(bet, bankroll=1000)
        
        assert stake > 0
        assert stake < 1000 * 0.05  # Should respect max cap
    
    def test_bounded_kelly(self):
        """Test bounded Kelly with floor protection."""
        strategy = BoundedKellyStake(kelly_fraction=0.25, floor_fraction=0.50)
        bet = BetOpportunity(prob=0.60, decimal_odds=2.0)
        
        # When bankroll is near floor, stake should be reduced
        stake_high = strategy.calculate_stake(bet, bankroll=1000, peak_bankroll=1000)
        stake_low = strategy.calculate_stake(bet, bankroll=600, peak_bankroll=1000)
        
        assert stake_high > 0
        assert stake_low >= 0


# ============================================================================
# Validator Tests
# ============================================================================

class TestLeakageValidator:
    """Tests for LeakageValidator."""
    
    def test_bsp_detection(self):
        """Test BSP column detection."""
        validator = LeakageValidator()
        
        data = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'bsp_price': [2.0, 2.1, 2.2],  # Should trigger warning
            'target': [1, 0, 1]
        })
        
        warnings = validator.check_bsp_leakage(data)
        
        assert len(warnings) > 0
        assert any(w.severity == LeakageSeverity.CRITICAL for w in warnings)
    
    def test_future_pattern_detection(self):
        """Test future-looking column detection."""
        validator = LeakageValidator()
        
        data = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'next_game_result': [1, 0, 1],  # Should trigger warning
            'target': [1, 0, 1]
        })
        
        warnings = validator.check_future_patterns(data)
        
        assert len(warnings) > 0
    
    def test_target_correlation_detection(self):
        """Test high correlation with target detection."""
        validator = LeakageValidator()
        
        data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'leaky_feature': np.random.randint(0, 2, 100),  # Random
            'home_win': np.random.randint(0, 2, 100)
        })
        
        # Make leaky_feature almost perfectly correlated with target
        data['leaky_feature'] = data['home_win'] + np.random.randn(100) * 0.01
        
        warnings = validator.check_target_leakage(data, 'home_win')
        
        assert len(warnings) > 0
        assert any(w.severity == LeakageSeverity.CRITICAL for w in warnings)
    
    def test_full_validation(self):
        """Test complete validation pipeline."""
        validator = LeakageValidator()
        
        data = pd.DataFrame({
            'game_date': pd.date_range('2023-01-01', periods=10),
            'feature_1': np.random.randn(10),
            'feature_2': np.random.randn(10),
            'home_win': np.random.randint(0, 2, 10)
        })
        
        report = validator.validate(data, 'game_date', 'home_win')
        
        assert isinstance(report, ValidationReport)
        assert report.features_checked > 0
        assert report.rows_checked == 10


# ============================================================================
# Engine Tests
# ============================================================================

class TestBacktestEngine:
    """Tests for BacktestEngine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        config = BacktestConfig(
            train_window=2,
            test_window=1,
            initial_bankroll=1000
        )
        engine = BacktestEngine(config=config)
        
        assert engine.config.train_window == 2
        assert engine.config.initial_bankroll == 1000
    
    def test_auto_split_seasons(self, sample_game_data):
        """Test automatic season splitting."""
        engine = BacktestEngine()
        
        train, test = engine._auto_split_seasons(sample_game_data, 'season')
        
        assert len(train) > 0
        assert len(test) > 0
        assert train != test
    
    def test_create_bet_opportunities(self):
        """Test bet opportunity creation."""
        engine = BacktestEngine()
        
        predictions = pd.DataFrame({
            'home_team': ['MEL', 'SYD'],
            'away_team': ['SYD', 'MEL'],
            'predicted_prob': [0.60, 0.55],
            'odds': [2.0, 2.2],
            'actual_outcome': [1, 0]
        })
        
        bets = engine._create_bet_opportunities(predictions, min_edge=0.02)
        
        assert len(bets) > 0
        assert all(isinstance(b[0], BetOpportunity) for b in bets)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the backtesting framework."""
    
    def test_full_backtest_pipeline(self, sample_game_data):
        """Test complete backtest pipeline."""
        # Create a mock model
        model = Mock()
        model.fit = Mock(return_value=model)
        model.predict_proba = Mock(return_value=np.random.uniform(0.4, 0.6, 50))
        
        engine = BacktestEngine(
            config=BacktestConfig(
                train_window=1,
                test_window=1,
                initial_bankroll=1000,
                validate_leakage=False
            )
        )
        
        features = ['feature_1', 'feature_2', 'feature_3']
        
        result = engine.run_backtest(
            model=model,
            data=sample_game_data,
            features=features,
            target='home_win',
            train_seasons=['2022-23'],
            test_seasons=['2023-24']
        )
        
        assert isinstance(result, BacktestResult)
        assert result.metrics is not None
        assert result.session is not None
        assert len(result.train_periods) > 0
        assert len(result.test_periods) > 0
    
    def test_metrics_to_dataframe_conversion(self, sample_bet_results):
        """Test that session results can be converted to DataFrame."""
        simulator = BetSimulator(random_seed=42)
        staking = FlatStake(10)
        
        bets = [
            (BetOpportunity(prob=0.55, decimal_odds=2.0), True),
            (BetOpportunity(prob=0.55, decimal_odds=2.0), False),
        ]
        
        session = simulator.simulate_session(bets, staking, 1000)
        df = session.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'stake' in df.columns
        assert 'profit' in df.columns
        assert 'won' in df.columns

"""
Tests for RL Bankroll Manager.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.strategies.rl_bankroll import (
    BettingState,
    BettingAction,
    RLBankrollManager,
    AdaptiveKellyManager
)


class TestBettingState:
    """Tests for BettingState."""
    
    def test_to_array(self):
        """State should convert to array."""
        state = BettingState(
            bankroll=1000,
            bankroll_pct_of_peak=0.9,
            recent_win_rate=0.6,
            current_streak=3,
            model_confidence=0.7,
            edge=0.05,
            kelly_fraction=0.1
        )
        
        arr = state.to_array()
        assert len(arr) == 6
        assert all(isinstance(x, (int, float, np.floating)) for x in arr)
    
    def test_discretize(self):
        """State should discretize to tuple."""
        state = BettingState(
            bankroll=1000,
            bankroll_pct_of_peak=0.8,
            recent_win_rate=0.5,
            current_streak=0,
            model_confidence=0.5,
            edge=0.0,
            kelly_fraction=0.0
        )
        
        discrete = state.discretize(n_bins=5)
        assert isinstance(discrete, tuple)
        assert all(0 <= x < 5 for x in discrete)


class TestBettingAction:
    """Tests for BettingAction."""
    
    def test_get_multiplier(self):
        """Actions should have correct multipliers."""
        assert BettingAction.get_multiplier(BettingAction.SKIP) == 0.0
        assert BettingAction.get_multiplier(BettingAction.BET_SMALL) == 0.5
        assert BettingAction.get_multiplier(BettingAction.BET_NORMAL) == 1.0
        assert BettingAction.get_multiplier(BettingAction.BET_LARGE) == 1.5
    
    def test_n_actions(self):
        """Should have 4 actions."""
        assert BettingAction.n_actions() == 4


class TestRLBankrollManager:
    """Tests for RLBankrollManager."""
    
    @pytest.fixture
    def manager(self):
        """Create fresh manager."""
        return RLBankrollManager(initial_bankroll=1000)
    
    def test_initialization(self, manager):
        """Manager should initialize correctly."""
        assert manager.bankroll == 1000
        assert manager.peak_bankroll == 1000
        assert manager.training == True
    
    def test_get_current_state(self, manager):
        """Should build state from context."""
        state = manager.get_current_state(
            confidence=0.7,
            edge=0.05,
            kelly=0.1
        )
        
        assert isinstance(state, BettingState)
        assert state.model_confidence == 0.7
        assert state.edge == 0.05
        assert state.kelly_fraction == 0.1
    
    def test_select_action_skips_negative_edge(self, manager):
        """Should skip when edge is negative."""
        state = manager.get_current_state(
            confidence=0.7,
            edge=-0.05,  # Negative edge
            kelly=0.0
        )
        
        action = manager.select_action(state)
        assert action == BettingAction.SKIP
    
    def test_recommend_bet_returns_tuple(self, manager):
        """Should return (action, stake) tuple."""
        state = manager.get_current_state(
            confidence=0.7,
            edge=0.05,
            kelly=0.1
        )
        
        action, stake = manager.recommend_bet(state)
        
        assert action in [0, 1, 2, 3]
        assert stake >= 0
    
    def test_update_changes_bankroll(self, manager):
        """Update should modify bankroll."""
        state = manager.get_current_state(
            confidence=0.7,
            edge=0.05,
            kelly=0.1
        )
        
        initial = manager.bankroll
        
        # Win a bet
        manager.update(state, BettingAction.BET_NORMAL, won=True, stake=100, odds=2.0)
        
        # Bankroll should increase
        assert manager.bankroll > initial
    
    def test_update_on_loss(self, manager):
        """Loss should decrease bankroll."""
        state = manager.get_current_state(
            confidence=0.7,
            edge=0.05,
            kelly=0.1
        )
        
        initial = manager.bankroll
        
        manager.update(state, BettingAction.BET_NORMAL, won=False, stake=100, odds=2.0)
        
        assert manager.bankroll < initial
    
    def test_streak_tracking(self, manager):
        """Should track win/loss streaks."""
        state = manager.get_current_state(
            confidence=0.7,
            edge=0.05,
            kelly=0.1
        )
        
        # Win 3 in a row
        for _ in range(3):
            manager.update(state, BettingAction.BET_NORMAL, won=True, stake=10, odds=2.0)
        
        assert manager.current_streak == 3
        
        # Lose 2 in a row
        for _ in range(2):
            manager.update(state, BettingAction.BET_NORMAL, won=False, stake=10, odds=2.0)
        
        assert manager.current_streak == -2
    
    def test_q_table_updates(self, manager):
        """Q-table should be populated during training."""
        state = manager.get_current_state(
            confidence=0.7,
            edge=0.05,
            kelly=0.1
        )
        
        initial_size = len(manager.q_table)
        
        # Take action and update
        action, stake = manager.recommend_bet(state)
        manager.update(state, action, won=True, stake=stake if stake > 0 else 10, odds=2.0)
        
        # Q-table should have entry
        assert len(manager.q_table) >= initial_size
    
    def test_performance_stats(self, manager):
        """Should compute performance statistics."""
        state = manager.get_current_state(
            confidence=0.7,
            edge=0.05,
            kelly=0.1
        )
        
        # Place some bets
        for i in range(10):
            manager.update(state, BettingAction.BET_NORMAL, won=(i % 2 == 0), stake=50, odds=2.0)
        
        stats = manager.get_performance_stats()
        
        assert 'total_bets' in stats
        assert 'win_rate' in stats
        assert 'final_bankroll' in stats
        assert stats['total_bets'] == 10
    
    def test_save_and_load(self, manager):
        """Should save and load model."""
        # Train a bit
        state = manager.get_current_state(
            confidence=0.7,
            edge=0.05,
            kelly=0.1
        )
        
        for i in range(5):
            action, _ = manager.recommend_bet(state)
            manager.update(state, action, won=True, stake=10, odds=2.0)
        
        q_table_size = len(manager.q_table)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            manager.save(str(filepath))
            
            # Load into new manager
            new_manager = RLBankrollManager()
            new_manager.load(str(filepath))
            
            assert len(new_manager.q_table) == q_table_size


class TestAdaptiveKellyManager:
    """Tests for AdaptiveKellyManager."""
    
    @pytest.fixture
    def manager(self):
        """Create fresh manager."""
        return AdaptiveKellyManager()
    
    def test_base_multiplier(self, manager):
        """With neutral performance, multiplier should be ~1."""
        multiplier = manager.get_kelly_multiplier(
            recent_wins=5,
            recent_total=10,
            current_drawdown=0.0
        )
        
        # Should be close to 1 with 50% win rate
        assert 0.8 <= multiplier <= 1.2
    
    def test_hot_streak_increases(self, manager):
        """Hot streak should increase multiplier."""
        multiplier = manager.get_kelly_multiplier(
            recent_wins=8,
            recent_total=10,  # 80% win rate
            current_drawdown=0.0
        )
        
        assert multiplier > 1.0
    
    def test_cold_streak_decreases(self, manager):
        """Cold streak should decrease multiplier."""
        multiplier = manager.get_kelly_multiplier(
            recent_wins=2,
            recent_total=10,  # 20% win rate
            current_drawdown=0.0
        )
        
        assert multiplier < 1.0
    
    def test_drawdown_reduces(self, manager):
        """Significant drawdown should reduce multiplier."""
        normal = manager.get_kelly_multiplier(
            recent_wins=5,
            recent_total=10,
            current_drawdown=0.0
        )
        
        in_drawdown = manager.get_kelly_multiplier(
            recent_wins=5,
            recent_total=10,
            current_drawdown=0.3  # 30% drawdown
        )
        
        assert in_drawdown < normal
    
    def test_update_tracks_results(self, manager):
        """Update should track results."""
        initial = manager.bankroll
        
        manager.update(won=True, stake=100, odds=2.0)
        
        assert manager.bankroll > initial
        assert len(manager.recent_results) == 1
        assert manager.recent_results[-1] == True
    
    def test_recommend_stake(self, manager):
        """Should recommend reasonable stake."""
        stake = manager.recommend_stake(base_kelly=0.1, bankroll=1000)
        
        # Should be some fraction of bankroll
        assert 0 < stake < 500  # Less than 50% of bankroll
    
    def test_current_drawdown(self, manager):
        """Should compute drawdown correctly."""
        assert manager.current_drawdown == 0.0
        
        # Lose a bet
        manager.update(won=False, stake=200, odds=2.0)
        
        assert manager.current_drawdown > 0


class TestTrainingOnHistorical:
    """Tests for training on historical data."""
    
    @pytest.fixture
    def historical_bets(self):
        """Create sample historical betting data."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'model_confidence': np.random.uniform(0.5, 0.9, n),
            'edge': np.random.uniform(-0.05, 0.15, n),
            'kelly': np.random.uniform(0.0, 0.15, n),
            'odds': np.random.uniform(1.5, 3.0, n),
            'won': np.random.binomial(1, 0.55, n)  # 55% win rate
        })
    
    def test_train_on_historical(self, historical_bets):
        """Should train without errors."""
        manager = RLBankrollManager()
        
        # Should complete without error
        manager.train_on_historical(historical_bets, n_epochs=2)
        
        # Q-table should be populated
        assert len(manager.q_table) > 0
        
        # Should not be in training mode after
        assert manager.training == False

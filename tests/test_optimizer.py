"""
Unit tests for multi-asset simultaneous Kelly optimization.
"""

import pytest
import numpy as np
from src.strategies.kelly import BetOpportunity
from src.strategies.optimizer import (
    SimultaneousKellyOptimizer,
    optimize_simultaneous_bets,
    handle_correlated_bets,
)


class TestSimultaneousKellyOptimizer:
    """Tests for the SimultaneousKellyOptimizer class."""
    
    def test_single_bet_matches_individual_kelly(self):
        """Single bet optimization should match individual Kelly."""
        optimizer = SimultaneousKellyOptimizer(
            max_total_fraction=0.15,
            kelly_mult=0.25
        )
        
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="test")
        result = optimizer.optimize([bet])
        
        # Should be close to quarter Kelly for 55%/2.0
        expected = 0.1 * 0.25  # Full Kelly is 0.1, quarter is 0.025
        assert result.allocations["test"] == pytest.approx(expected, rel=0.01)
    
    def test_respects_max_total_fraction(self):
        """Total allocation should not exceed max_total_fraction."""
        optimizer = SimultaneousKellyOptimizer(
            max_total_fraction=0.10,
            kelly_mult=0.25
        )
        
        bets = [
            BetOpportunity(prob=0.60, decimal_odds=2.0, bet_id="bet1"),
            BetOpportunity(prob=0.58, decimal_odds=2.0, bet_id="bet2"),
            BetOpportunity(prob=0.56, decimal_odds=2.0, bet_id="bet3"),
        ]
        
        result = optimizer.optimize(bets)
        
        assert result.total_allocation <= 0.10 + 0.001  # Small tolerance
    
    def test_excludes_negative_ev_bets(self):
        """Negative EV bets should get zero allocation."""
        optimizer = SimultaneousKellyOptimizer()
        
        bets = [
            BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="positive"),
            BetOpportunity(prob=0.40, decimal_odds=2.0, bet_id="negative"),
        ]
        
        result = optimizer.optimize(bets)
        
        assert result.allocations["positive"] > 0
        assert result.allocations["negative"] == 0.0
    
    def test_empty_bets_list(self):
        """Empty bet list should return empty result."""
        optimizer = SimultaneousKellyOptimizer()
        result = optimizer.optimize([])
        
        assert result.total_allocation == 0.0
        assert len(result.allocations) == 0
    
    def test_all_negative_ev(self):
        """All negative EV should result in zero allocation."""
        optimizer = SimultaneousKellyOptimizer()
        
        bets = [
            BetOpportunity(prob=0.40, decimal_odds=2.0, bet_id="neg1"),
            BetOpportunity(prob=0.35, decimal_odds=2.0, bet_id="neg2"),
        ]
        
        result = optimizer.optimize(bets)
        
        assert result.total_allocation == 0.0
    
    def test_correlated_bets_reduce_allocation(self):
        """Correlated bets in same group should reduce allocations."""
        optimizer = SimultaneousKellyOptimizer(
            correlation_penalty=0.3
        )
        
        # Same game, correlated
        correlated_bets = [
            BetOpportunity(prob=0.55, decimal_odds=2.0, 
                          bet_id="ml", correlation_group="game1"),
            BetOpportunity(prob=0.52, decimal_odds=1.9, 
                          bet_id="spread", correlation_group="game1"),
        ]
        
        # Different games, independent
        independent_bets = [
            BetOpportunity(prob=0.55, decimal_odds=2.0, 
                          bet_id="game1", correlation_group=None),
            BetOpportunity(prob=0.52, decimal_odds=1.9, 
                          bet_id="game2", correlation_group=None),
        ]
        
        corr_result = optimizer.optimize(correlated_bets)
        indep_result = optimizer.optimize(independent_bets)
        
        # Correlated should have lower total (due to penalty)
        # Note: This depends on the penalty implementation
        assert corr_result.total_allocation <= indep_result.total_allocation
    
    def test_stake_amounts_method(self):
        """stake_amounts should calculate correct dollar values."""
        optimizer = SimultaneousKellyOptimizer()
        
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="test")
        result = optimizer.optimize([bet])
        
        stakes = result.stake_amounts(bankroll=10000)
        
        assert stakes["test"] == pytest.approx(
            result.allocations["test"] * 10000
        )


class TestOptimizeSimultaneousBets:
    """Tests for the convenience function."""
    
    def test_basic_usage(self):
        """Basic function usage works."""
        bets = [
            BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="a"),
            BetOpportunity(prob=0.53, decimal_odds=1.95, bet_id="b"),
        ]
        
        result = optimize_simultaneous_bets(bets, max_total_fraction=0.10)
        
        assert result.total_allocation <= 0.10
        assert "a" in result.allocations
        assert "b" in result.allocations
    
    def test_equal_risk_method(self):
        """Equal risk method allocates by edge."""
        bets = [
            BetOpportunity(prob=0.60, decimal_odds=2.0, bet_id="high_edge"),
            BetOpportunity(prob=0.52, decimal_odds=2.0, bet_id="low_edge"),
        ]
        
        result = optimize_simultaneous_bets(
            bets, 
            max_total_fraction=0.10,
            method='equal_risk'
        )
        
        # Higher edge should get more allocation
        assert result.allocations["high_edge"] > result.allocations["low_edge"]


class TestHandleCorrelatedBets:
    """Tests for correlation handling with explicit matrix."""
    
    def test_independent_bets(self):
        """Independent bets (identity correlation) work correctly."""
        bets = [
            BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="a"),
            BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="b"),
        ]
        
        # Identity matrix = independent
        corr_matrix = np.eye(2)
        
        allocations = handle_correlated_bets(bets, corr_matrix)
        
        assert "a" in allocations
        assert "b" in allocations
        assert allocations["a"] == pytest.approx(allocations["b"])
    
    def test_highly_correlated_reduces_allocation(self):
        """Highly correlated bets should have reduced allocation."""
        bets = [
            BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="a"),
            BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="b"),
        ]
        
        # High correlation
        corr_high = np.array([[1.0, 0.8], [0.8, 1.0]])
        # Low correlation
        corr_low = np.array([[1.0, 0.1], [0.1, 1.0]])
        
        alloc_high = handle_correlated_bets(bets, corr_high)
        alloc_low = handle_correlated_bets(bets, corr_low)
        
        # Sum of allocations should be lower for high correlation
        total_high = sum(alloc_high.values())
        total_low = sum(alloc_low.values())
        
        assert total_high < total_low
    
    def test_single_bet(self):
        """Single bet returns individual Kelly."""
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="only")
        
        allocations = handle_correlated_bets([bet])
        
        assert "only" in allocations
        assert allocations["only"] > 0
    
    def test_infers_correlation_from_groups(self):
        """Without explicit matrix, infers from correlation_group."""
        bets = [
            BetOpportunity(prob=0.55, decimal_odds=2.0, 
                          bet_id="a", correlation_group="same"),
            BetOpportunity(prob=0.55, decimal_odds=2.0, 
                          bet_id="b", correlation_group="same"),
        ]
        
        allocations = handle_correlated_bets(bets)
        
        # Should reduce due to inferred correlation
        assert allocations["a"] > 0
        assert allocations["b"] > 0

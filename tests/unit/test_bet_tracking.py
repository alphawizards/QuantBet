"""
Unit tests for Bet Tracking Logic.

Tests the core bet tracking calculations including:
- ROI calculation
- Win rate calculation
- Equity curve generation
- Statistics aggregation
- Edge cases and error handling
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Import functions from source module
from src.betting.bet_tracker import (
    calculate_roi,
    calculate_win_rate,
    generate_equity_curve,
    calculate_bet_statistics
)


# ============================================================================
# Unit Tests
# ============================================================================

@pytest.mark.unit
class TestROICalculation:
    """Test Return on Investment (ROI) calculation."""
    
    def test_roi_all_wins(self):
        """Test ROI with all winning bets."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win'},
            {'stake': 100, 'odds': 2.0, 'result': 'win'},
        ]
        
        roi = calculate_roi(bets)
        # Profit = 2*100 - 200 = 200 - 200 = 0... wait
        # Returns = 100*2 + 100*2 = 400
        # Stake = 200
        # Profit = 400 - 200 = 200
        # ROI = 200/200 * 100 = 100%
        assert roi == 100.0
    
    def test_roi_all_losses(self):
        """Test ROI with all losing bets."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'loss'},
            {'stake': 100, 'odds': 2.0, 'result': 'loss'},
        ]
        
        roi = calculate_roi(bets)
        # Returns = 0
        # Stake = 200
        # Profit = -200
        # ROI = -200/200 * 100 = -100%
        assert roi == -100.0
    
    def test_roi_mixed_results(self):
        """Test ROI with mixed win/loss results."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win'},   # +100
            {'stake': 100, 'odds': 2.0, 'result': 'loss'},  # -100
            {'stake': 100, 'odds': 1.5, 'result': 'win'},   # +50
        ]
        
        roi = calculate_roi(bets)
        # Returns = 200 + 0 + 150 = 350
        # Stake = 300
        # Profit = 50
        # ROI = 50/300 * 100 = 16.67%
        assert abs(roi - 16.67) < 0.1
    
    def test_roi_with_pending_bets(self):
        """Test ROI ignores pending bets (no result)."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win'},
            {'stake': 100, 'odds': 2.0, 'result': None},  # Pending - ignored
        ]
        
        roi = calculate_roi(bets)
        # Only first bet counts: (200-100)/100 * 100 = 100%
        assert roi == 100.0
    
    def test_roi_empty_bets_raises_error(self):
        """Test that empty bets raise ValueError."""
        with pytest.raises(ValueError, match="no bets"):
            calculate_roi([])
    
    def test_roi_only_pending_bets(self):
        """Test ROI when all bets are pending."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': None},
            {'stake': 100, 'odds': 2.0, 'result': None},
        ]
        
        roi = calculate_roi(bets)
        assert roi == 0.0


@pytest.mark.unit
class TestWinRateCalculation:
    """Test win rate calculation."""
    
    def test_win_rate_all_wins(self):
        """Test win rate with 100% wins."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win'},
            {'stake': 100, 'odds': 1.5, 'result': 'win'},
        ]
        
        win_rate = calculate_win_rate(bets)
        assert win_rate == 100.0
    
    def test_win_rate_all_losses(self):
        """Test win rate with 0% wins."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'loss'},
            {'stake': 100, 'odds': 1.5, 'result': 'loss'},
        ]
        
        win_rate = calculate_win_rate(bets)
        assert win_rate == 0.0
    
    def test_win_rate_fifty_percent(self):
        """Test win rate with 50% wins."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win'},
            {'stake': 100, 'odds': 2.0, 'result': 'loss'},
        ]
        
        win_rate = calculate_win_rate(bets)
        assert win_rate == 50.0
    
    def test_win_rate_with_pending_bets(self):
        """Test win rate ignores pending bets."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win'},
            {'stake': 100, 'odds': 2.0, 'result': None},  # Pending - ignored
            {'stake': 100, 'odds': 2.0, 'result': 'loss'},
        ]
        
        win_rate = calculate_win_rate(bets)
        # 1 win out of 2 completed = 50%
        assert win_rate == 50.0
    
    def test_win_rate_empty_bets_raises_error(self):
        """Test that empty bets raise ValueError."""
        with pytest.raises(ValueError, match="no bets"):
            calculate_win_rate([])


@pytest.mark.unit
class TestEquityCurve:
    """Test equity curve generation."""
    
    def test_equity_curve_all_wins(self):
        """Test equity curve with increasing bankroll."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win', 'timestamp': datetime(2024, 1, 1)},
            {'stake': 100, 'odds': 2.0, 'result': 'win', 'timestamp': datetime(2024, 1, 2)},
        ]
        
        curve = generate_equity_curve(bets, starting_bankroll=1000)
        
        assert len(curve) == 2
        assert curve[0]['bankroll'] == 1100  # 1000 + 100 profit
        assert curve[1]['bankroll'] == 1200  # 1100 + 100 profit
        assert curve[1]['cumulative_profit'] == 200
    
    def test_equity_curve_all_losses(self):
        """Test equity curve with decreasing bankroll."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'loss', 'timestamp': datetime(2024, 1, 1)},
            {'stake': 100, 'odds': 2.0, 'result': 'loss', 'timestamp': datetime(2024, 1, 2)},
        ]
        
        curve = generate_equity_curve(bets, starting_bankroll=1000)
        
        assert len(curve) == 2
        assert curve[0]['bankroll'] == 900   # 1000 - 100 loss
        assert curve[1]['bankroll'] == 800   # 900 - 100 loss
        assert curve[1]['cumulative_profit'] == -200
    
    def test_equity_curve_mixed_results(self):
        """Test equity curve with mixed results."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win', 'timestamp': datetime(2024, 1, 1)},
            {'stake': 100, 'odds': 2.0, 'result': 'loss', 'timestamp': datetime(2024, 1, 2)},
        ]
        
        curve = generate_equity_curve(bets, starting_bankroll=1000)
        
        assert len(curve) == 2
        assert curve[0]['bankroll'] == 1100  # Win +100
        assert curve[1]['bankroll'] == 1000  # Loss -100, back to starting
        assert curve[1]['cumulative_profit'] == 0
    
    def test_equity_curve_ignores_pending_bets(self):
        """Test equity curve skips pending bets."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win', 'timestamp': datetime(2024, 1, 1)},
            {'stake': 100, 'odds': 2.0, 'result': None, 'timestamp': datetime(2024, 1, 2)},
        ]
        
        curve = generate_equity_curve(bets, starting_bankroll=1000)
        
        # Only 1 point (pending bet skipped)
        assert len(curve) == 1
        assert curve[0]['bankroll'] == 1100
    
    def test_equity_curve_empty_bets(self):
        """Test equity curve with no bets."""
        curve = generate_equity_curve([])
        assert curve == []
    
    def test_equity_curve_sorts_by_timestamp(self):
        """Test equity curve sorts bets chronologically."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win', 'timestamp': datetime(2024, 1, 3)},
            {'stake': 100, 'odds': 2.0, 'result': 'loss', 'timestamp': datetime(2024, 1, 1)},
            {'stake': 100, 'odds': 2.0, 'result': 'win', 'timestamp': datetime(2024, 1, 2)},
        ]
        
        curve = generate_equity_curve(bets, starting_bankroll=1000)
        
        # Should be ordered: loss, win, win
        assert curve[0]['bankroll'] == 900   # Loss first
        assert curve[1]['bankroll'] == 1000  # Win second
        assert curve[2]['bankroll'] == 1100  # Win third


@pytest.mark.unit
class TestBetStatistics:
    """Test comprehensive bet statistics calculation."""
    
    def test_statistics_comprehensive(self):
        """Test all statistics calculated correctly."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'win', 'timestamp': datetime(2024, 1, 1)},
            {'stake': 100, 'odds': 2.0, 'result': 'loss', 'timestamp': datetime(2024, 1, 2)},
            {'stake': 100, 'odds': 3.0, 'result': 'win', 'timestamp': datetime(2024, 1, 3)},
            {'stake': 50, 'odds': 1.5, 'result': None, 'timestamp': datetime(2024, 1, 4)},
        ]
        
        stats = calculate_bet_statistics(bets)
        
        assert stats['total_bets'] == 4
        assert stats['completed_bets'] == 3
        assert stats['pending_bets'] == 1
        assert abs(stats['win_rate'] - 66.67) < 0.1  # 2/3 wins
        assert stats['total_staked'] == 300
        # Profit: +100 + (-100) + 200 = +200
        assert stats['total_profit'] == 200
        assert abs(stats['roi'] - 66.67) < 0.1  # 200/300 * 100
        assert abs(stats['average_odds'] - 2.33) < 0.1  # (2+2+3)/3
        assert stats['largest_win'] == 200  # 100 * (3.0 - 1)
        assert stats['largest_loss'] == -100
    
    def test_statistics_empty_bets(self):
        """Test statistics with no bets."""
        stats = calculate_bet_statistics([])
        
        assert stats['total_bets'] == 0
        assert stats['completed_bets'] == 0
        assert stats['roi'] == 0.0
        assert stats['win_rate'] == 0.0
    
    def test_statistics_all_pending(self):
        """Test statistics with all pending bets."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': None, 'timestamp': datetime(2024, 1, 1)},
        ]
        
        stats = calculate_bet_statistics(bets)
        
        assert stats['total_bets'] == 1
        assert stats['completed_bets'] == 0
        assert stats['pending_bets'] == 1
        assert stats['roi'] == 0.0
        assert stats['win_rate'] == 0.0


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_stake_bets(self):
        """Test handling of bets with zero stake."""
        bets = [
            {'stake': 0, 'odds': 2.0, 'result': 'win'},
        ]
        
        stats = calculate_bet_statistics(bets)
        assert stats['total_staked'] == 0
        assert stats['roi'] == 0.0  # Avoid division by zero
    
    def test_very_high_odds(self):
        """Test handling of very high odds."""
        bets = [
            {'stake': 10, 'odds': 100.0, 'result': 'win', 'timestamp': datetime(2024, 1, 1)},
        ]
        
        curve = generate_equity_curve(bets, starting_bankroll=1000)
        # Profit = 10 * (100 - 1) = 990
        assert curve[0]['bankroll'] == 1990
    
    def test_push_result(self):
        """Test handling of push/void bets."""
        bets = [
            {'stake': 100, 'odds': 2.0, 'result': 'push', 'timestamp': datetime(2024, 1, 1)},
        ]
        
        curve = generate_equity_curve(bets, starting_bankroll=1000)
        assert curve[0]['bankroll'] == 1000  # No change
        assert curve[0]['cumulative_profit'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

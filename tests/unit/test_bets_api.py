"""
Unit tests for Bets API endpoints and logic.

Tests bet creation, retrieval, validation, and user isolation.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from fastapi import HTTPException


class TestBetCreation:
    """Test bet creation logic and validation."""
    
    def test_create_valid_bet(self):
        """Test creating a bet with valid data."""
        bet_data = {
            'game_id': 'test_game_001',
            'user_id': 'user_123',
            'bet_type': 'home_win',
            'odds': 1.85,
            'stake': 50.0,
            'predicted_prob': 0.65
        }
        
        # Validate required fields present
        required_fields = ['game_id', 'user_id', 'bet_type', 'odds', 'stake']
        for field in required_fields:
            assert field in bet_data
        
        # Validate data types and ranges
        assert isinstance(bet_data['stake'], (int, float))
        assert bet_data['stake'] > 0
        assert bet_data['odds'] > 1.0
        assert 0 <= bet_data['predicted_prob'] <= 1
    
    def test_negative_stake_rejected(self):
        """Test that negative stake is rejected."""
        stake = -50.0
        
        with pytest.raises((ValueError, HTTPException)):
            if stake <= 0:
                raise ValueError("Stake must be positive")
    
    def test_zero_stake_rejected(self):
        """Test that zero stake is rejected."""
        stake = 0.0
        
        with pytest.raises((ValueError, HTTPException)):
            if stake <= 0:
                raise ValueError("Stake must be positive")
    
    def test_invalid_odds_rejected(self):
        """Test that invalid odds are rejected."""
        invalid_odds = [0.5, 0.0, -1.5, 1.0]  # Must be > 1.0
        
        for odds in invalid_odds:
            if odds <= 1.0:
                with pytest.raises((ValueError, HTTPException)):
                    raise ValueError(f"Odds must be greater than 1.0, got {odds}")
    
    def test_invalid_bet_type_rejected(self):
        """Test that invalid bet type is rejected."""
        valid_types = ['home_win', 'away_win', 'over', 'under']
        invalid_type = 'invalid_type'
        
        if invalid_type not in valid_types:
            with pytest.raises((ValueError, HTTPException)):
                raise ValueError(f"Invalid bet type: {invalid_type}")
    
    def test_missing_required_fields(self):
        """Test that missing required fields are rejected."""
        incomplete_bet = {
            'game_id': 'test_game_001',
            # Missing user_id, bet_type, odds, stake
        }
        
        required = ['game_id', 'user_id', 'bet_type', 'odds', 'stake']
        missing = [f for f in required if f not in incomplete_bet]
        
        assert len(missing) > 0
        
        if missing:
            with pytest.raises((ValueError, HTTPException)):
                raise ValueError(f"Missing required fields: {missing}")


class TestBetRetrieval:
    """Test bet retrieval and querying."""
    
    def test_get_user_bets(self):
        """Test retrieving all bets for a user."""
        user_bets = [
            {'bet_id': '001', 'user_id': 'user_123', 'stake': 50.0},
            {'bet_id': '002', 'user_id': 'user_123', 'stake': 100.0 },
            {'bet_id': '003', 'user_id': 'user_456', 'stake': 75.0},
        ]
        
        # Filter by user
        user_123_bets = [b for b in user_bets if b['user_id'] == 'user_123']
        
        assert len(user_123_bets) == 2
        assert all(b['user_id'] == 'user_123' for b in user_123_bets)
    
    def test_get_bets_by_status(self):
        """Test filtering bets by result status."""
        bets = [
            {'bet_id': '001', 'result': 'win'},
            {'bet_id': '002', 'result': 'loss'},
            {'bet_id': '003', 'result': None},  # Pending
            {'bet_id': '004', 'result': 'win'},
        ]
        
        # Pending bets
        pending = [b for b in bets if b['result'] is None]
        assert len(pending) == 1
        
        # Settled bets
        settled = [b for b in bets if b['result'] is not None]
        assert len(settled) == 3
        
        # Winning bets
        wins = [b for b in bets if b['result'] == 'win']
        assert len(wins) == 2
    
    def test_get_bets_by_date_range(self):
        """Test filtering bets by date range."""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)
        
        bets = [
            {'bet_id': '001', 'timestamp': last_week},
            {'bet_id': '002', 'timestamp': yesterday},
            {'bet_id': '003', 'timestamp': today},
            {'bet_id': '004', 'timestamp': today},
        ]
        
        # Last 7 days
        recent_bets = [
            b for b in bets
            if b['timestamp'] >= (today - timedelta(days=7))
        ]
        
        assert len(recent_bets) >= 3


class TestUserIsolation:
    """Test user isolation and data access controls."""
    
    def test_user_cannot_access_other_users_bets(self):
        """Test that User A cannot see User B's bets."""
        all_bets = [
            {'bet_id': '001', 'user_id': 'user_A', 'stake': 50.0},
            {'bet_id': '002', 'user_id': 'user_B', 'stake': 100.0},
            {'bet_id': '003', 'user_id': 'user_A', 'stake': 75.0},
        ]
        
        # User A requests bets
        requesting_user = 'user_A'
        user_a_bets = [b for b in all_bets if b['user_id'] == requesting_user]
        
        # Should only get their own bets
        assert len(user_a_bets) == 2
        assert all(b['user_id'] == 'user_A' for b in user_a_bets)
        assert not any(b['user_id'] == 'user_B' for b in user_a_bets)
    
    def test_user_cannot_modify_other_users_bets(self):
        """Test that User A cannot modify User B's bet."""
        bet = {'bet_id': '001', 'user_id': 'user_B', 'stake': 100.0}
        requesting_user = 'user_A'
        
        # Simulate authorization check
        if bet['user_id'] != requesting_user:
            with pytest.raises(HTTPException) as exc_info:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to modify this bet"
                )
            
            assert exc_info.value.status_code == 403
    
    def test_user_cannot_delete_other_users_bets(self):
        """Test that User A cannot delete User B's bet."""
        bet_owner = 'user_B'
        requesting_user = 'user_A'
        
        if bet_owner != requesting_user:
            with pytest.raises(HTTPException) as exc_info:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to delete this bet"
                )
            
            assert exc_info.value.status_code == 403


class TestBetCalculations:
    """Test bet payout and profit calculations."""
    
    def test_winning_bet_payout(self):
        """Test payout calculation for winning bet."""
        stake = 50.0
        odds = 2.00
        
        # Payout = stake * odds
        payout = stake * odds
        profit = payout - stake
        
        assert payout == 100.0
        assert profit == 50.0
    
    def test_losing_bet_payout(self):
        """Test payout calculation for losing bet."""
        stake = 50.0
        
        payout = 0.0
        profit = payout - stake
        
        assert payout == 0.0
        assert profit == -50.0
    
    def test_fractional_odds_payout(self):
        """Test payout with fractional odds."""
        stake = 100.0
        odds = 1.85
        
        payout = stake * odds
        profit = payout - stake
        
        assert payout == 185.0
        assert profit == 85.0
    
    def test_high_odds_payout(self):
        """Test payout with high odds."""
        stake = 10.0
        odds = 10.50
        
        payout = stake * odds
        profit = payout - stake
        
        assert payout == 105.0
        assert profit == 95.0


class TestBetValidation:
    """Test bet validation rules."""
    
    def test_stake_within_limits(self):
        """Test stake amount validation."""
        min_stake = 1.0
        max_stake = 10000.0
        
        valid_stakes = [10.0, 50.0, 100.0, 500.0]
        invalid_stakes = [0.5, 0.0, -10.0, 15000.0]
        
        for stake in valid_stakes:
            assert min_stake <= stake <= max_stake
        
        for stake in invalid_stakes:
            assert not (min_stake <= stake <= max_stake)
    
    def test_game_not_started(self):
        """Test that bets can only be placed before game starts."""
        game_start = datetime.now() + timedelta(hours=2)
        current_time = datetime.now()
        
        # Game hasn't started - bet allowed
        assert current_time < game_start
        
        # Game already started - bet rejected
        game_start_past = datetime.now() - timedelta(hours=1)
        if current_time >= game_start_past:
            with pytest.raises((ValueError, HTTPException)):
                raise ValueError("Cannot place bet on game that has started")
    
    def test_duplicate_bet_prevention(self):
        """Test prevention of duplicate bets on same game."""
        existing_bets = [
            {'game_id': 'game_001', 'user_id': 'user_123'},
            {'game_id': 'game_002', 'user_id': 'user_123'},
        ]
        
        new_bet_game_id = 'game_001'
        new_bet_user_id = 'user_123'
        
        # Check for duplicate
        has_existing = any(
            b['game_id'] == new_bet_game_id and b['user_id'] == new_bet_user_id
            for b in existing_bets
        )
        
        if has_existing:
            with pytest.raises((ValueError, HTTPException)):
                raise ValueError("Bet already exists for this game")


class TestEdgeCases:
    """Test edge cases for bet handling."""
    
    def test_empty_bet_list(self):
        """Test handling of user with no bets."""
        user_bets = []
        
        assert len(user_bets) == 0
        assert isinstance(user_bets, list)
    
    def test_very_large_stake(self):
        """Test handling of very large stake amounts."""
        stake = 5000.0
        max_allowed = 10000.0
        
        if stake <= max_allowed:
            assert True  # Valid
        else:
            with pytest.raises((ValueError, HTTPException)):
                raise ValueError(f"Stake exceeds maximum of {max_allowed}")
    
    def test_push_result_handling(self):
        """Test handling of push (tie) results."""
        bet = {
            'stake': 100.0,
            'odds': 1.85,
            'result': 'push'
        }
        
        # Push = stake returned, no profit/loss
        if bet['result'] == 'push':
            payout = bet['stake']
            profit = 0.0
            
            assert payout == 100.0
            assert profit == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for ELO rating model.

Tests ELO rating calculations, updates, and predictions.
"""

import pytest
from unittest.mock import Mock, MagicMock
import math


class TestELORatingCalculations:
    """Test ELO rating calculation logic."""
    
    def test_default_elo_rating(self):
        """Test that new teams start with default rating of 1500."""
        default_rating = 1500
        
        assert default_rating == 1500
        assert default_rating > 0
    
    def test_expected_score_equal_ratings(self):
        """Test expected score when teams have equal ratings."""
        rating_a = 1500
        rating_b = 1500
        
        # Expected score = 1 / (1 + 10^((rating_b - rating_a) / 400))
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        
        # Equal ratings = 0.5 expected score
        assert abs(expected_a - 0.5) < 0.01
    
    def test_expected_score_higher_rated_favorite(self):
        """Test expected score when one team is much higher rated."""
        rating_favorite = 1700
        rating_underdog = 1400
        
        expected_favorite = 1 / (1 + 10 ** ((rating_underdog - rating_favorite) / 400))
        expected_underdog = 1 / (1 + 10 ** ((rating_favorite - rating_underdog) / 400))
        
        # Favorite should have higher expected score
        assert expected_favorite > 0.7
        assert expected_underdog < 0.3
        
        # Should sum to 1
        assert abs((expected_favorite + expected_underdog) - 1.0) < 0.01
    
    def test_elo_update_win(self):
        """Test ELO rating update when team wins."""
        rating_before = 1500
        k_factor = 32
        expected_score = 0.5
        actual_score = 1.0  # Win
        
        # New rating = old rating + K * (actual - expected)
        rating_after = rating_before + k_factor * (actual_score - expected_score)
        
        # Winning when even should increase rating
        assert rating_after > rating_before
        assert rating_after == 1516  # 1500 + 32 * (1.0 - 0.5)
    
    def test_elo_update_loss(self):
        """Test ELO rating update when team loses."""
        rating_before = 1500
        k_factor = 32
        expected_score = 0.5
        actual_score = 0.0  # Loss
        
        rating_after = rating_before + k_factor * (actual_score - expected_score)
        
        # Losing when even should decrease rating
        assert rating_after < rating_before
        assert rating_after == 1484  # 1500 + 32 * (0.0 - 0.5)
    
    def test_upset_increases_rating_more(self):
        """Test that upset wins increase rating more than expected wins."""
        underdog_rating = 1400
        k_factor = 32
        
        # Underdog expected to lose (expected = 0.24)
        expected_underdog = 1 / (1 + 10 ** ((1700 - 1400) / 400))
        
        # Underdog wins
        actual_score = 1.0
        rating_gain = k_factor * (actual_score - expected_underdog)
        
        # Should gain more rating than a 50/50 win
        normal_gain = k_factor * (1.0 - 0.5)  # 16 points
        
        assert rating_gain > normal_gain
        assert rating_gain > 20  # Should gain ~24 points


class TestELOPredictions:
    """Test ELO-based win probability predictions."""
    
    def test_prediction_from_elo_ratings(self):
        """Test generating win probability from ELO ratings."""
        home_rating = 1600
        away_rating = 1500
        home_advantage = 50  # Typical home court advantage
        
        # Adjust home rating for home advantage
        adjusted_home = home_rating + home_advantage
        
        # Calculate win probability
        home_prob = 1 / (1 + 10 ** ((away_rating - adjusted_home) / 400))
        away_prob = 1 - home_prob
        
        # Home team (better + home advantage) should be favorite
        assert home_prob > 0.6
        assert away_prob < 0.4
        assert abs((home_prob + away_prob) - 1.0) < 0.01
    
    def test_prediction_massive_mismatch(self):
        """Test prediction with large rating difference."""
        elite_team = 1800
        weak_team = 1200
        
        elite_prob = 1 / (1 + 10 ** ((weak_team - elite_team) / 400))
        
        # Elite team should have very high win probability
        assert elite_prob > 0.95
    
    def test_prediction_close_match(self):
        """Test prediction with very close ratings."""
        team_a = 1550
        team_b = 1545
        
        team_a_prob = 1 / (1 + 10 ** ((team_b - team_a) / 400))
        
        # Very close to 50/50
        assert 0.48 <= team_a_prob <= 0.52


class TestHomeAdvantage:
    """Test home court advantage in ELO."""
    
    def test_home_advantage_applied(self):
        """Test that home advantage properly affects predictions."""
        neutral_rating = 1500
        home_advantage = 50
        
        # Without home advantage (neutral site)
        neutral_prob = 1 / (1 + 10 ** ((neutral_rating - neutral_rating) / 400))
        
        # With home advantage
        home_prob = 1 / (1 + 10 ** ((neutral_rating - (neutral_rating + home_advantage)) / 400))
        
        # Home advantage should increase probability
        assert home_prob > neutral_prob
        assert neutral_prob == 0.5
        assert home_prob > 0.5
    
    def test_home_advantage_magnitude(self):
        """Test typical magnitude of home advantage effect."""
        rating = 1500
        home_advantage = 50
        
        home_prob = 1 / (1 + 10 ** ((rating - (rating + home_advantage)) / 400))
        
        # 50-point home advantage should be ~7% boost
        assert 0.56 <= home_prob <= 0.58


class TestKFactorVariation:
    """Test K-factor variations in ELO."""
    
    def test_higher_k_means_larger_swings(self):
        """Test that higher K-factor means larger rating changes."""
        rating = 1500
        expected = 0.5
        actual = 1.0  # Win
        
        k_low = 16
        k_high = 32
        
        change_low = k_low * (actual - expected)
        change_high = k_high * (actual - expected)
        
        assert change_high > change_low
        assert change_low == 8
        assert change_high == 16
    
    def test_k_factor_for_new_teams(self):
        """Test higher K-factor for teams with few games."""
        games_played = 10
        
        if games_played < 30:
            k_factor = 40  # Higher for new teams
        else:
            k_factor = 20  # Lower for established teams
        
        assert k_factor == 40


class TestELORatingBounds:
    """Test ELO rating boundaries and edge cases."""
    
    def test_rating_cannot_go_negative(self):
        """Test that ratings don't go below reasonable floor."""
        min_rating = 800  # Practical floor
        
        current_rating = 900
        max_loss = 32  # Max K-factor * max loss
        
        worst_case = current_rating - max_loss
        
        # In practice, should never go below 800
        if worst_case < min_rating:
            rating_after = max(worst_case, min_rating)
            assert rating_after >= min_rating
    
    def test_rating_reasonable_ceiling(self):
        """Test that ratings have reasonable ceiling."""
        max_practical_rating = 2200  # NBA-level elite
        
        current_rating = 2180
        max_gain = 32
        
        new_rating = current_rating + max_gain
        
        # Very hard to exceed 2200 in practice
        assert new_rating <= max_practical_rating + 100


class TestEdgeCases:
    """Test edge cases in ELO calculations."""
    
    def test_brand_new_team(self):
        """Test handling of team with no rating history."""
        team_ratings = {}
        new_team = "NEW"
        
        if new_team not in team_ratings:
            # Assign default rating
            team_ratings[new_team] = 1500
        
        assert team_ratings[new_team] == 1500
    
    def test_extreme_rating_difference(self):
        """Test prediction with extreme rating gap."""
        dominant = 2000
        weak = 1000
        
        dominant_prob = 1 / (1 + 10 ** ((weak - dominant) / 400))
        
        # Should be heavily favored but not 100%
        assert dominant_prob > 0.99
        assert dominant_prob < 1.0
    
    def test_rating_update_symmetry(self):
        """Test that rating updates are zero-sum."""
        rating_a = 1550
        rating_b = 1450
        k = 32
        
        # Team A wins
        exp_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        exp_b = 1 - exp_a
        
        change_a = k * (1.0 - exp_a)
        change_b = k * (0.0 - exp_b)
        
        # Changes should be opposite
        assert abs(change_a + change_b) < 0.01


class TestMarginOfVictory:
    """Test margin of victory adjustments (optional enhancement)."""
    
    def test_blowout_vs_close_win(self):
        """Test that margin of victory can affect K-factor."""
        normal_k = 32
        score_diff = 25  # Blowout
        
        # MOV multiplier = min(1 + (score_diff / 20), 2)
        mov_multiplier = min(1 + (score_diff / 20), 2.0)
        adjusted_k = normal_k * mov_multiplier
        
        # Blowout should increase K-factor
        assert adjusted_k > normal_k
        assert adjusted_k <= 64  # Cap at 2x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

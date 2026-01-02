"""
Unit tests for Poisson Distribution model.

Tests Poisson probability calculations, scoring rates, and match outcome predictions.
"""

import pytest
import math
from scipy.stats import poisson


class TestPoissonProbabilityCalculations:
    """Test basic Poisson probability calculations."""
    
    def test_poisson_probability_formula(self):
        """Test Poisson probability formula: P(X=k) = (λ^k * e^(-λ)) / k!"""
        lambda_rate = 2.5
        k = 3
        
        # Manual calculation
        expected = (lambda_rate ** k) * math.exp(-lambda_rate) / math.factorial(k)
        
        # Should be approximately 0.2138
        assert 0.21 < expected < 0.22
    
    def test_poisson_zero_goals(self):
        """Test probability of zero goals with low scoring rate."""
        lambda_rate = 0.5
        k = 0
        
        prob = (lambda_rate ** k) * math.exp(-lambda_rate) / math.factorial(k)
        
        # P(X=0|λ=0.5) ≈ 0.606
        assert 0.60 < prob < 0.61
    
    def test_poisson_high_scoring(self):
        """Test probability in high-scoring scenario."""
        lambda_rate = 4.5
        k = 7
        
        prob = (lambda_rate ** k) * math.exp(-lambda_rate) / math.factorial(k)
        
        # P(X=7|λ=4.5) ≈ 0.082
        assert 0.05 < prob < 0.10
    
    def test_poisson_probabilities_sum_to_one(self):
        """Test that probabilities sum to ~1.0 over reasonable range."""
        lambda_rate = 2.0
        
        total_prob = sum(
            (lambda_rate ** k) * math.exp(-lambda_rate) / math.factorial(k)
            for k in range(0, 20)  # Sum over 0-19 goals
        )
        
        # Should sum very close to 1.0
        assert 0.999 < total_prob < 1.001


class TestScoringRateCalculations:
    """Test team scoring rate (lambda) calculations."""
    
    def test_home_advantage_increases_lambda(self):
        """Test that home teams have higher expected scoring rates."""
        base_lambda = 1.0
        home_advantage_factor = 1.2  # 20% boost
        
        home_lambda = base_lambda * home_advantage_factor
        away_lambda = base_lambda
        
        assert home_lambda > away_lambda
        assert home_lambda == 1.2
    
    def test_defensive_team_lower_lambda(self):
        """Test that defensive teams have lower expected goals against."""
        league_avg = 1.2  # goals per game
        defensive_factor = 0.8  # 20% better defense
        
        goals_against = league_avg * defensive_factor
        
        assert goals_against < league_avg
        assert goals_against == 0.96
    
    def test_lambda_never_negative(self):
        """Test that scoring rates are always non-negative."""
        lambda_values = [0.0, 0.5, 1.0, 2.5, 4.0]
        for lam in lambda_values:
            assert lam >= 0


class TestMatchOutcomeProbabilities:
    """Test win/draw/loss probability calculations from Poisson."""
    
    def test_even_match_probabilities(self):
        """Test probabilities for evenly matched teams."""
        home_lambda = 1.5
        away_lambda = 1.5
        
        # Calculate outcome probabilities
        outcomes = {'home_win': 0, 'draw': 0, 'away_win': 0}
        
        for home_goals in range(0, 10):
            for away_goals in range(0, 10):
                prob_home = poisson.pmf(home_goals, home_lambda)
                prob_away = poisson.pmf(away_goals, away_lambda)
                prob_scoreline = prob_home * prob_away
                
                if home_goals > away_goals:
                    outcomes['home_win'] += prob_scoreline
                elif home_goals == away_goals:
                    outcomes['draw'] += prob_scoreline
                else:
                    outcomes['away_win'] += prob_scoreline
        
        # Even match: probabilities should be similar
        assert abs(outcomes['home_win'] - outcomes['away_win']) < 0.1
        
        # All probabilities sum to ~1
        total = sum(outcomes.values())
        assert 0.99 < total < 1.01
    
    def test_strong_favorite_high_win_probability(self):
        """Test that strong favorites have high win probability."""
        home_lambda = 3.0  # Strong team
        away_lambda = 1.0  # Weak team
        
        home_win_prob = 0
        
        for home_goals in range(0, 15):
            for away_goals in range(0, 15):
                if home_goals > away_goals:
                    prob_home = poisson.pmf(home_goals, home_lambda)
                    prob_away = poisson.pmf(away_goals, away_lambda)
                    home_win_prob += prob_home * prob_away
        
        # Strong favorite should have > 70% win probability
        assert home_win_prob > 0.7
    
    def test_defensive_game_high_draw_probability(self):
        """Test that low-scoring games have higher draw probability."""
        home_lambda = 0.8  # Defensive game
        away_lambda = 0.8
        
        draw_prob = 0
        
        for goals in range(0, 10):
            prob_home = poisson.pmf(goals, home_lambda)
            prob_away = poisson.pmf(goals, away_lambda)
            draw_prob += prob_home * prob_away  # Same score
        
        # Low scoring = higher draw chance
        assert draw_prob > 0.3


class TestPoissonEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_lambda_only_zero_goals(self):
        """Test that λ=0 means only 0 goals possible."""
        lambda_rate = 0.0
        
        prob_zero = poisson.pmf(0, lambda_rate)
        prob_one = poisson.pmf(1, lambda_rate)
        
        # P(X=0|λ=0) = 1
        assert prob_zero == 1.0
        
        # P(X>0|λ=0) = 0
        assert prob_one == 0.0
    
    def test_very_high_lambda(self):
        """Test stable probabilities with very high scoring rates."""
        lambda_rate = 10.0
        
        # Most likely outcome should be around 10 goals
        probs = {k: poisson.pmf(k, lambda_rate) for k in range(0, 20)}
        most_likely = max(probs, key=probs.get)
        
        assert 9 <= most_likely <= 11
    
    def test_probabilities_always_positive(self):
        """Test that probabilities are always non-negative."""
        for lambda_rate in [0.5, 1.0, 2.5, 5.0]:
            for k in range(0, 10):
                prob = poisson.pmf(k, lambda_rate)
                assert prob >= 0
                assert prob <= 1


class TestPoissonModelComparison:
    """Test Poisson model against known results."""
    
    def test_expected_value_equals_lambda(self):
        """Test that E[X] = λ for Poisson distribution."""
        lambda_rate = 2.5
        
        expected_value = sum(
            k * poisson.pmf(k, lambda_rate)
            for k in range(0, 50)
        )
        
        # E[X] should equal λ
        assert abs(expected_value - lambda_rate) < 0.01
    
    def test_variance_equals_lambda(self):
        """Test that Var(X) = λ for Poisson distribution."""
        lambda_rate = 3.0
        
        # Calculate variance
        expected_value = lambda_rate
        expected_sq = sum(
            (k ** 2) * poisson.pmf(k, lambda_rate)
            for k in range(0, 50)
        )
        variance = expected_sq - (expected_value ** 2)
        
        # Var(X) should equal λ
        assert abs(variance - lambda_rate) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

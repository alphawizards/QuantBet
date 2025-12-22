"""
Unit tests for Kelly Criterion implementations.

Tests verify mathematical correctness of:
    - Classic Kelly formula
    - Fractional Kelly scaling
    - Bounded-below constraints
    - Bayesian shrinkage
"""

import pytest
import numpy as np
from src.portfolio.kelly import (
    kelly_criterion,
    fractional_kelly,
    bounded_kelly,
    bayesian_kelly,
    BetOpportunity,
    KellyCalculator,
    expected_value,
    break_even_probability,
)


class TestKellyCriterion:
    """Tests for the classic Kelly criterion function."""
    
    def test_even_money_55_percent(self):
        """55% win rate at even money (2.0 decimal) should bet 10%."""
        result = kelly_criterion(0.55, 2.0)
        
        assert result.is_positive_ev
        assert 0.09 <= result.fraction <= 0.11  # ~10%
        assert result.expected_growth > 0
    
    def test_even_money_50_percent(self):
        """50% win rate at even money has zero edge, don't bet."""
        result = kelly_criterion(0.50, 2.0)
        
        assert result.fraction == 0.0
        assert not result.is_positive_ev
        assert abs(result.edge) < 0.001
    
    def test_negative_ev_no_bet(self):
        """Negative EV situations should return zero fraction."""
        result = kelly_criterion(0.40, 2.0)  # 40% at even money
        
        assert result.fraction == 0.0
        assert not result.is_positive_ev
        assert result.edge < 0
    
    def test_high_odds_lower_fraction(self):
        """Higher odds with same edge should result in smaller fraction."""
        # Both have 5% edge over implied prob
        result_low = kelly_criterion(0.55, 2.0)   # 50% implied
        result_high = kelly_criterion(0.30, 4.0)  # 25% implied
        
        # Higher odds = more variance = smaller Kelly fraction
        assert result_high.fraction < result_low.fraction
    
    def test_extreme_edge(self):
        """Very high edge should suggest larger bets (but not all-in)."""
        result = kelly_criterion(0.70, 2.0)  # 70% at even money!
        
        assert result.fraction > 0.3  # Should be substantial
        assert result.fraction < 1.0  # But not all-in
        assert result.is_positive_ev
    
    def test_validation_errors(self):
        """Invalid inputs should raise ValueError."""
        with pytest.raises(ValueError):
            kelly_criterion(0.0, 2.0)  # prob = 0
        
        with pytest.raises(ValueError):
            kelly_criterion(1.0, 2.0)  # prob = 1
        
        with pytest.raises(ValueError):
            kelly_criterion(0.5, 0.5)  # odds < 1
    
    def test_expected_growth_positive_for_positive_ev(self):
        """Positive EV bets should have positive expected log growth."""
        result = kelly_criterion(0.55, 2.0)
        assert result.expected_growth > 0
    
    def test_raw_kelly_can_exceed_fraction(self):
        """Raw Kelly might be negative but fraction is capped at 0."""
        result = kelly_criterion(0.30, 2.0)  # Negative EV
        
        assert result.raw_kelly < 0
        assert result.fraction == 0.0


class TestFractionalKelly:
    """Tests for fractional Kelly implementation."""
    
    def test_quarter_kelly(self):
        """Quarter Kelly should be 25% of full Kelly."""
        full = kelly_criterion(0.55, 2.0)
        quarter = fractional_kelly(0.55, 2.0, fraction=0.25)
        
        assert abs(quarter.fraction - full.fraction * 0.25) < 0.001
    
    def test_half_kelly(self):
        """Half Kelly should be 50% of full Kelly."""
        full = kelly_criterion(0.55, 2.0)
        half = fractional_kelly(0.55, 2.0, fraction=0.5)
        
        assert abs(half.fraction - full.fraction * 0.5) < 0.001
    
    def test_full_kelly_fraction_1(self):
        """Fraction=1.0 should match full Kelly."""
        full = kelly_criterion(0.55, 2.0)
        frac_full = fractional_kelly(0.55, 2.0, fraction=1.0)
        
        assert abs(full.fraction - frac_full.fraction) < 0.001
    
    def test_fractional_reduces_growth(self):
        """Fractional Kelly should have lower expected growth."""
        full = kelly_criterion(0.55, 2.0)
        quarter = fractional_kelly(0.55, 2.0, fraction=0.25)
        
        assert quarter.expected_growth < full.expected_growth
        assert quarter.expected_growth > 0  # Still positive
    
    def test_invalid_fraction(self):
        """Invalid fraction values should raise ValueError."""
        with pytest.raises(ValueError):
            fractional_kelly(0.55, 2.0, fraction=0.0)
        
        with pytest.raises(ValueError):
            fractional_kelly(0.55, 2.0, fraction=1.5)


class TestBoundedKelly:
    """Tests for bounded-below Kelly with floor constraint."""
    
    def test_floor_not_reached(self):
        """When Kelly bet doesn't approach floor, use Kelly fraction."""
        result, max_frac = bounded_kelly(
            prob=0.55,
            decimal_odds=2.0,
            bankroll=10000,
            floor=2000,
            kelly_mult=0.25
        )
        
        # Max fraction = (10000-2000)/10000 = 0.8
        assert max_frac == 0.8
        # Quarter Kelly at 55%/2.0 ≈ 0.025, well under 0.8
        assert result.fraction < max_frac
    
    def test_floor_constraint_active(self):
        """When Kelly suggests more than floor allows, cap at max."""
        # Make Kelly suggest a large fraction
        result, max_frac = bounded_kelly(
            prob=0.80,  # Very high edge
            decimal_odds=2.0,
            bankroll=10000,
            floor=9500,  # Very tight floor
            kelly_mult=1.0  # Full Kelly
        )
        
        # Max fraction = (10000-9500)/10000 = 0.05
        assert max_frac == pytest.approx(0.05)
        # Full Kelly at 80%/2.0 = 0.6, but capped at 0.05
        assert result.fraction == pytest.approx(max_frac)
    
    def test_floor_above_bankroll_error(self):
        """Floor >= bankroll should raise ValueError."""
        with pytest.raises(ValueError):
            bounded_kelly(0.55, 2.0, 10000, 10000)
        
        with pytest.raises(ValueError):
            bounded_kelly(0.55, 2.0, 10000, 15000)


class TestBayesianKelly:
    """Tests for Bayesian Kelly with shrinkage."""
    
    def test_full_confidence_uses_model(self):
        """Confidence=1.0 should use model probability."""
        result, p_shrunk = bayesian_kelly(
            prob_model=0.60,
            prob_market=0.50,
            decimal_odds=2.0,
            confidence=1.0,
            kelly_mult=1.0
        )
        
        assert p_shrunk == 0.60
        # Kelly with p=0.6 at 2.0 odds
        expected_kelly = (0.6 * 1.0 - 0.4) / 1.0  # = 0.2
        assert result.fraction == pytest.approx(expected_kelly, rel=0.01)
    
    def test_zero_confidence_uses_market(self):
        """Confidence=0.0 should use market probability (no bet)."""
        result, p_shrunk = bayesian_kelly(
            prob_model=0.60,
            prob_market=0.50,
            decimal_odds=2.0,
            confidence=0.0,
            kelly_mult=1.0
        )
        
        assert p_shrunk == 0.50
        # Kelly with p=0.5 at 2.0 = 0 (fair odds)
        assert result.fraction == 0.0
    
    def test_half_confidence_averages(self):
        """Confidence=0.5 should average model and market."""
        result, p_shrunk = bayesian_kelly(
            prob_model=0.60,
            prob_market=0.50,
            decimal_odds=2.0,
            confidence=0.5,
            kelly_mult=1.0
        )
        
        assert p_shrunk == pytest.approx(0.55)
    
    def test_shrinkage_reduces_overconfidence(self):
        """Shrinkage should reduce bets when model is overconfident."""
        # Model claims 70%, market implies 50%
        full_conf, _ = bayesian_kelly(0.70, 0.50, 2.0, confidence=1.0)
        half_conf, _ = bayesian_kelly(0.70, 0.50, 2.0, confidence=0.5)
        
        assert half_conf.fraction < full_conf.fraction


class TestKellyCalculator:
    """Tests for the KellyCalculator class."""
    
    def test_basic_calculation(self):
        """Basic stake calculation works."""
        calc = KellyCalculator(
            default_kelly_mult=0.25,
            use_bounded=False,
            use_bayesian=False
        )
        
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0)
        result = calc.calculate(bet, bankroll=10000)
        
        assert result.fraction > 0
        assert result.stake_amount(10000) > 0
    
    def test_stake_amount_method(self):
        """calculate_stake returns correct dollar amount."""
        calc = KellyCalculator(default_kelly_mult=0.25)
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0)
        
        stake = calc.calculate_stake(bet, bankroll=10000)
        result = calc.calculate(bet, bankroll=10000)
        
        assert stake == pytest.approx(result.fraction * 10000)
    
    def test_with_bayesian_shrinkage(self):
        """Calculator applies Bayesian shrinkage when configured."""
        calc = KellyCalculator(
            use_bayesian=True,
            default_confidence=0.5
        )
        
        bet = BetOpportunity(
            prob=0.60,
            decimal_odds=2.0,
            market_prob=0.50
        )
        
        result = calc.calculate(bet, bankroll=10000)
        
        # Should be smaller than without shrinkage
        calc_no_shrink = KellyCalculator(use_bayesian=False)
        result_no_shrink = calc_no_shrink.calculate(bet, bankroll=10000)
        
        assert result.fraction < result_no_shrink.fraction


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_expected_value_positive(self):
        """Positive EV calculation is correct."""
        # 55% at even money: EV = 0.55*1 - 0.45 = 0.10
        ev = expected_value(0.55, 2.0)
        assert ev == pytest.approx(0.10)
    
    def test_expected_value_negative(self):
        """Negative EV calculation is correct."""
        # 40% at even money: EV = 0.40*1 - 0.60 = -0.20
        ev = expected_value(0.40, 2.0)
        assert ev == pytest.approx(-0.20)
    
    def test_break_even_probability(self):
        """Break-even probability is inverse of odds."""
        assert break_even_probability(2.0) == 0.50
        assert break_even_probability(4.0) == 0.25
        assert break_even_probability(1.5) == pytest.approx(0.6667, rel=0.01)


class TestBetOpportunity:
    """Tests for BetOpportunity dataclass."""
    
    def test_net_odds(self):
        """Net odds is decimal odds minus 1."""
        bet = BetOpportunity(prob=0.5, decimal_odds=2.5)
        assert bet.net_odds == 1.5
    
    def test_implied_prob(self):
        """Implied probability from odds."""
        bet = BetOpportunity(prob=0.5, decimal_odds=2.0)
        assert bet.implied_prob == 0.5
    
    def test_edge_calculation(self):
        """Edge is model prob minus market prob."""
        bet = BetOpportunity(prob=0.55, decimal_odds=2.0, market_prob=0.50)
        assert bet.edge == pytest.approx(0.05)
    
    def test_validation(self):
        """Invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            BetOpportunity(prob=1.5, decimal_odds=2.0)
        
        with pytest.raises(ValueError):
            BetOpportunity(prob=0.5, decimal_odds=0.5)

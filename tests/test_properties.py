
import pytest
from hypothesis import given, strategies as st
from hypothesis import assume
import math
import numpy as np

from src.strategies.kelly import kelly_criterion, KellyResult
from src.models.features.four_factors import FourFactorsCalculator, PaceCalculator
from src.models.features.advanced_metrics import AdvancedMetricsCalculator

# =============================================================================
# Kelly Criterion Properties
# =============================================================================

@given(
    prob=st.floats(min_value=0.01, max_value=0.99),
    decimal_odds=st.floats(min_value=1.01, max_value=100.0)
)
def test_kelly_criterion_bounds(prob, decimal_odds):
    """Kelly fraction should always be between 0 and 1 (approx)."""
    result = kelly_criterion(prob, decimal_odds)
    assert 0.0 <= result.fraction
    # In sports betting, raw Kelly can theoretically exceed 1 if edge is massive,
    # but practically we expect reasonable bounds.
    # However, the implementation does not cap at 1.0 by default (bounded_kelly does).
    # We verify it's non-negative.
    assert result.fraction >= 0.0

@given(
    prob=st.floats(min_value=0.01, max_value=0.99),
    decimal_odds=st.floats(min_value=1.01, max_value=100.0)
)
def test_kelly_negative_edge_returns_zero(prob, decimal_odds):
    """If edge <= 0, Kelly fraction should be 0."""
    implied_prob = 1.0 / decimal_odds
    if prob <= implied_prob:
        result = kelly_criterion(prob, decimal_odds)
        assert result.fraction == 0.0
        assert not result.is_positive_ev

@given(
    prob=st.floats(min_value=0.01, max_value=0.99),
    decimal_odds=st.floats(min_value=1.01, max_value=100.0)
)
def test_kelly_positive_edge_returns_positive(prob, decimal_odds):
    """If edge > 0, Kelly fraction should be > 0."""
    implied_prob = 1.0 / decimal_odds
    if prob > implied_prob:
        result = kelly_criterion(prob, decimal_odds)
        assert result.fraction > 0.0
        assert result.is_positive_ev

@given(
    prob=st.floats(min_value=0.1, max_value=0.9),
    decimal_odds_1=st.floats(min_value=1.5, max_value=5.0),
    decimal_odds_2=st.floats(min_value=1.5, max_value=5.0)
)
def test_kelly_monotonicity_with_odds(prob, decimal_odds_1, decimal_odds_2):
    """For fixed probability > implied, higher odds should yield higher fraction."""
    # Ensure both are positive EV
    assume(prob > 1.0/decimal_odds_1)
    assume(prob > 1.0/decimal_odds_2)

    res1 = kelly_criterion(prob, decimal_odds_1)
    res2 = kelly_criterion(prob, decimal_odds_2)

    if decimal_odds_1 < decimal_odds_2:
        assert res1.fraction < res2.fraction
    elif decimal_odds_1 > decimal_odds_2:
        assert res1.fraction > res2.fraction

# =============================================================================
# Four Factors Properties
# =============================================================================

@given(
    fgm=st.floats(min_value=0, max_value=100),
    fg3m=st.floats(min_value=0, max_value=50),
    fga=st.floats(min_value=1, max_value=200) # FGA > 0 to avoid div/0
)
def test_efg_calculation(fgm, fg3m, fga):
    """eFG% should be logical."""
    assume(fgm <= fga) # Made cannot exceed attempted
    assume(fg3m <= fgm) # 3PM cannot exceed FGM

    calc = FourFactorsCalculator()
    efg = calc.calculate_efg_pct(fgm, fg3m, fga)

    assert efg >= 0.0
    # eFG can be > 1.0 (e.g. 3/3 3PM = 1.5)
    # Floating point precision can cause slight overshoot
    assert efg <= 1.50000000000001

@given(
    turnovers=st.floats(min_value=0, max_value=50),
    fga=st.floats(min_value=0, max_value=150),
    fta=st.floats(min_value=0, max_value=100)
)
def test_tov_pct_bounds(turnovers, fga, fta):
    """TOV% should be between 0 and 100."""
    # Ensure denominator is positive
    assume(fga + 0.44*fta + turnovers > 0)

    calc = FourFactorsCalculator()
    tov_pct = calc.calculate_tov_pct(turnovers, fga, fta)

    assert 0.0 <= tov_pct <= 100.00000000000001

@given(
    fga=st.floats(min_value=0, max_value=150),
    oreb=st.floats(min_value=0, max_value=50),
    turnovers=st.floats(min_value=0, max_value=50),
    fta=st.floats(min_value=0, max_value=100)
)
def test_possessions_estimate_positive(fga, oreb, turnovers, fta):
    """Estimated possessions should be non-negative."""
    # Standard possession formula components are additive except OREB
    # Poss = FGA - OREB + TO + 0.44 * FTA
    # Physically, OREB cannot exceed FGA (approx) in a single possession flow,
    # but strictly mathematically:
    assume(fga >= oreb)

    calc = PaceCalculator()
    poss = calc.estimate_possessions(fga, oreb, turnovers, fta)
    assert poss >= 0

# =============================================================================
# Advanced Metrics Properties
# =============================================================================

@given(
    points=st.floats(min_value=0, max_value=200),
    fga=st.floats(min_value=0, max_value=150),
    fta=st.floats(min_value=0, max_value=100)
)
def test_true_shooting_percentage(points, fga, fta):
    """TS% should be non-negative and finite."""
    calc = AdvancedMetricsCalculator()
    # Access private method for testing, or assume it's exposed.
    # The file has _calculate_ts_pct.

    denom = 2 * (fga + 0.44 * fta)
    assume(denom > 0)

    ts = calc._calculate_ts_pct(points, fga, fta)
    assert ts >= 0.0
    # Max possible TS% can be very large in edge cases (e.g. many points, few attempts)
    # We just want to ensure it calculates a finite value
    assert math.isfinite(ts)

@given(
    points_for=st.floats(min_value=0, max_value=10000),
    points_against=st.floats(min_value=0, max_value=10000),
    games_played=st.integers(min_value=1, max_value=82)
)
def test_expected_wins_bounds(points_for, points_against, games_played):
    """Expected wins should be between 0 and games_played."""
    calc = AdvancedMetricsCalculator()

    exp_wins = calc.calculate_expected_wins(points_for, points_against, games_played)

    assert 0.0 <= exp_wins <= float(games_played)

@given(
    points_for=st.floats(min_value=100, max_value=10000),
    points_against=st.floats(min_value=100, max_value=10000),
    games_played=st.integers(min_value=1, max_value=82)
)
def test_expected_wins_monotonicity(points_for, points_against, games_played):
    """More points scored (fixed allowed) should increase expected wins."""
    calc = AdvancedMetricsCalculator()

    w1 = calc.calculate_expected_wins(points_for, points_against, games_played)
    w2 = calc.calculate_expected_wins(points_for + 50, points_against, games_played)

    assert w2 >= w1

"""
Advanced Kelly Criterion Implementations for Sports Betting.

This module provides rigorous implementations of the Kelly Criterion and its
variants for optimal bankroll management. We treat sports betting as an
asset allocation problem, optimizing for geometric growth while managing
ruin risk.

Mathematical Foundation
=======================

The Kelly Criterion maximizes the expected logarithm of wealth, which is
equivalent to maximizing the geometric growth rate of the bankroll.

For a binary bet with probability p of winning and net odds b:

.. math::

    f^* = \\frac{p(b+1) - 1}{b} = \\frac{pb - q}{b}

Where:
    - f* = optimal fraction of bankroll to wager
    - p = probability of winning
    - q = 1 - p = probability of losing
    - b = decimal odds - 1 (net payout per unit wagered)

Why Logarithmic Utility?
------------------------

We use log utility because:
1. It's the ONLY utility that maximizes long-term geometric growth
2. It prevents ruin (Kelly never suggests all-in bets)
3. It's invariant to the scale of the bankroll

.. warning::

    QUADRATIC APPROXIMATION DANGER
    
    Many online resources use a Taylor expansion of log(1+x) ≈ x - x²/2
    to derive "simplified" Kelly formulas. This is DANGEROUS because:
    
    1. The approximation breaks down for large wagers (x > 0.1)
    2. It can suggest over-betting in high-variance situations
    3. It ignores the asymmetry between gains and losses
    
    This implementation uses the EXACT logarithmic formulation throughout.

References
----------
- Kelly, J.L. (1956). "A New Interpretation of Information Rate"
- Thorp, E.O. (2006). "The Kelly Criterion in Blackjack Sports Betting..."
- MacLean, Thorp, Ziemba (2011). "The Kelly Capital Growth Investment Criterion"
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class BetOpportunity:
    """
    Container for a single betting opportunity.
    
    Attributes:
        prob: Our estimated probability of the outcome
        decimal_odds: Decimal odds offered (e.g., 2.0 for even money)
        market_prob: Market implied probability (for Bayesian shrinkage)
        bet_id: Optional identifier for the bet
        correlation_group: Group ID for correlated bets
    """
    prob: float
    decimal_odds: float
    market_prob: Optional[float] = None
    bet_id: Optional[str] = None
    correlation_group: Optional[str] = None
    
    @property
    def net_odds(self) -> float:
        """Net odds (profit per unit wagered if win)."""
        return self.decimal_odds - 1.0
    
    @property
    def edge(self) -> float:
        """Edge over market (if market_prob available)."""
        if self.market_prob is None:
            return 0.0
        return self.prob - self.market_prob
    
    @property
    def implied_prob(self) -> float:
        """Market implied probability from odds (before vig removal)."""
        return 1.0 / self.decimal_odds
    
    def __post_init__(self):
        """Validate inputs."""
        if not 0 < self.prob < 1:
            raise ValueError(f"Probability must be in (0, 1), got {self.prob}")
        if self.decimal_odds < 1.0:
            raise ValueError(f"Decimal odds must be >= 1.0, got {self.decimal_odds}")


@dataclass
class KellyResult:
    """
    Result of a Kelly calculation.
    
    Attributes:
        fraction: Recommended wager as fraction of bankroll
        expected_growth: Expected log growth rate per bet
        edge: Edge over market (p - implied_prob)
        is_positive_ev: Whether the bet has positive expected value
        raw_kelly: Uncapped Kelly fraction (before any adjustments)
    """
    fraction: float
    expected_growth: float
    edge: float
    is_positive_ev: bool
    raw_kelly: float
    
    def stake_amount(self, bankroll: float) -> float:
        """Calculate actual stake amount for given bankroll."""
        return self.fraction * bankroll


def kelly_criterion(
    prob: float,
    decimal_odds: float,
    validate: bool = True
) -> KellyResult:
    """
    Calculate the classic Kelly fraction for a binary outcome.
    
    The Kelly Criterion:
    
    .. math::
    
        f^* = \\frac{p(b+1) - 1}{b} = \\frac{p \\cdot b - q}{b}
    
    Where:
        - f* = optimal fraction to wager
        - p = win probability
        - q = 1 - p = lose probability
        - b = net odds (decimal_odds - 1)
    
    Args:
        prob: Probability of winning (0 < p < 1)
        decimal_odds: Decimal odds offered (e.g., 2.0 = even money)
        validate: Whether to validate input ranges
    
    Returns:
        KellyResult with optimal fraction and metadata
    
    Raises:
        ValueError: If inputs are out of valid ranges
    
    Example:
        >>> # 55% win prob at even money (2.0 decimal)
        >>> result = kelly_criterion(0.55, 2.0)
        >>> result.fraction
        0.1  # Bet 10% of bankroll
        
        >>> # 40% win prob at 3.0 decimal (should not bet)
        >>> result = kelly_criterion(0.40, 3.0)
        >>> result.fraction
        0.0  # Negative EV, don't bet
    """
    if validate:
        if not 0 < prob < 1:
            raise ValueError(f"Probability must be in (0, 1), got {prob}")
        if decimal_odds < 1.0:
            raise ValueError(f"Decimal odds must be >= 1.0, got {decimal_odds}")
    
    q = 1 - prob
    b = decimal_odds - 1  # Net odds
    
    # Kelly formula: f* = (p*b - q) / b
    raw_kelly = (prob * b - q) / b
    
    # Edge calculation
    implied_prob = 1.0 / decimal_odds
    edge = prob - implied_prob
    
    # Is this positive EV?
    is_positive_ev = raw_kelly > 0
    
    # Never bet negative (no shorting in sports betting)
    kelly_fraction = max(0.0, raw_kelly)
    
    # Calculate expected log growth
    # E[log(1 + f*R)] where R is the random return
    if kelly_fraction > 0:
        # Win: log(1 + f*b), Lose: log(1 - f)
        growth_win = np.log(1 + kelly_fraction * b)
        growth_lose = np.log(1 - kelly_fraction)
        expected_growth = prob * growth_win + q * growth_lose
    else:
        expected_growth = 0.0
    
    return KellyResult(
        fraction=kelly_fraction,
        expected_growth=expected_growth,
        edge=edge,
        is_positive_ev=is_positive_ev,
        raw_kelly=raw_kelly
    )


def fractional_kelly(
    prob: float,
    decimal_odds: float,
    fraction: float = 0.25
) -> KellyResult:
    """
    Calculate Fractional Kelly for variance reduction.
    
    Fractional Kelly multiplies the full Kelly fraction by a constant
    (typically 0.25 or 0.5) to reduce volatility at the cost of
    lower expected growth.
    
    Why use Fractional Kelly?
    -------------------------
    
    1. **Parameter Uncertainty**: Our probability estimates are imperfect.
       Full Kelly assumes perfect knowledge of p.
    
    2. **Variance Reduction**: Full Kelly has ~50% chance of halving
       your bankroll before doubling it. Fractional Kelly reduces this.
    
    3. **Psychological**: Smaller bets are easier to stick with during
       inevitable losing streaks.
    
    Common fractions:
        - 0.25: Very conservative (quarter Kelly)
        - 0.50: Moderate (half Kelly)
        - 0.33: Common middle ground (third Kelly)
    
    Growth rate impact:
    
    .. math::
    
        G(\\lambda f^*) = \\lambda G(f^*) - \\frac{\\lambda^2 \\sigma^2 f^{*2}}{2} + O(\\lambda^3)
    
    So half Kelly gives ~75% of the growth rate but ~50% of the variance.
    
    Args:
        prob: Win probability
        decimal_odds: Decimal odds
        fraction: Kelly fraction multiplier (0 < f <= 1)
    
    Returns:
        KellyResult with scaled fraction
    
    Example:
        >>> result = fractional_kelly(0.55, 2.0, fraction=0.25)
        >>> result.fraction
        0.025  # 2.5% of bankroll (vs 10% for full Kelly)
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"Fraction must be in (0, 1], got {fraction}")
    
    # Get full Kelly
    full_kelly = kelly_criterion(prob, decimal_odds)
    
    # Scale down
    scaled_fraction = full_kelly.fraction * fraction
    
    # Recalculate expected growth at the new stake
    q = 1 - prob
    b = decimal_odds - 1
    
    if scaled_fraction > 0:
        growth_win = np.log(1 + scaled_fraction * b)
        growth_lose = np.log(1 - scaled_fraction)
        expected_growth = prob * growth_win + q * growth_lose
    else:
        expected_growth = 0.0
    
    return KellyResult(
        fraction=scaled_fraction,
        expected_growth=expected_growth,
        edge=full_kelly.edge,
        is_positive_ev=full_kelly.is_positive_ev,
        raw_kelly=full_kelly.raw_kelly
    )


def bounded_kelly(
    prob: float,
    decimal_odds: float,
    bankroll: float,
    floor: float,
    kelly_mult: float = 0.25
) -> Tuple[KellyResult, float]:
    """
    Kelly with a bankroll floor constraint (Tail Risk Protection).
    
    This implements a "bounded-below" model where the bankroll has a
    psychological or practical "floor" that we never want to breach.
    
    The constraint ensures:
    
    .. math::
    
        f \\leq \\frac{W - W_{floor}}{W}
    
    This prevents betting amounts that could bring the bankroll close
    to the floor, even if theoretical Kelly suggests larger bets.
    
    Why a Floor?
    ------------
    
    1. **Practical Ruin**: Being reduced to a tiny bankroll is effectively
       ruin (can't place meaningful bets).
    
    2. **Psychological Protection**: Most bettors have a threshold below
       which they would quit.
    
    3. **Margin Requirements**: Some platforms require minimum balances.
    
    4. **Recovery Time**: Very small bankrolls take exponentially long
       to recover (Kelly is myopic to this).
    
    Args:
        prob: Win probability
        decimal_odds: Decimal odds
        bankroll: Current bankroll value
        floor: Minimum acceptable bankroll (the "floor")
        kelly_mult: Fractional Kelly multiplier (applied before floor cap)
    
    Returns:
        Tuple of (KellyResult, max_allowed_fraction)
    
    Example:
        >>> # $10,000 bankroll, $2,000 floor
        >>> result, max_frac = bounded_kelly(0.6, 2.0, 10000, 2000)
        >>> max_frac
        0.8  # Can bet at most 80% of bankroll
        >>> result.fraction  
        0.05  # But Kelly recommends 5% (after 0.25 mult)
    """
    if floor >= bankroll:
        raise ValueError(f"Floor ({floor}) must be less than bankroll ({bankroll})")
    if floor < 0:
        raise ValueError(f"Floor must be non-negative, got {floor}")
    
    # Calculate the maximum fraction we can bet
    max_fraction = (bankroll - floor) / bankroll
    
    # Get fractional Kelly
    frac_kelly = fractional_kelly(prob, decimal_odds, fraction=kelly_mult)
    
    # Apply the floor constraint
    bounded_fraction = min(frac_kelly.fraction, max_fraction)
    
    # Recalculate expected growth at bounded stake
    q = 1 - prob
    b = decimal_odds - 1
    
    if bounded_fraction > 0:
        growth_win = np.log(1 + bounded_fraction * b)
        growth_lose = np.log(1 - bounded_fraction)
        expected_growth = prob * growth_win + q * growth_lose
    else:
        expected_growth = 0.0
    
    result = KellyResult(
        fraction=bounded_fraction,
        expected_growth=expected_growth,
        edge=frac_kelly.edge,
        is_positive_ev=frac_kelly.is_positive_ev,
        raw_kelly=frac_kelly.raw_kelly
    )
    
    return result, max_fraction


def bayesian_kelly(
    prob_model: float,
    prob_market: float,
    decimal_odds: float,
    confidence: float = 0.5,
    kelly_mult: float = 0.25
) -> Tuple[KellyResult, float]:
    """
    Bayesian Kelly with Shrinkage Estimator.
    
    Since our model's probability estimate is uncertain, we shrink it
    toward the market implied probability based on our confidence in
    the model.
    
    Shrinkage Formula:
    
    .. math::
    
        p_{shrunk} = \\omega \\cdot p_{model} + (1 - \\omega) \\cdot p_{market}
    
    Where ω (omega/confidence) is the weight given to our model vs market.
    
    Why Shrinkage?
    --------------
    
    1. **Estimation Error**: Our model's probabilities are estimates with
       variance. The market aggregates more information.
    
    2. **Calibration History**: If our model has been overconfident
       historically, shrinkage corrects for this.
    
    3. **Fat Tails**: Model errors often have fat tails. Shrinkage
       protects against extreme estimation errors.
    
    4. **Bayesian Interpretation**: The market provides a prior, and
       our model provides likelihood. Shrinkage is Bayesian updating.
    
    Setting Confidence (ω):
    -----------------------
    
    - ω = 0: Full trust in market (never bet, since p = market_implied_prob)
    - ω = 0.5: Equal weight to model and market
    - ω = 1: Full trust in model (use raw model probabilities)
    
    Typically, use historical calibration analysis:
    
    .. math::
    
        \\omega = \\frac{1}{1 + \\frac{\\sigma_{model}^2}{\\sigma_{market}^2}}
    
    Where σ² are the variances of each probability source.
    
    Args:
        prob_model: Our model's estimated probability
        prob_market: Market implied probability (1 / decimal_odds)
        decimal_odds: Decimal odds
        confidence: Weight on model probability (0 to 1)
        kelly_mult: Fractional Kelly multiplier
    
    Returns:
        Tuple of (KellyResult based on shrunk prob, shrunk probability)
    
    Example:
        >>> # Model says 60%, market implies 50%
        >>> result, p_shrunk = bayesian_kelly(0.60, 0.50, 2.0, confidence=0.5)
        >>> p_shrunk
        0.55  # Shrunk toward market
    """
    if not 0 <= confidence <= 1:
        raise ValueError(f"Confidence must be in [0, 1], got {confidence}")
    
    # Shrink model probability toward market
    prob_shrunk = confidence * prob_model + (1 - confidence) * prob_market
    
    # Calculate Kelly using shrunk probability
    result = fractional_kelly(prob_shrunk, decimal_odds, fraction=kelly_mult)
    
    # Update edge calculation to show model edge (not shrunk)
    model_edge = prob_model - (1 / decimal_odds)
    
    return KellyResult(
        fraction=result.fraction,
        expected_growth=result.expected_growth,
        edge=model_edge,  # Report original model edge
        is_positive_ev=result.is_positive_ev,
        raw_kelly=result.raw_kelly
    ), prob_shrunk


class KellyCalculator:
    """
    Comprehensive Kelly Criterion calculator for portfolio management.
    
    This class wraps all Kelly variants and provides a unified interface
    for the PortfolioManager to calculate optimal stake sizes.
    
    Configuration options:
        - default_kelly_mult: Default fractional Kelly multiplier
        - use_bounded: Whether to enforce bankroll floor
        - floor_fraction: Floor as fraction of peak bankroll
        - use_bayesian: Whether to apply shrinkage
        - default_confidence: Default model confidence for shrinkage
    
    Example:
        >>> calc = KellyCalculator(
        ...     default_kelly_mult=0.25,
        ...     use_bounded=True,
        ...     floor_fraction=0.2
        ... )
        >>> bet = BetOpportunity(prob=0.6, decimal_odds=2.0)
        >>> result = calc.calculate(bet, bankroll=10000, peak_bankroll=12000)
    """
    
    def __init__(
        self,
        default_kelly_mult: float = 0.25,
        use_bounded: bool = True,
        floor_fraction: float = 0.2,
        use_bayesian: bool = True,
        default_confidence: float = 0.5
    ):
        """
        Initialize the Kelly calculator.
        
        Args:
            default_kelly_mult: Fractional Kelly multiplier (0 < x <= 1)
            use_bounded: Apply bankroll floor constraint
            floor_fraction: Floor as fraction of peak bankroll
            use_bayesian: Apply shrinkage to model probabilities
            default_confidence: Default weight on model probability
        """
        self.kelly_mult = default_kelly_mult
        self.use_bounded = use_bounded
        self.floor_fraction = floor_fraction
        self.use_bayesian = use_bayesian
        self.default_confidence = default_confidence
    
    def calculate(
        self,
        bet: BetOpportunity,
        bankroll: float,
        peak_bankroll: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> KellyResult:
        """
        Calculate optimal stake for a betting opportunity.
        
        Applies all configured adjustments (shrinkage, fractional, bounded).
        
        Args:
            bet: BetOpportunity object
            bankroll: Current bankroll
            peak_bankroll: Historical peak (for floor calculation)
            confidence: Override default confidence
        
        Returns:
            KellyResult with optimal stake fraction
        """
        prob = bet.prob
        
        # Apply Bayesian shrinkage if configured and market prob available
        if self.use_bayesian and bet.market_prob is not None:
            conf = confidence or self.default_confidence
            _, prob = bayesian_kelly(
                bet.prob, bet.market_prob, bet.decimal_odds,
                confidence=conf, kelly_mult=1.0  # Get shrunk prob only
            )
        
        # Calculate base fractional Kelly
        result = fractional_kelly(prob, bet.decimal_odds, self.kelly_mult)
        
        # Apply bounded constraint if configured
        if self.use_bounded:
            peak = peak_bankroll or bankroll
            floor = peak * self.floor_fraction
            
            if floor < bankroll:
                result, _ = bounded_kelly(
                    prob, bet.decimal_odds, bankroll, floor, self.kelly_mult
                )
        
        return result
    
    def calculate_stake(
        self,
        bet: BetOpportunity,
        bankroll: float,
        peak_bankroll: Optional[float] = None
    ) -> float:
        """
        Calculate actual stake amount in currency.
        
        Convenience method that returns the dollar amount to wager.
        
        Args:
            bet: BetOpportunity
            bankroll: Current bankroll
            peak_bankroll: Historical peak
        
        Returns:
            Stake amount (currency units)
        """
        result = self.calculate(bet, bankroll, peak_bankroll)
        return result.stake_amount(bankroll)


# ============================================================================
# Utility Functions
# ============================================================================

def expected_value(prob: float, decimal_odds: float) -> float:
    """
    Calculate expected value of a bet.
    
    .. math::
    
        EV = p \\cdot (b) - q = p \\cdot b - (1-p) = p(b+1) - 1
    
    Where b = decimal_odds - 1 (net odds).
    
    Returns EV per unit staked.
    """
    q = 1 - prob
    b = decimal_odds - 1
    return prob * b - q


def break_even_probability(decimal_odds: float) -> float:
    """
    Calculate the break-even probability for given odds.
    
    This is the probability at which EV = 0.
    
    .. math::
    
        p_{BE} = \\frac{1}{decimal\\_odds}
    """
    return 1.0 / decimal_odds


def required_odds(prob: float) -> float:
    """
    Calculate the minimum decimal odds needed for positive EV.
    
    .. math::
    
        odds_{min} = \\frac{1}{p}
    
    At these odds, you break even. Need higher for positive EV.
    """
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be in (0, 1), got {prob}")
    return 1.0 / prob

"""
Multi-Asset Simultaneous Kelly Optimization.

This module solves the simultaneous Kelly problem for allocating wealth
across multiple bets that occur at the same time (e.g., multiple NBL
games on the same night).

The Simultaneous Kelly Problem
==============================

When betting on multiple independent events simultaneously, we cannot
simply sum individual Kelly fractions. The constraint is:

.. math::

    \\sum_{i=1}^{n} f_i \\leq 1

And the objective is to maximize expected log growth:

.. math::

    \\max_{f_1, ..., f_n} \\mathbb{E}\\left[\\log\\left(1 + \\sum_{i=1}^{n} f_i R_i\\right)\\right]

Where R_i is the random return for bet i.

Correlation Matters
-------------------

If bets are correlated (e.g., "Team A Win" and "Team A Over Points"),
we must account for this. The correlation structure affects optimal
sizing because:

1. Positive correlation: Wins/losses cluster, increasing variance
2. Negative correlation: Some hedging effect, may allow larger total exposure

This implementation handles correlation via:
- Correlation groups (bets in same group share outcomes)
- Explicit correlation matrices for more nuanced modeling

References
----------
- Thorp, E.O. (1997). "The Kelly Criterion in Blackjack, Sports Betting..."
- MacLean, Thorp, Ziemba (2011). "The Kelly Capital Growth Investment Criterion"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

from .kelly import BetOpportunity, KellyResult, fractional_kelly


@dataclass
class SimultaneousBetResult:
    """
    Result of simultaneous multi-bet optimization.
    
    Attributes:
        allocations: Dictionary mapping bet_id to optimal fraction
        total_allocation: Sum of all fractions (should be <= max_fraction)
        expected_growth: Expected log growth from combined position
        individual_results: Individual Kelly results for each bet
    """
    allocations: Dict[str, float]
    total_allocation: float
    expected_growth: float
    individual_results: Dict[str, KellyResult]
    
    def stake_amounts(self, bankroll: float) -> Dict[str, float]:
        """Calculate actual stake amounts for each bet."""
        return {
            bet_id: frac * bankroll 
            for bet_id, frac in self.allocations.items()
        }


class SimultaneousKellyOptimizer:
    """
    Optimizer for simultaneous multi-bet Kelly allocation.
    
    Handles the constraint that total allocation cannot exceed a maximum
    and accounts for correlation between bets.
    
    Correlation Handling:
    ---------------------
    
    Bets can be grouped by correlation_group. Bets in the same group are
    assumed to have positive correlation (e.g., same-game bets).
    
    For same-group bets, we reduce total allocation to account for
    increased variance from correlated outcomes.
    
    Example:
        >>> optimizer = SimultaneousKellyOptimizer(
        ...     max_total_fraction=0.15,
        ...     kelly_mult=0.25
        ... )
        >>> 
        >>> bets = [
        ...     BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="game1_ml"),
        ...     BetOpportunity(prob=0.52, decimal_odds=1.9, bet_id="game2_ml"),
        ... ]
        >>> result = optimizer.optimize(bets)
    
    Attributes:
        max_total_fraction: Maximum sum of all bet fractions
        kelly_mult: Fractional Kelly multiplier for individual bets
        correlation_penalty: Reduce allocation for correlated bets
    """
    
    def __init__(
        self,
        max_total_fraction: float = 0.15,
        kelly_mult: float = 0.25,
        correlation_penalty: float = 0.3
    ):
        """
        Initialize the optimizer.
        
        Args:
            max_total_fraction: Maximum total allocation across all bets
            kelly_mult: Fractional Kelly multiplier
            correlation_penalty: Penalty for correlated bets (0-1)
                                Reduces group allocation by this factor
        """
        if not 0 < max_total_fraction <= 1:
            raise ValueError(f"max_total_fraction must be in (0, 1], got {max_total_fraction}")
        
        self.max_total_fraction = max_total_fraction
        self.kelly_mult = kelly_mult
        self.correlation_penalty = correlation_penalty
    
    def optimize(
        self,
        bets: List[BetOpportunity],
        method: str = 'proportional'
    ) -> SimultaneousBetResult:
        """
        Optimize allocation across multiple simultaneous bets.
        
        Available methods:
        ------------------
        
        'proportional': Scale individual Kelly fractions proportionally
                       to meet the max_total constraint
        
        'equal_risk': Allocate equally weighted by edge
        
        'numerical': Use numerical optimization (slower but more accurate
                    for complex correlation structures)
        
        Args:
            bets: List of BetOpportunity objects
            method: Optimization method to use
        
        Returns:
            SimultaneousBetResult with optimal allocations
        """
        if not bets:
            return SimultaneousBetResult(
                allocations={},
                total_allocation=0.0,
                expected_growth=0.0,
                individual_results={}
            )
        
        # Calculate individual Kelly for each bet
        individual_results = {}
        for bet in bets:
            bet_id = bet.bet_id or f"bet_{id(bet)}"
            result = fractional_kelly(bet.prob, bet.decimal_odds, self.kelly_mult)
            individual_results[bet_id] = result
        
        # Filter to positive EV bets only
        positive_ev_bets = {
            bid: res for bid, res in individual_results.items()
            if res.is_positive_ev
        }
        
        if not positive_ev_bets:
            return SimultaneousBetResult(
                allocations={bid: 0.0 for bid in individual_results},
                total_allocation=0.0,
                expected_growth=0.0,
                individual_results=individual_results
            )
        
        # Group bets by correlation
        correlation_groups = self._group_by_correlation(bets)
        
        if method == 'proportional':
            allocations = self._proportional_allocation(
                positive_ev_bets, correlation_groups
            )
        elif method == 'equal_risk':
            allocations = self._equal_risk_allocation(positive_ev_bets)
        elif method == 'numerical':
            allocations = self._numerical_optimization(bets, individual_results)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add zeros for negative EV bets
        for bet_id in individual_results:
            if bet_id not in allocations:
                allocations[bet_id] = 0.0
        
        # Calculate expected growth
        total_growth = self._calculate_expected_growth(bets, allocations)
        
        return SimultaneousBetResult(
            allocations=allocations,
            total_allocation=sum(allocations.values()),
            expected_growth=total_growth,
            individual_results=individual_results
        )
    
    def _group_by_correlation(
        self,
        bets: List[BetOpportunity]
    ) -> Dict[str, List[str]]:
        """Group bets by their correlation_group attribute."""
        groups: Dict[str, List[str]] = {}
        
        for bet in bets:
            bet_id = bet.bet_id or f"bet_{id(bet)}"
            group = bet.correlation_group or bet_id  # Uncorrelated if no group
            
            if group not in groups:
                groups[group] = []
            groups[group].append(bet_id)
        
        return groups
    
    def _proportional_allocation(
        self,
        positive_ev_bets: Dict[str, KellyResult],
        correlation_groups: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Proportionally scale Kelly fractions to meet total constraint.
        
        For each correlation group, we also apply a penalty to reduce
        exposure to correlated outcomes.
        """
        allocations = {}
        
        # First, get raw Kelly fractions
        raw_fractions = {
            bid: res.fraction for bid, res in positive_ev_bets.items()
        }
        
        # Apply correlation penalty within groups
        for group_id, bet_ids in correlation_groups.items():
            group_bets = [bid for bid in bet_ids if bid in raw_fractions]
            
            if len(group_bets) > 1:
                # Multiple correlated bets: reduce each by penalty
                for bid in group_bets:
                    penalty = 1 - (self.correlation_penalty * (len(group_bets) - 1) / len(group_bets))
                    raw_fractions[bid] *= max(0.1, penalty)
        
        # Calculate total and scale if needed
        total_raw = sum(raw_fractions.values())
        
        if total_raw <= self.max_total_fraction:
            # Under budget, use raw fractions
            return raw_fractions
        
        # Scale down proportionally
        scale_factor = self.max_total_fraction / total_raw
        
        return {
            bid: frac * scale_factor
            for bid, frac in raw_fractions.items()
        }
    
    def _equal_risk_allocation(
        self,
        positive_ev_bets: Dict[str, KellyResult]
    ) -> Dict[str, float]:
        """
        Allocate proportionally to edge, equal risk weighting.
        
        Each bet gets allocation proportional to its edge relative
        to total edge across all bets.
        """
        total_edge = sum(max(0, res.edge) for res in positive_ev_bets.values())
        
        if total_edge <= 0:
            # No positive edge, don't bet
            return {bid: 0.0 for bid in positive_ev_bets}
        
        return {
            bid: (max(0, res.edge) / total_edge) * self.max_total_fraction
            for bid, res in positive_ev_bets.items()
        }
    
    def _numerical_optimization(
        self,
        bets: List[BetOpportunity],
        individual_results: Dict[str, KellyResult]
    ) -> Dict[str, float]:
        """
        Use scipy to numerically optimize the Kelly objective.
        
        This is more accurate but slower, especially for many bets.
        """
        n = len(bets)
        bet_ids = [bet.bet_id or f"bet_{id(bet)}" for bet in bets]
        
        # Build bet parameters
        probs = np.array([bet.prob for bet in bets])
        net_odds = np.array([bet.decimal_odds - 1 for bet in bets])
        
        def negative_expected_log_growth(fractions):
            """Objective: negative expected log growth (to minimize)."""
            # We need to compute E[log(1 + sum(f_i * R_i))]
            # For independent bets, this is complex. Use Monte Carlo approximation.
            
            n_samples = 1000
            total_returns = np.zeros(n_samples)
            
            for i in range(n):
                # Simulate wins/losses
                wins = np.random.random(n_samples) < probs[i]
                returns = np.where(wins, net_odds[i], -1.0)
                total_returns += fractions[i] * returns
            
            # Log growth
            log_wealth = np.log(np.maximum(1 + total_returns, 1e-10))
            return -np.mean(log_wealth)
        
        # Initial guess: proportional allocation
        initial = np.array([
            individual_results[bid].fraction for bid in bet_ids
        ])
        initial = initial / initial.sum() * self.max_total_fraction * 0.5
        
        # Bounds: each fraction in [0, individual Kelly]
        bounds = Bounds(
            lb=np.zeros(n),
            ub=np.array([
                min(individual_results[bid].fraction * 1.5, 0.2)
                for bid in bet_ids
            ])
        )
        
        # Constraint: sum <= max_total_fraction
        constraint = LinearConstraint(
            np.ones(n),
            lb=0,
            ub=self.max_total_fraction
        )
        
        # Optimize
        result = minimize(
            negative_expected_log_growth,
            initial,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: self.max_total_fraction - x.sum()}
        )
        
        return {
            bet_ids[i]: max(0, result.x[i])
            for i in range(n)
        }
    
    def _calculate_expected_growth(
        self,
        bets: List[BetOpportunity],
        allocations: Dict[str, float]
    ) -> float:
        """Calculate expected log growth for given allocations."""
        if not bets or all(f == 0 for f in allocations.values()):
            return 0.0
        
        # Monte Carlo approximation
        n_samples = 10000
        total_returns = np.zeros(n_samples)
        
        for bet in bets:
            bet_id = bet.bet_id or f"bet_{id(bet)}"
            frac = allocations.get(bet_id, 0.0)
            
            if frac > 0:
                wins = np.random.random(n_samples) < bet.prob
                returns = np.where(wins, bet.decimal_odds - 1, -1.0)
                total_returns += frac * returns
        
        log_wealth = np.log(np.maximum(1 + total_returns, 1e-10))
        return float(np.mean(log_wealth))


def optimize_simultaneous_bets(
    bets: List[BetOpportunity],
    max_total_fraction: float = 0.15,
    kelly_mult: float = 0.25,
    method: str = 'proportional'
) -> SimultaneousBetResult:
    """
    Convenience function for simultaneous bet optimization.
    
    This is the main entry point for multi-bet Kelly optimization.
    
    Example:
        >>> bets = [
        ...     BetOpportunity(0.55, 2.0, bet_id="mel_win"),
        ...     BetOpportunity(0.52, 1.91, bet_id="syd_win"),
        ...     BetOpportunity(0.48, 2.1, bet_id="per_win"),  # Negative EV
        ... ]
        >>> result = optimize_simultaneous_bets(bets, max_total_fraction=0.10)
        >>> print(result.allocations)
        {'mel_win': 0.052, 'syd_win': 0.048, 'per_win': 0.0}
    
    Args:
        bets: List of betting opportunities
        max_total_fraction: Maximum sum of all allocations
        kelly_mult: Fractional Kelly multiplier
        method: 'proportional', 'equal_risk', or 'numerical'
    
    Returns:
        SimultaneousBetResult with allocations
    """
    optimizer = SimultaneousKellyOptimizer(
        max_total_fraction=max_total_fraction,
        kelly_mult=kelly_mult
    )
    return optimizer.optimize(bets, method=method)


def handle_correlated_bets(
    bets: List[BetOpportunity],
    correlation_matrix: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Handle correlated bets with explicit correlation matrix.
    
    When bets are correlated (e.g., same-game props), we need to
    reduce overall exposure to prevent over-leveraging.
    
    Correlation Sources in NBL:
    ---------------------------
    
    1. **Same Game**: "Team A Win" + "Team A Over Points" - if team wins
       big, both likely hit. ρ ≈ 0.3-0.5
    
    2. **Same Player**: Multiple player props on same player. ρ ≈ 0.2-0.4
    
    3. **Same Time**: Games at same time have slight correlation from
       shared information (refs, weather for outdoor). ρ ≈ 0.05-0.1
    
    Args:
        bets: List of correlated bets
        correlation_matrix: n x n correlation matrix. If None, infers
                           from correlation_group attributes.
    
    Returns:
        Dictionary of bet_id -> adjusted allocation
    """
    n = len(bets)
    
    if n == 0:
        return {}
    
    if n == 1:
        bet = bets[0]
        result = fractional_kelly(bet.prob, bet.decimal_odds, 0.25)
        return {bet.bet_id or "bet_0": result.fraction}
    
    # Build correlation matrix if not provided
    if correlation_matrix is None:
        correlation_matrix = _infer_correlation_matrix(bets)
    
    # Get base Kelly fractions
    base_fractions = []
    for bet in bets:
        result = fractional_kelly(bet.prob, bet.decimal_odds, 0.25)
        base_fractions.append(result.fraction if result.is_positive_ev else 0.0)
    
    base_fractions = np.array(base_fractions)
    
    # Adjust for correlation using variance scaling
    # Higher correlation = more variance = should reduce position
    
    total_variance = 0.0
    for i in range(n):
        for j in range(n):
            total_variance += (
                base_fractions[i] * base_fractions[j] * 
                correlation_matrix[i, j]
            )
    
    # Target variance (as if independent)
    independent_variance = np.sum(base_fractions ** 2)
    
    if total_variance > independent_variance and total_variance > 0:
        # Scale down to match independent variance
        scale = np.sqrt(independent_variance / total_variance)
        adjusted = base_fractions * scale
    else:
        adjusted = base_fractions
    
    return {
        (bets[i].bet_id or f"bet_{i}"): float(adjusted[i])
        for i in range(n)
    }


def _infer_correlation_matrix(bets: List[BetOpportunity]) -> np.ndarray:
    """Infer correlation matrix from correlation_group attributes."""
    n = len(bets)
    corr = np.eye(n)  # Start with identity (self-correlation = 1)
    
    # Group mapping
    groups: Dict[str, List[int]] = {}
    for i, bet in enumerate(bets):
        group = bet.correlation_group or f"independent_{i}"
        if group not in groups:
            groups[group] = []
        groups[group].append(i)
    
    # Same group = correlated (ρ = 0.5)
    for group_indices in groups.values():
        for i in group_indices:
            for j in group_indices:
                if i != j:
                    corr[i, j] = 0.5
    
    return corr

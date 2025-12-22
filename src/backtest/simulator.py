"""
Bet Execution Simulator Module.

Simulates realistic bet placement with various staking strategies.
Implements Betfair Rule #5: Know When A Backtest Is Too Good To Be True
by modeling odds slippage and fill rate assumptions.

Staking Strategies:
    - FlatStake: Fixed amount per bet
    - PercentageStake: Fixed % of current bankroll
    - FractionalKelly: 25% (or custom) of Kelly optimal
    - FullKelly: Full Kelly optimal (high variance)
    - BoundedKelly: Kelly with floor protection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.portfolio.kelly import (
    BetOpportunity, 
    KellyResult, 
    kelly_criterion,
    fractional_kelly,
    bounded_kelly
)


class BetOutcome(Enum):
    """Possible outcomes of a bet."""
    WIN = "win"
    LOSS = "loss"
    VOID = "void"
    PUSH = "push"


@dataclass
class BetResult:
    """Result of a single bet execution."""
    bet_id: str
    stake: float
    odds: float
    prob_predicted: float
    outcome: BetOutcome
    profit: float
    pre_bankroll: float
    post_bankroll: float
    slippage_applied: float = 0.0
    
    @property
    def won(self) -> bool:
        """Whether the bet was a winner."""
        return self.outcome == BetOutcome.WIN
    
    @property
    def return_on_stake(self) -> float:
        """Return as percentage of stake."""
        return self.profit / self.stake if self.stake > 0 else 0


@dataclass
class SessionResult:
    """Result of a betting session (multiple bets)."""
    bets: List[BetResult] = field(default_factory=list)
    initial_bankroll: float = 0.0
    final_bankroll: float = 0.0
    
    @property
    def total_profit(self) -> float:
        """Total profit/loss."""
        return self.final_bankroll - self.initial_bankroll
    
    @property
    def roi(self) -> float:
        """Return on investment."""
        total_staked = sum(b.stake for b in self.bets)
        return self.total_profit / total_staked if total_staked > 0 else 0
    
    @property
    def num_bets(self) -> int:
        """Number of bets placed."""
        return len(self.bets)
    
    @property
    def num_wins(self) -> int:
        """Number of winning bets."""
        return sum(1 for b in self.bets if b.won)
    
    @property
    def win_rate(self) -> float:
        """Proportion of winning bets."""
        return self.num_wins / self.num_bets if self.num_bets > 0 else 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert bet results to DataFrame for analysis."""
        return pd.DataFrame([
            {
                'bet_id': b.bet_id,
                'stake': b.stake,
                'odds': b.odds,
                'prob_predicted': b.prob_predicted,
                'won': b.won,
                'outcome': b.outcome.value,
                'profit': b.profit,
                'pre_bankroll': b.pre_bankroll,
                'post_bankroll': b.post_bankroll,
                'slippage': b.slippage_applied,
            }
            for b in self.bets
        ])


# ============================================================================
# Staking Strategies
# ============================================================================

class StakingStrategy(ABC):
    """Abstract base class for staking strategies."""
    
    @abstractmethod
    def calculate_stake(
        self,
        bet: BetOpportunity,
        bankroll: float,
        peak_bankroll: Optional[float] = None
    ) -> float:
        """
        Calculate stake for a bet.
        
        Args:
            bet: Bet opportunity with probability and odds
            bankroll: Current bankroll
            peak_bankroll: Highest bankroll achieved (for bounded strategies)
        
        Returns:
            Stake amount (may be 0 if no bet recommended)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        pass


class FlatStake(StakingStrategy):
    """
    Fixed stake amount per bet.
    
    Simple and robust, but doesn't adjust for edge or bankroll size.
    """
    
    def __init__(self, stake_amount: float = 10.0):
        """
        Args:
            stake_amount: Fixed amount to stake per bet
        """
        self.stake_amount = stake_amount
    
    def calculate_stake(
        self,
        bet: BetOpportunity,
        bankroll: float,
        peak_bankroll: Optional[float] = None
    ) -> float:
        # Only bet positive EV opportunities
        edge = bet.prob - (1 / bet.decimal_odds)
        if edge <= 0:
            return 0.0
        
        # Don't bet more than bankroll
        return min(self.stake_amount, bankroll)
    
    @property
    def name(self) -> str:
        return f"Flat Stake (${self.stake_amount:.2f})"


class PercentageStake(StakingStrategy):
    """
    Fixed percentage of current bankroll.
    
    Scale naturally with wins/losses but doesn't adjust for edge size.
    """
    
    def __init__(self, percentage: float = 0.02):
        """
        Args:
            percentage: Fraction of bankroll to stake (0.02 = 2%)
        """
        self.percentage = percentage
    
    def calculate_stake(
        self,
        bet: BetOpportunity,
        bankroll: float,
        peak_bankroll: Optional[float] = None
    ) -> float:
        # Only bet positive EV opportunities
        edge = bet.prob - (1 / bet.decimal_odds)
        if edge <= 0:
            return 0.0
        
        return bankroll * self.percentage
    
    @property
    def name(self) -> str:
        return f"Percentage ({self.percentage:.1%})"


class FractionalKellyStake(StakingStrategy):
    """
    Fractional Kelly Criterion staking.
    
    Stakes a fraction of the Kelly optimal to reduce variance.
    Default 0.25 (quarter Kelly) is widely recommended.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_stake_pct: float = 0.05
    ):
        """
        Args:
            kelly_fraction: Fraction of Kelly optimal (0.25 = quarter Kelly)
            max_stake_pct: Maximum stake as % of bankroll (safety cap)
        """
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
    
    def calculate_stake(
        self,
        bet: BetOpportunity,
        bankroll: float,
        peak_bankroll: Optional[float] = None
    ) -> float:
        result = fractional_kelly(
            prob=bet.prob,
            decimal_odds=bet.decimal_odds,
            fraction=self.kelly_fraction
        )
        
        if result.fraction <= 0:
            return 0.0
        
        stake = result.stake_amount(bankroll)
        
        # Apply safety cap
        max_stake = bankroll * self.max_stake_pct
        return min(stake, max_stake)
    
    @property
    def name(self) -> str:
        return f"Fractional Kelly ({self.kelly_fraction:.0%})"


class FullKellyStake(StakingStrategy):
    """
    Full Kelly Criterion staking.
    
    Maximizes geometric growth but with high variance.
    Not recommended for practical use - use Fractional Kelly instead.
    """
    
    def __init__(self, max_stake_pct: float = 0.20):
        """
        Args:
            max_stake_pct: Maximum stake as % of bankroll (safety cap)
        """
        self.max_stake_pct = max_stake_pct
    
    def calculate_stake(
        self,
        bet: BetOpportunity,
        bankroll: float,
        peak_bankroll: Optional[float] = None
    ) -> float:
        result = kelly_criterion(
            prob=bet.prob,
            decimal_odds=bet.decimal_odds
        )
        
        if result.fraction <= 0:
            return 0.0
        
        stake = result.stake_amount(bankroll)
        
        # Apply safety cap
        max_stake = bankroll * self.max_stake_pct
        return min(stake, max_stake)
    
    @property
    def name(self) -> str:
        return "Full Kelly"


class BoundedKellyStake(StakingStrategy):
    """
    Kelly with floor protection.
    
    Reduces stake to protect a minimum bankroll level.
    Implements Betfair Rule #7: Know How To Manage Your Bankroll.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        floor_fraction: float = 0.20,
        max_stake_pct: float = 0.05
    ):
        """
        Args:
            kelly_fraction: Fraction of Kelly optimal
            floor_fraction: Minimum bankroll to protect (as % of peak)
            max_stake_pct: Maximum stake as % of bankroll
        """
        self.kelly_fraction = kelly_fraction
        self.floor_fraction = floor_fraction
        self.max_stake_pct = max_stake_pct
    
    def calculate_stake(
        self,
        bet: BetOpportunity,
        bankroll: float,
        peak_bankroll: Optional[float] = None
    ) -> float:
        # Use current bankroll as peak if not provided
        peak = peak_bankroll or bankroll
        floor = peak * self.floor_fraction
        
        result, max_frac = bounded_kelly(
            prob=bet.prob,
            decimal_odds=bet.decimal_odds,
            bankroll=bankroll,
            floor=floor,
            kelly_mult=self.kelly_fraction
        )
        
        if result.fraction <= 0:
            return 0.0
        
        stake = result.stake_amount(bankroll)
        
        # Apply safety cap
        max_stake = bankroll * self.max_stake_pct
        return min(stake, max_stake)
    
    @property
    def name(self) -> str:
        return f"Bounded Kelly ({self.kelly_fraction:.0%}, floor={self.floor_fraction:.0%})"


# ============================================================================
# Bet Simulator
# ============================================================================

class BetSimulator:
    """
    Simulates bet execution with realistic assumptions.
    
    Models:
        - Odds slippage: Actual odds may be worse than quoted
        - Fill rates: Not all bets get matched
        - Minimum bet sizes: Exchanges have minimum stakes
    
    Implements Betfair Rule #5: Realistic fill assumptions.
    """
    
    def __init__(
        self,
        slippage_rate: float = 0.02,
        fill_rate: float = 1.0,
        min_stake: float = 2.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize bet simulator.
        
        Args:
            slippage_rate: Average odds slippage (0.02 = 2%)
            fill_rate: Probability that bet gets matched (1.0 = always)
            min_stake: Minimum stake accepted
            random_seed: For reproducibility
        """
        self.slippage_rate = slippage_rate
        self.fill_rate = fill_rate
        self.min_stake = min_stake
        self.rng = np.random.default_rng(random_seed)
    
    def simulate_bet(
        self,
        bet: BetOpportunity,
        stake: float,
        actual_outcome: bool,
        bankroll: float
    ) -> BetResult:
        """
        Simulate execution of a single bet.
        
        Args:
            bet: Bet opportunity
            stake: Amount to stake
            actual_outcome: True if the bet wins
            bankroll: Current bankroll before bet
        
        Returns:
            BetResult with execution details
        """
        # Check minimum stake
        if stake < self.min_stake:
            return BetResult(
                bet_id=bet.bet_id or "unknown",
                stake=0.0,
                odds=bet.decimal_odds,
                prob_predicted=bet.prob,
                outcome=BetOutcome.VOID,
                profit=0.0,
                pre_bankroll=bankroll,
                post_bankroll=bankroll,
                slippage_applied=0.0
            )
        
        # Check fill rate
        if self.rng.random() > self.fill_rate:
            return BetResult(
                bet_id=bet.bet_id or "unknown",
                stake=0.0,
                odds=bet.decimal_odds,
                prob_predicted=bet.prob,
                outcome=BetOutcome.VOID,
                profit=0.0,
                pre_bankroll=bankroll,
                post_bankroll=bankroll,
                slippage_applied=0.0
            )
        
        # Apply odds slippage (unfavorable adjustment)
        slippage = self.slippage_rate * (self.rng.random() * 2)  # 0 to 2x avg
        actual_odds = bet.decimal_odds * (1 - slippage)
        actual_odds = max(actual_odds, 1.01)  # Minimum odds
        
        # Calculate profit/loss
        if actual_outcome:
            profit = stake * (actual_odds - 1)
            outcome = BetOutcome.WIN
        else:
            profit = -stake
            outcome = BetOutcome.LOSS
        
        return BetResult(
            bet_id=bet.bet_id or "unknown",
            stake=stake,
            odds=actual_odds,
            prob_predicted=bet.prob,
            outcome=outcome,
            profit=profit,
            pre_bankroll=bankroll,
            post_bankroll=bankroll + profit,
            slippage_applied=slippage
        )
    
    def simulate_session(
        self,
        bets: List[Tuple[BetOpportunity, bool]],
        staking_strategy: StakingStrategy,
        initial_bankroll: float = 1000.0
    ) -> SessionResult:
        """
        Simulate a session of multiple bets.
        
        Args:
            bets: List of (BetOpportunity, actual_outcome) tuples
            staking_strategy: Strategy for calculating stakes
            initial_bankroll: Starting bankroll
        
        Returns:
            SessionResult with all bet results
        """
        results = []
        bankroll = initial_bankroll
        peak_bankroll = initial_bankroll
        
        for bet, actual_outcome in bets:
            # Calculate stake using strategy
            stake = staking_strategy.calculate_stake(
                bet=bet,
                bankroll=bankroll,
                peak_bankroll=peak_bankroll
            )
            
            # Simulate bet execution
            result = self.simulate_bet(
                bet=bet,
                stake=stake,
                actual_outcome=actual_outcome,
                bankroll=bankroll
            )
            
            results.append(result)
            
            # Update bankroll
            bankroll = result.post_bankroll
            peak_bankroll = max(peak_bankroll, bankroll)
            
            # Stop if bankroll depleted
            if bankroll < self.min_stake:
                break
        
        return SessionResult(
            bets=results,
            initial_bankroll=initial_bankroll,
            final_bankroll=bankroll
        )
    
    def run_monte_carlo(
        self,
        bets: List[BetOpportunity],
        staking_strategy: StakingStrategy,
        n_simulations: int = 1000,
        initial_bankroll: float = 1000.0
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation to estimate outcome distribution.
        
        Simulates betting outcomes according to predicted probabilities,
        not actual outcomes.
        
        Args:
            bets: List of bet opportunities
            staking_strategy: Staking strategy to use
            n_simulations: Number of simulations
            initial_bankroll: Starting bankroll
        
        Returns:
            Dictionary with simulation statistics
        """
        final_bankrolls = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Generate random outcomes according to probabilities
            outcomes = [(bet, self.rng.random() < bet.prob) for bet in bets]
            
            result = self.simulate_session(
                bets=outcomes,
                staking_strategy=staking_strategy,
                initial_bankroll=initial_bankroll
            )
            
            final_bankrolls.append(result.final_bankroll)
            
            # Calculate max drawdown for this simulation
            equity = [initial_bankroll]
            for bet_result in result.bets:
                equity.append(bet_result.post_bankroll)
            
            equity = np.array(equity)
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max
            max_drawdowns.append(drawdowns.max())
        
        final_bankrolls = np.array(final_bankrolls)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            'mean_final_bankroll': final_bankrolls.mean(),
            'median_final_bankroll': np.median(final_bankrolls),
            'std_final_bankroll': final_bankrolls.std(),
            'prob_profit': (final_bankrolls > initial_bankroll).mean(),
            'prob_ruin': (final_bankrolls < initial_bankroll * 0.1).mean(),
            'mean_max_drawdown': max_drawdowns.mean(),
            'percentile_5': np.percentile(final_bankrolls, 5),
            'percentile_95': np.percentile(final_bankrolls, 95),
        }

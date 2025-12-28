"""
Reinforcement Learning Bankroll Manager.

Uses RL concepts to learn optimal bet sizing based on:
    - Current bankroll state
    - Model confidence
    - Recent betting performance
    - Edge magnitude

This implements a lightweight Q-learning approach without requiring
the full FinRL library, making it easier to integrate and deploy.

The agent learns when to:
    - Bet aggressively (high confidence, high edge)
    - Bet conservatively (uncertain, small edge)
    - Skip bets (negative expectation)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BettingState:
    """State representation for the RL agent."""
    bankroll: float                    # Current bankroll
    bankroll_pct_of_peak: float       # Drawdown indicator
    recent_win_rate: float            # Win rate last N bets
    current_streak: int               # Positive = wins, negative = losses
    model_confidence: float           # Model's confidence (0-1)
    edge: float                       # Predicted edge
    kelly_fraction: float             # Theoretical Kelly stake
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for Q-table lookup."""
        return np.array([
            self.bankroll_pct_of_peak,
            self.recent_win_rate,
            np.clip(self.current_streak / 10, -1, 1),  # Normalize
            self.model_confidence,
            np.clip(self.edge * 10, -1, 1),  # Scale edge
            np.clip(self.kelly_fraction * 4, 0, 1)  # Scale Kelly
        ])
    
    def discretize(self, n_bins: int = 5) -> Tuple[int, ...]:
        """Discretize state for Q-table."""
        arr = self.to_array()
        # Clip to [0, 1] range for binning
        arr = np.clip(arr, 0, 1)
        bins = np.floor(arr * (n_bins - 1)).astype(int)
        return tuple(bins)


@dataclass
class BettingAction:
    """Action space for betting decisions."""
    SKIP = 0          # Don't bet
    BET_SMALL = 1     # 0.5x Kelly
    BET_NORMAL = 2    # 1.0x Kelly 
    BET_LARGE = 3     # 1.5x Kelly
    
    @classmethod
    def get_multiplier(cls, action: int) -> float:
        """Convert action to Kelly multiplier."""
        multipliers = {
            cls.SKIP: 0.0,
            cls.BET_SMALL: 0.5,
            cls.BET_NORMAL: 1.0,
            cls.BET_LARGE: 1.5
        }
        return multipliers.get(action, 1.0)
    
    @classmethod
    def n_actions(cls) -> int:
        """Number of possible actions."""
        return 4


class RLBankrollManager:
    """
    Reinforcement Learning-based bankroll manager.
    
    Uses Q-learning to learn optimal bet sizing based on state.
    The agent learns from experience which bet sizes work best
    in different situations (drawdown, hot streak, etc.)
    
    Example:
        >>> manager = RLBankrollManager(initial_bankroll=1000)
        >>> 
        >>> # Get bet recommendation
        >>> state = manager.get_current_state(confidence=0.7, edge=0.05, kelly=0.1)
        >>> action, stake = manager.recommend_bet(state)
        >>> 
        >>> # Update after result
        >>> manager.update(state, action, won=True, stake=stake)
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
        n_bins: int = 5
    ):
        """
        Initialize the RL bankroll manager.
        
        Args:
            initial_bankroll: Starting bankroll
            learning_rate: Q-learning alpha
            discount_factor: Q-learning gamma
            exploration_rate: Epsilon for exploration
            n_bins: Discretization bins per state dimension
        """
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.n_bins = n_bins
        
        # Q-table: maps discretized state -> action values
        self.q_table: Dict[Tuple, np.ndarray] = {}
        
        # History tracking
        self.bet_history: List[Dict] = []
        self.recent_results: List[bool] = []
        self.current_streak = 0
        
        # Training mode
        self.training = True
    
    def _get_q_values(self, state: BettingState) -> np.ndarray:
        """Get Q-values for a state, initializing if needed."""
        key = state.discretize(self.n_bins)
        
        if key not in self.q_table:
            # Initialize optimistically to encourage exploration
            self.q_table[key] = np.ones(BettingAction.n_actions()) * 0.1
        
        return self.q_table[key]
    
    def _update_q_value(
        self,
        state: BettingState,
        action: int,
        reward: float,
        next_state: Optional[BettingState] = None
    ):
        """Update Q-value using TD learning."""
        key = state.discretize(self.n_bins)
        
        if key not in self.q_table:
            self.q_table[key] = np.ones(BettingAction.n_actions()) * 0.1
        
        # Max future Q-value
        if next_state is not None:
            next_q = np.max(self._get_q_values(next_state))
        else:
            next_q = 0
        
        # TD update
        current_q = self.q_table[key][action]
        self.q_table[key][action] = current_q + self.lr * (
            reward + self.gamma * next_q - current_q
        )
    
    def get_current_state(
        self,
        confidence: float,
        edge: float,
        kelly: float
    ) -> BettingState:
        """
        Build current state from betting context.
        
        Args:
            confidence: Model confidence (0-1)
            edge: Predicted edge vs bet
            kelly: Theoretical Kelly fraction
        
        Returns:
            BettingState for decision making
        """
        # Recent performance
        recent_n = min(20, len(self.recent_results))
        if recent_n > 0:
            recent_win_rate = sum(self.recent_results[-recent_n:]) / recent_n
        else:
            recent_win_rate = 0.5
        
        return BettingState(
            bankroll=self.bankroll,
            bankroll_pct_of_peak=self.bankroll / max(self.peak_bankroll, 1),
            recent_win_rate=recent_win_rate,
            current_streak=self.current_streak,
            model_confidence=confidence,
            edge=edge,
            kelly_fraction=kelly
        )
    
    def select_action(self, state: BettingState) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current betting state
        
        Returns:
            Action index
        """
        # Always skip if edge is negative
        if state.edge <= 0:
            return BettingAction.SKIP
        
        # Epsilon-greedy during training
        if self.training and np.random.random() < self.epsilon:
            return np.random.randint(BettingAction.n_actions())
        
        # Greedy action
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))
    
    def recommend_bet(
        self,
        state: BettingState,
        max_stake_pct: float = 0.05
    ) -> Tuple[int, float]:
        """
        Recommend bet action and stake amount.
        
        Args:
            state: Current betting state
            max_stake_pct: Maximum stake as fraction of bankroll
        
        Returns:
            Tuple of (action, stake_amount)
        """
        action = self.select_action(state)
        
        if action == BettingAction.SKIP:
            return action, 0.0
        
        # Calculate stake
        multiplier = BettingAction.get_multiplier(action)
        base_stake = state.kelly_fraction * self.bankroll * multiplier
        
        # Apply max stake limit
        max_stake = self.bankroll * max_stake_pct
        stake = min(base_stake, max_stake)
        stake = max(0, stake)
        
        return action, stake
    
    def calculate_reward(
        self,
        won: bool,
        stake: float,
        odds: float = 2.0
    ) -> float:
        """
        Calculate reward for the RL agent.
        
        Reward structure:
        - Positive for profitable bets
        - Penalty for large losses
        - Small reward for skipping negative EV
        
        Args:
            won: Whether bet won
            stake: Stake amount
            odds: Decimal odds
        
        Returns:
            Reward value
        """
        if stake == 0:
            # Small reward for skipping
            return 0.01
        
        if won:
            profit = stake * (odds - 1)
            # Scale reward by profit relative to bankroll
            reward = profit / self.initial_bankroll
        else:
            loss = stake
            # Penalize losses more heavily to encourage caution
            reward = -1.5 * (loss / self.initial_bankroll)
        
        return reward
    
    def update(
        self,
        state: BettingState,
        action: int,
        won: bool,
        stake: float,
        odds: float = 2.0
    ):
        """
        Update agent after bet result.
        
        Args:
            state: State when bet was placed
            action: Action taken
            won: Whether bet won
            stake: Stake amount
            odds: Decimal odds
        """
        # Calculate reward
        reward = self.calculate_reward(won, stake, odds)
        
        # Update bankroll
        if won:
            self.bankroll += stake * (odds - 1)
        else:
            self.bankroll -= stake
        
        # Update peak
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        
        # Update streak
        if stake > 0:
            if won:
                self.current_streak = max(1, self.current_streak + 1)
            else:
                self.current_streak = min(-1, self.current_streak - 1)
            self.recent_results.append(won)
        
        # Get next state for TD learning
        next_state = self.get_current_state(
            confidence=state.model_confidence,
            edge=state.edge,
            kelly=state.kelly_fraction
        )
        
        # Update Q-value
        self._update_q_value(state, action, reward, next_state)
        
        # Record history
        self.bet_history.append({
            'bankroll_before': state.bankroll,
            'bankroll_after': self.bankroll,
            'action': action,
            'stake': stake,
            'odds': odds,
            'won': won,
            'reward': reward,
            'edge': state.edge,
            'confidence': state.model_confidence
        })
    
    def train_on_historical(
        self,
        historical_bets: pd.DataFrame,
        n_epochs: int = 10
    ):
        """
        Train the agent on historical betting data.
        
        Args:
            historical_bets: DataFrame with columns:
                - model_confidence, edge, kelly, odds, won
            n_epochs: Number of training epochs
        """
        self.training = True
        
        for epoch in range(n_epochs):
            # Reset for each epoch
            self.bankroll = self.initial_bankroll
            self.peak_bankroll = self.initial_bankroll
            self.recent_results = []
            self.current_streak = 0
            
            for _, bet in historical_bets.iterrows():
                state = self.get_current_state(
                    confidence=bet['model_confidence'],
                    edge=bet['edge'],
                    kelly=bet['kelly']
                )
                
                action, stake = self.recommend_bet(state)
                
                self.update(
                    state=state,
                    action=action,
                    won=bet['won'],
                    stake=stake,
                    odds=bet['odds']
                )
            
            # Decay exploration
            self.epsilon *= 0.95
            
            final_bankroll = self.bankroll
            roi = (final_bankroll - self.initial_bankroll) / self.initial_bankroll
            logger.debug(f"Epoch {epoch + 1}: Final bankroll ${final_bankroll:.2f}, ROI {roi:.1%}")
        
        self.training = False
        logger.info(f"Training complete. Q-table has {len(self.q_table)} state entries.")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.bet_history:
            return {}
        
        df = pd.DataFrame(self.bet_history)
        bets_placed = df[df['stake'] > 0]
        
        if bets_placed.empty:
            return {}
        
        return {
            'total_bets': len(bets_placed),
            'win_rate': bets_placed['won'].mean(),
            'total_staked': bets_placed['stake'].sum(),
            'total_reward': df['reward'].sum(),
            'avg_stake': bets_placed['stake'].mean(),
            'final_bankroll': self.bankroll,
            'roi': (self.bankroll - self.initial_bankroll) / self.initial_bankroll,
            'max_drawdown': 1 - (df['bankroll_after'].min() / self.peak_bankroll)
        }
    
    def save(self, filepath: str):
        """Save the trained model."""
        data = {
            'q_table': self.q_table,
            'n_bins': self.n_bins,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.n_bins = data['n_bins']
        self.lr = data.get('lr', 0.1)
        self.gamma = data.get('gamma', 0.95)
        self.epsilon = data.get('epsilon', 0.1)
        self.training = False
        
        logger.info(f"Model loaded from {filepath}")


class AdaptiveKellyManager:
    """
    Adaptive Kelly criterion that adjusts based on recent performance.
    
    A simpler alternative to full RL that dynamically adjusts
    Kelly fraction based on:
        - Recent win rate
        - Current drawdown
        - Streak
    
    Example:
        >>> manager = AdaptiveKellyManager()
        >>> adjusted = manager.get_kelly_multiplier(recent_wins=7, recent_total=10)
        >>> stake = base_kelly * adjusted * bankroll
    """
    
    def __init__(
        self,
        base_fraction: float = 0.25,
        min_fraction: float = 0.10,
        max_fraction: float = 0.50,
        drawdown_threshold: float = 0.15
    ):
        """
        Initialize adaptive Kelly manager.
        
        Args:
            base_fraction: Default Kelly fraction
            min_fraction: Minimum during drawdowns
            max_fraction: Maximum during hot streaks
            drawdown_threshold: Drawdown level to trigger reduction
        """
        self.base_fraction = base_fraction
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.drawdown_threshold = drawdown_threshold
        
        self.bankroll = 1000.0
        self.peak_bankroll = 1000.0
        self.recent_results: List[bool] = []
    
    def get_kelly_multiplier(
        self,
        recent_wins: Optional[int] = None,
        recent_total: Optional[int] = None,
        current_drawdown: Optional[float] = None
    ) -> float:
        """
        Calculate Kelly fraction multiplier.
        
        Args:
            recent_wins: Wins in recent window
            recent_total: Total bets in recent window
            current_drawdown: Current drawdown percentage
        
        Returns:
            Multiplier to apply to base Kelly fraction
        """
        multiplier = 1.0
        
        # Adjust for recent performance
        if recent_wins is not None and recent_total is not None and recent_total > 0:
            win_rate = recent_wins / recent_total
            expected = 0.5  # Assume calibrated model
            
            if win_rate > expected + 0.1:
                # Hot streak - increase slightly
                multiplier *= min(1.3, 1 + (win_rate - expected))
            elif win_rate < expected - 0.1:
                # Cold streak - decrease
                multiplier *= max(0.5, 1 - (expected - win_rate))
        
        # Adjust for drawdown
        if current_drawdown is not None:
            if current_drawdown > self.drawdown_threshold:
                # In significant drawdown - reduce exposure
                reduction = min(0.5, current_drawdown * 2)
                multiplier *= (1 - reduction)
        
        # Apply bounds
        final_fraction = self.base_fraction * multiplier
        final_fraction = max(self.min_fraction, min(self.max_fraction, final_fraction))
        
        return final_fraction / self.base_fraction  # Return multiplier
    
    def update(self, won: bool, stake: float, odds: float):
        """Update state after bet."""
        if won:
            self.bankroll += stake * (odds - 1)
        else:
            self.bankroll -= stake
        
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        self.recent_results.append(won)
        
        # Keep last 20
        if len(self.recent_results) > 20:
            self.recent_results = self.recent_results[-20:]
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        return 1 - (self.bankroll / self.peak_bankroll)
    
    def recommend_stake(
        self,
        base_kelly: float,
        bankroll: Optional[float] = None
    ) -> float:
        """
        Recommend stake with adaptive Kelly.
        
        Args:
            base_kelly: Theoretical Kelly fraction
            bankroll: Current bankroll (uses internal if None)
        
        Returns:
            Recommended stake amount
        """
        bankroll = bankroll or self.bankroll
        
        recent_wins = sum(self.recent_results[-10:]) if self.recent_results else 5
        recent_total = len(self.recent_results[-10:]) if self.recent_results else 10
        
        multiplier = self.get_kelly_multiplier(
            recent_wins=recent_wins,
            recent_total=recent_total,
            current_drawdown=self.current_drawdown
        )
        
        adjusted_kelly = base_kelly * multiplier
        stake = bankroll * adjusted_kelly
        
        return stake

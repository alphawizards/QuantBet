"""
Bet Tracking Logic.

Provides core betting tracking calculations including:
- ROI (Return on Investment) calculation
- Win rate calculation
- Equity curve generation
- Comprehensive betting statistics

Example:
    >>> from src.betting.bet_tracker import calculate_roi, generate_equity_curve
    >>> bets = [
    ...     {'stake': 100, 'odds': 2.0, 'result': 'win'},
    ...     {'stake': 100, 'odds': 2.0, 'result': 'loss'},
    ... ]
    >>> roi = calculate_roi(bets)
    >>> roi
    0.0
"""

import numpy as np
from typing import List, Dict, Any
from datetime import datetime


def calculate_roi(bets: List[Dict[str, Any]]) -> float:
    """
   

 Calculate Return on Investment (ROI).
    
    ROI = (Total Profit / Total Stake) * 100
    
    Args:
        bets: List of bet dictionaries with 'stake', 'odds', 'result' keys
    
    Returns:
        ROI as a percentage (100.0 = 100% ROI, -50.0 = -50% ROI)
    
    Raises:
        ValueError: If bets list is empty
    
    Example:
        >>> bets = [{'stake': 100, 'odds': 2.0, 'result': 'win'}]
        >>> roi = calculate_roi(bets)
        >>> roi
        100.0  # 100% ROI (doubled money)
    """
    if not bets:
        raise ValueError("Cannot calculate ROI with no bets")
    
    completed_bets = [b for b in bets if b.get('result') is not None]
    
    if not completed_bets:
        return 0.0
    
    total_stake = sum(b['stake'] for b in completed_bets)
    total_returns = sum(
        b['stake'] * b['odds'] if b['result'] == 'win' else 0
        for b in completed_bets
    )
    total_profit = total_returns - total_stake
    
    if total_stake == 0:
        return 0.0
    
    return (total_profit / total_stake) * 100


def calculate_win_rate(bets: List[Dict[str, Any]]) -> float:
    """
    Calculate win rate as percentage of bets won.
    
    Win Rate = (Wins / Total Bets) * 100
    
    Args:
        bets: List of bet dictionaries
    
    Returns:
        Win rate as a percentage (0-100)
    
    Raises:
        ValueError: If bets list is empty
    
    Example:
        >>> bets = [
        ...     {'stake': 100, 'odds': 2.0, 'result': 'win'},
        ...     {'stake': 100, 'odds': 2.0, 'result': 'loss'},
        ... ]
        >>> win_rate = calculate_win_rate(bets)
        >>> win_rate
        50.0  # 50% win rate
    """
    if not bets:
        raise ValueError("Cannot calculate win rate with no bets")
    
    completed_bets = [b for b in bets if b.get('result') is not None]
    
    if not completed_bets:
        return 0.0
    
    wins = sum(1 for b in completed_bets if b['result'] == 'win')
    
    return (wins / len(completed_bets)) * 100


def generate_equity_curve(
    bets: List[Dict[str, Any]], 
    starting_bankroll: float = 1000.0
) -> List[Dict[str, Any]]:
    """
    Generate equity curve showing bankroll over time.
    
    Tracks how the bankroll changes after each bet, sorted chronologically.
    
    Args:
        bets: List of bet dictionaries with 'stake', 'odds', 'result', 'timestamp'
        starting_bankroll: Initial bankroll amount (default 1000.0)
    
    Returns:
        List of equity curve points, each containing:
        - timestamp: When the bet was placed
        - bankroll: Current bankroll after this bet
        - cumulative_profit: Total profit/loss from start
        - bet_id: Identifier for the bet (if available)
    
    Example:
        >>> from datetime import datetime
        >>> bets = [
        ...     {'stake': 100, 'odds': 2.0, 'result': 'win', 
        ...      'timestamp': datetime(2024, 1, 1)},
        ...     {'stake': 100, 'odds': 2.0, 'result': 'loss', 
        ...      'timestamp': datetime(2024, 1, 2)},
        ... ]
        >>> curve = generate_equity_curve(bets, starting_bankroll=1000)
        >>> curve[0]['bankroll']
        1100.0  # Won 100
        >>> curve[1]['bankroll']
        1000.0  # Lost 100, back to starting
    """
    if not bets:
        return []
    
    # Sort bets by timestamp
    sorted_bets = sorted(bets, key=lambda b: b.get('timestamp', datetime.now()))
    
    curve = []
    bankroll = starting_bankroll
    cumulative_profit = 0.0
    
    for bet in sorted_bets:
        if bet.get('result') is None:
            continue  # Skip bets without results
        
        stake = bet['stake']
        
        if bet['result'] == 'win':
            profit = stake * (bet['odds'] - 1)
        elif bet['result'] == 'loss':
            profit = -stake
        else:  # push/void
            profit = 0
        
        bankroll += profit
        cumulative_profit += profit
        
        curve.append({
            'timestamp': bet['timestamp'],
            'bankroll': bankroll,
            'cumulative_profit': cumulative_profit,
            'bet_id': bet.get('bet_id', 'unknown')
        })
    
    return curve


def calculate_bet_statistics(bets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive betting statistics.
    
    Aggregates all betting metrics into a single summary.
    
    Args:
        bets: List of bet dictionaries
    
    Returns:
        Dictionary containing:
        - total_bets: Total number of bets placed
        - completed_bets: Bets with results
        - pending_bets: Bets without results
        - roi: Return on investment percentage
        - win_rate: Percentage of bets won
        - total_staked: Total amount wagered
        - total_profit: Total profit/loss
        - average_odds: Average odds taken
        - largest_win: Largest single win amount
        - largest_loss: Largest single loss amount
    
    Example:
        >>> bets = [
        ...     {'stake': 100, 'odds': 2.0, 'result': 'win'},
        ...     {'stake': 100, 'odds': 2.0, 'result': 'loss'},
        ... ]
        >>> stats = calculate_bet_statistics(bets)
        >>> stats['roi']
        0.0
        >>> stats['win_rate']
        50.0
    """
    if not bets:
        return {
            'total_bets': 0,
            'completed_bets': 0,
            'pending_bets': 0,
            'roi': 0.0,
            'win_rate': 0.0,
            'total_staked': 0.0,
            'total_profit': 0.0,
            'average_odds': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
    
    completed = [b for b in bets if b.get('result') is not None]
    
    # Calculate basic metrics
    total_stake = sum(b['stake'] for b in completed)
    profits = []
    
    for bet in completed:
        if bet['result'] == 'win':
            profit = bet['stake'] * (bet['odds'] - 1)
        elif bet['result'] == 'loss':
            profit = -bet['stake']
        else:  # push
            profit = 0
        profits.append(profit)
    
    total_profit = sum(profits)
    
    return {
        'total_bets': len(bets),
        'completed_bets': len(completed),
        'pending_bets': len(bets) - len(completed),
        'roi': (total_profit / total_stake * 100) if total_stake > 0 else 0.0,
        'win_rate': calculate_win_rate(bets) if completed else 0.0,
        'total_staked': total_stake,
        'total_profit': total_profit,
        'average_odds': np.mean([b['odds'] for b in completed]) if completed else 0.0,
        'largest_win': max(profits) if profits else 0.0,
        'largest_loss': min(profits) if profits else 0.0
    }

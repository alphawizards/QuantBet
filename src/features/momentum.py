"""
Momentum Features for Team Performance Analysis.

Implements technical analysis-style indicators adapted for basketball:
    - RSI (Relative Strength Index) for team form
    - Moving average crossovers for trend detection
    - Streak momentum scores

These features capture whether a team is "hot" or "cold" beyond
simple win percentage.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_rsi(
    margins: np.ndarray,
    period: int = 5
) -> float:
    """
    Calculate RSI-like momentum indicator for scoring margins.
    
    RSI measures the momentum of a team's performance using their
    scoring margins. High RSI (>70) suggests "hot" team, low RSI (<30)
    suggests "cold" team.
    
    Formula adapted from financial RSI:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
    
    Args:
        margins: Array of scoring margins (positive = win)
        period: Number of games for calculation
    
    Returns:
        RSI value between 0-100, or 50 if insufficient data
    """
    if len(margins) < 2:
        return 50.0  # Neutral
    
    # Use last 'period' games
    margins = margins[-period:] if len(margins) > period else margins
    
    # Calculate gains and losses
    gains = np.where(margins > 0, margins, 0)
    losses = np.where(margins < 0, -margins, 0)
    
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def calculate_moving_average(
    values: np.ndarray,
    period: int
) -> Optional[float]:
    """
    Calculate simple moving average.
    
    Args:
        values: Array of values
        period: MA period
    
    Returns:
        Moving average or None if insufficient data
    """
    if len(values) < period:
        return None
    return np.mean(values[-period:])


def calculate_ma_crossover(
    margins: np.ndarray,
    short_period: int = 3,
    long_period: int = 7
) -> float:
    """
    Calculate moving average crossover signal.
    
    Positive value = short MA above long MA (uptrend)
    Negative value = short MA below long MA (downtrend)
    
    Args:
        margins: Array of scoring margins
        short_period: Short MA period
        long_period: Long MA period
    
    Returns:
        Difference between short and long MA, or 0 if insufficient data
    """
    if len(margins) < long_period:
        return 0.0
    
    short_ma = np.mean(margins[-short_period:])
    long_ma = np.mean(margins[-long_period:])
    
    return round(short_ma - long_ma, 2)


def calculate_volatility(
    margins: np.ndarray,
    period: int = 5
) -> float:
    """
    Calculate performance volatility (standard deviation of margins).
    
    High volatility = unpredictable team
    Low volatility = consistent team
    
    Args:
        margins: Array of scoring margins
        period: Number of games
    
    Returns:
        Standard deviation of margins
    """
    if len(margins) < 2:
        return 10.0  # Default
    
    margins = margins[-period:] if len(margins) > period else margins
    return round(np.std(margins), 2)


def calculate_momentum_score(
    margins: np.ndarray,
    period: int = 5
) -> float:
    """
    Calculate composite momentum score combining multiple indicators.
    
    Score range: -100 to +100
        - Positive = team trending up
        - Negative = team trending down
        - Near zero = neutral/stable
    
    Args:
        margins: Array of scoring margins
        period: Lookback period
    
    Returns:
        Composite momentum score
    """
    if len(margins) < 2:
        return 0.0
    
    # RSI component (0-100 -> -50 to +50)
    rsi = calculate_rsi(margins, period)
    rsi_component = rsi - 50  # Center around 0
    
    # Trend component (MA crossover)
    crossover = calculate_ma_crossover(margins, min(3, len(margins)), min(7, len(margins)))
    trend_component = np.clip(crossover * 5, -25, 25)  # Scale and cap
    
    # Recent wins component
    recent = margins[-min(3, len(margins)):]
    wins_component = (np.sum(recent > 0) / len(recent) - 0.5) * 50
    
    # Combine
    momentum = rsi_component * 0.4 + trend_component * 0.3 + wins_component * 0.3
    
    return round(np.clip(momentum, -100, 100), 2)


def get_team_margins(
    team_code: str,
    game_date: pd.Timestamp,
    historical_data: pd.DataFrame,
    n_games: int = 10
) -> np.ndarray:
    """
    Extract scoring margins for a team from historical data.
    
    Args:
        team_code: Team identifier
        game_date: Cutoff date (only games before this)
        historical_data: DataFrame with game results
        n_games: Max games to retrieve
    
    Returns:
        Array of scoring margins (positive = win)
    """
    # Filter to team's games before the date
    team_games = historical_data[
        ((historical_data['home_team'] == team_code) |
         (historical_data['away_team'] == team_code)) &
        (historical_data['game_date'] < game_date)
    ].sort_values('game_date', ascending=False).head(n_games)
    
    if team_games.empty:
        return np.array([])
    
    margins = []
    for _, game in team_games.iterrows():
        if game['home_team'] == team_code:
            margin = game['home_score'] - game['away_score']
        else:
            margin = game['away_score'] - game['home_score']
        margins.append(margin)
    
    # Reverse to chronological order (oldest first)
    return np.array(margins[::-1])


def calculate_momentum_features(
    team_code: str,
    game_date: pd.Timestamp,
    historical_data: pd.DataFrame,
    n_games: int = 10
) -> Dict[str, float]:
    """
    Calculate all momentum features for a team.
    
    Args:
        team_code: Team identifier
        game_date: Game date for prediction
        historical_data: Historical game results
        n_games: Lookback window
    
    Returns:
        Dictionary of momentum features:
            - momentum_rsi: RSI indicator (0-100)
            - momentum_trend: MA crossover signal
            - momentum_volatility: Performance consistency
            - momentum_score: Composite score (-100 to +100)
            - momentum_last3_wpct: Win % in last 3 games
    """
    margins = get_team_margins(team_code, game_date, historical_data, n_games)
    
    if len(margins) < 2:
        return {
            'momentum_rsi': 50.0,
            'momentum_trend': 0.0,
            'momentum_volatility': 10.0,
            'momentum_score': 0.0,
            'momentum_last3_wpct': 0.5
        }
    
    # Calculate features
    rsi = calculate_rsi(margins, period=5)
    trend = calculate_ma_crossover(margins, short_period=3, long_period=7)
    volatility = calculate_volatility(margins, period=5)
    score = calculate_momentum_score(margins, period=5)
    
    # Last 3 games win percentage
    last3 = margins[-3:] if len(margins) >= 3 else margins
    last3_wpct = np.sum(last3 > 0) / len(last3) if len(last3) > 0 else 0.5
    
    return {
        'momentum_rsi': rsi,
        'momentum_trend': trend,
        'momentum_volatility': volatility,
        'momentum_score': score,
        'momentum_last3_wpct': round(last3_wpct, 3)
    }


def calculate_momentum_differential(
    home_features: Dict[str, float],
    away_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate momentum differential between home and away teams.
    
    Args:
        home_features: Home team momentum features
        away_features: Away team momentum features
    
    Returns:
        Dictionary of differential features
    """
    return {
        'momentum_rsi_diff': home_features['momentum_rsi'] - away_features['momentum_rsi'],
        'momentum_trend_diff': home_features['momentum_trend'] - away_features['momentum_trend'],
        'momentum_score_diff': home_features['momentum_score'] - away_features['momentum_score'],
        'momentum_wpct_diff': home_features['momentum_last3_wpct'] - away_features['momentum_last3_wpct'],
        'momentum_volatility_ratio': (
            home_features['momentum_volatility'] / 
            max(away_features['momentum_volatility'], 1)
        )
    }

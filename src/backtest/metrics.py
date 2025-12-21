"""
Backtesting Performance Metrics Module.

Comprehensive metrics for evaluating betting model performance,
aligned with Betfair's 10 Golden Rules of Automation.

Metrics Categories:
    1. Profitability - ROI, P&L, Yield
    2. Risk-Adjusted - Sharpe, Sortino, Calmar
    3. Drawdown Analysis - Max DD, Duration
    4. Statistical - Win rate, calibration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestMetrics:
    """
    Comprehensive backtesting performance metrics.
    
    Organized by category for easy interpretation.
    All ratios are annualized assuming ~150 NBL games per season.
    
    Attributes:
        roi: Return on Investment (total P&L / total staked)
        profit_loss: Absolute profit/loss in currency units
        yield_per_bet: Average return per bet
        total_bets: Number of bets placed
        total_staked: Total amount wagered
        
        sharpe_ratio: Annualized risk-adjusted return
        sortino_ratio: Downside risk-adjusted return
        calmar_ratio: Return / Max Drawdown
        
        max_drawdown: Maximum peak-to-trough decline
        max_drawdown_pct: Max drawdown as percentage
        max_drawdown_duration: Bets to recover from max DD
        current_drawdown_pct: Current drawdown percentage
        
        win_rate: Proportion of winning bets
        avg_odds_winner: Average odds on winning bets
        avg_odds_loser: Average odds on losing bets
        
        brier_score: Probability prediction accuracy
        log_loss: Log-likelihood loss
        calibration_error: Expected Calibration Error
    """
    # Profitability
    roi: float = 0.0
    profit_loss: float = 0.0
    yield_per_bet: float = 0.0
    total_bets: int = 0
    total_staked: float = 0.0
    
    # Risk-Adjusted Returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown_pct: float = 0.0
    
    # Win/Loss Statistics
    win_rate: float = 0.0
    avg_odds_winner: float = 0.0
    avg_odds_loser: float = 0.0
    profit_factor: float = 0.0  # Gross profit / Gross loss
    
    # Prediction Quality
    brier_score: float = 0.0
    log_loss: float = 0.0
    calibration_error: float = 0.0
    
    # Kelly-Specific
    geometric_growth_rate: float = 0.0
    edge_utilization: float = 0.0  # Actual return / Theoretical Kelly return
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return f"""
═══════════════════════════════════════════════════════════════
                    BACKTEST PERFORMANCE REPORT
═══════════════════════════════════════════════════════════════

📊 PROFITABILITY
   ROI:                 {self.roi:+.2%}
   Profit/Loss:         ${self.profit_loss:+,.2f}
   Yield per Bet:       {self.yield_per_bet:+.2%}
   Total Bets:          {self.total_bets:,}
   Total Staked:        ${self.total_staked:,.2f}

📈 RISK-ADJUSTED RETURNS
   Sharpe Ratio:        {self.sharpe_ratio:.2f}
   Sortino Ratio:       {self.sortino_ratio:.2f}
   Calmar Ratio:        {self.calmar_ratio:.2f}

📉 DRAWDOWN ANALYSIS
   Max Drawdown:        {self.max_drawdown_pct:.2%}
   Max DD Duration:     {self.max_drawdown_duration} bets
   Current Drawdown:    {self.current_drawdown_pct:.2%}

🎯 WIN/LOSS STATISTICS
   Win Rate:            {self.win_rate:.2%}
   Avg Odds (Winners):  {self.avg_odds_winner:.2f}
   Avg Odds (Losers):   {self.avg_odds_loser:.2f}
   Profit Factor:       {self.profit_factor:.2f}

🔬 PREDICTION QUALITY
   Brier Score:         {self.brier_score:.4f}
   Log Loss:            {self.log_loss:.4f}
   Calibration Error:   {self.calibration_error:.4f}

⚡ KELLY METRICS
   Geometric Growth:    {self.geometric_growth_rate:.4f}
   Edge Utilization:    {self.edge_utilization:.2%}

═══════════════════════════════════════════════════════════════
"""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'roi': self.roi,
            'profit_loss': self.profit_loss,
            'yield_per_bet': self.yield_per_bet,
            'total_bets': self.total_bets,
            'total_staked': self.total_staked,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_drawdown_duration': self.max_drawdown_duration,
            'current_drawdown_pct': self.current_drawdown_pct,
            'win_rate': self.win_rate,
            'avg_odds_winner': self.avg_odds_winner,
            'avg_odds_loser': self.avg_odds_loser,
            'profit_factor': self.profit_factor,
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'calibration_error': self.calibration_error,
            'geometric_growth_rate': self.geometric_growth_rate,
            'edge_utilization': self.edge_utilization,
        }


def calculate_metrics(
    bet_results: pd.DataFrame,
    predictions: Optional[np.ndarray] = None,
    actuals: Optional[np.ndarray] = None,
    initial_bankroll: float = 1000.0,
    annualization_factor: float = 150  # ~NBL games per season
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics from bet results.
    
    Args:
        bet_results: DataFrame with columns:
            - stake: Amount wagered
            - odds: Decimal odds
            - won: Boolean, True if bet won
            - profit: Profit/loss from bet
        predictions: Optional probability predictions for calibration
        actuals: Optional actual outcomes for calibration
        initial_bankroll: Starting bankroll for drawdown calculation
        annualization_factor: Scale factor for Sharpe/Sortino
    
    Returns:
        BacktestMetrics with all calculated values
    """
    if len(bet_results) == 0:
        return BacktestMetrics()
    
    # Basic statistics
    total_bets = len(bet_results)
    total_staked = bet_results['stake'].sum()
    profit_loss = bet_results['profit'].sum()
    roi = profit_loss / total_staked if total_staked > 0 else 0
    yield_per_bet = profit_loss / total_bets if total_bets > 0 else 0
    
    # Win/Loss stats
    winners = bet_results[bet_results['won'] == True]
    losers = bet_results[bet_results['won'] == False]
    
    win_rate = len(winners) / total_bets if total_bets > 0 else 0
    avg_odds_winner = winners['odds'].mean() if len(winners) > 0 else 0
    avg_odds_loser = losers['odds'].mean() if len(losers) > 0 else 0
    
    gross_profit = winners['profit'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['profit'].sum()) if len(losers) > 0 else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate equity curve
    equity_curve = calculate_equity_curve(bet_results, initial_bankroll)
    
    # Drawdown analysis
    max_dd, max_dd_pct, max_dd_duration = calculate_drawdown_metrics(equity_curve)
    current_dd_pct = (equity_curve.max() - equity_curve.iloc[-1]) / equity_curve.max()
    
    # Risk-adjusted returns
    returns = bet_results['profit'] / bet_results['stake']
    sharpe = calculate_sharpe_ratio(returns, annualization_factor)
    sortino = calculate_sortino_ratio(returns, annualization_factor)
    calmar = roi / max_dd_pct if max_dd_pct > 0 else 0
    
    # Prediction quality (if provided)
    brier = 0.0
    logloss = 0.0
    ece = 0.0
    
    if predictions is not None and actuals is not None:
        brier = calculate_brier_score(predictions, actuals)
        logloss = calculate_log_loss(predictions, actuals)
        ece = calculate_calibration_error(predictions, actuals)
    
    # Geometric growth rate
    log_returns = np.log1p(returns.clip(lower=-0.99))
    geo_growth = log_returns.mean()
    
    return BacktestMetrics(
        roi=roi,
        profit_loss=profit_loss,
        yield_per_bet=yield_per_bet,
        total_bets=total_bets,
        total_staked=total_staked,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_duration=max_dd_duration,
        current_drawdown_pct=current_dd_pct,
        win_rate=win_rate,
        avg_odds_winner=avg_odds_winner,
        avg_odds_loser=avg_odds_loser,
        profit_factor=profit_factor,
        brier_score=brier,
        log_loss=logloss,
        calibration_error=ece,
        geometric_growth_rate=geo_growth,
        edge_utilization=0.0,  # Requires theoretical Kelly return
    )


def calculate_equity_curve(
    bet_results: pd.DataFrame,
    initial_bankroll: float = 1000.0
) -> pd.Series:
    """
    Calculate cumulative equity curve from bet results.
    
    Args:
        bet_results: DataFrame with 'profit' column
        initial_bankroll: Starting bankroll
    
    Returns:
        Series with cumulative equity
    """
    cumulative_pnl = bet_results['profit'].cumsum()
    equity = initial_bankroll + cumulative_pnl
    return equity


def calculate_drawdown_metrics(
    equity_curve: pd.Series
) -> Tuple[float, float, int]:
    """
    Calculate maximum drawdown and duration.
    
    Args:
        equity_curve: Series of equity values
    
    Returns:
        Tuple of (max_drawdown_absolute, max_drawdown_pct, max_duration)
    """
    # Running maximum
    running_max = equity_curve.cummax()
    
    # Drawdown series
    drawdown = running_max - equity_curve
    drawdown_pct = drawdown / running_max
    
    # Maximum drawdown
    max_dd = drawdown.max()
    max_dd_pct = drawdown_pct.max()
    
    # Drawdown duration (bets to recover)
    in_drawdown = drawdown > 0
    
    max_duration = 0
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_dd, max_dd_pct, max_duration


def calculate_sharpe_ratio(
    returns: pd.Series,
    annualization_factor: float = 150,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of per-bet returns
        annualization_factor: Scale factor (NBL games per season)
        risk_free_rate: Risk-free rate (annual)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / annualization_factor)
    
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    annualization_factor: float = 150,
    target_return: float = 0.0
) -> float:
    """
    Calculate annualized Sortino ratio (downside risk only).
    
    Args:
        returns: Series of per-bet returns
        annualization_factor: Scale factor
        target_return: Minimum acceptable return
    
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean()
    
    # Downside deviation
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return float('inf') if mean_return > 0 else 0.0
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    
    if downside_std == 0:
        return 0.0
    
    sortino = (mean_return / downside_std) * np.sqrt(annualization_factor)
    return sortino


def calculate_brier_score(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> float:
    """
    Calculate Brier score for probability predictions.
    
    Lower is better. Perfect = 0, Random = 0.25
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes (0 or 1)
    
    Returns:
        Brier score
    """
    return np.mean((predictions - actuals) ** 2)


def calculate_log_loss(
    predictions: np.ndarray,
    actuals: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Calculate log loss (cross-entropy).
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes
        eps: Small value to prevent log(0)
    
    Returns:
        Log loss
    """
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(
        actuals * np.log(predictions) + 
        (1 - actuals) * np.log(1 - predictions)
    )


def calculate_calibration_error(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Measures how well predicted probabilities match actual frequencies.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes
        n_bins: Number of probability bins
    
    Returns:
        Expected Calibration Error
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        
        if mask.sum() == 0:
            continue
        
        bin_confidence = predictions[mask].mean()
        bin_accuracy = actuals[mask].mean()
        bin_size = mask.sum() / len(predictions)
        
        ece += bin_size * abs(bin_accuracy - bin_confidence)
    
    return ece


def compare_metrics(
    metrics_a: BacktestMetrics,
    metrics_b: BacktestMetrics,
    name_a: str = "Model A",
    name_b: str = "Model B"
) -> str:
    """
    Generate comparison table between two sets of metrics.
    
    Args:
        metrics_a: First model's metrics
        metrics_b: Second model's metrics
        name_a: Name for first model
        name_b: Name for second model
    
    Returns:
        Formatted comparison string
    """
    def delta(a: float, b: float, higher_is_better: bool = True) -> str:
        diff = a - b
        if higher_is_better:
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        else:
            arrow = "↓" if diff > 0 else "↑" if diff < 0 else "="
        return f"{diff:+.4f} {arrow}"
    
    return f"""
╔════════════════════════════════════════════════════════════════════╗
║                    MODEL COMPARISON: {name_a} vs {name_b}
╠════════════════════════════════════════════════════════════════════╣
║ Metric                │ {name_a:^12} │ {name_b:^12} │ Delta
╠═══════════════════════╪══════════════╪══════════════╪═══════════════╣
║ ROI                   │ {metrics_a.roi:+.2%}      │ {metrics_b.roi:+.2%}      │ {delta(metrics_a.roi, metrics_b.roi)}
║ Sharpe Ratio          │ {metrics_a.sharpe_ratio:+.2f}        │ {metrics_b.sharpe_ratio:+.2f}        │ {delta(metrics_a.sharpe_ratio, metrics_b.sharpe_ratio)}
║ Max Drawdown          │ {metrics_a.max_drawdown_pct:.2%}       │ {metrics_b.max_drawdown_pct:.2%}       │ {delta(metrics_a.max_drawdown_pct, metrics_b.max_drawdown_pct, False)}
║ Win Rate              │ {metrics_a.win_rate:.2%}       │ {metrics_b.win_rate:.2%}       │ {delta(metrics_a.win_rate, metrics_b.win_rate)}
║ Brier Score           │ {metrics_a.brier_score:.4f}       │ {metrics_b.brier_score:.4f}       │ {delta(metrics_a.brier_score, metrics_b.brier_score, False)}
╚═══════════════════════╧══════════════╧══════════════╧═══════════════╝
"""

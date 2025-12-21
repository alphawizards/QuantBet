"""
Risk Management and Continuous-Time Modeling for Sports Betting.

This module provides tools for:
    1. Ruin probability calculation
    2. Geometric Brownian Motion (GBM) bankroll forecasting
    3. Drawdown analysis
    4. Risk metrics (Sharpe, Sortino, Kelly growth rate)

Continuous-Time Approximation
=============================

While sports betting is inherently discrete (each bet is a binary outcome),
we can model long-term bankroll evolution using continuous-time stochastic
processes for:
    - Forecasting future bankroll distribution
    - Calculating confidence intervals on growth
    - Understanding tail risk behavior

The key model is Geometric Brownian Motion (GBM):

.. math::

    dW = \\mu W dt + \\sigma W dB_t

Where:
    - W = bankroll
    - μ = drift (expected growth rate from Kelly strategy)
    - σ = volatility of returns
    - B_t = standard Brownian motion

This has the solution:

.. math::

    W(t) = W_0 \\exp\\left((\\mu - \\frac{\\sigma^2}{2})t + \\sigma B_t\\right)

Which gives a log-normal distribution for future bankroll values.

References
----------
- Merton, R.C. (1969). "Lifetime Portfolio Selection under Uncertainty"
- Thorp, E.O. (2006). "The Kelly Criterion in Blackjack Sports Betting..."
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class RiskMetrics:
    """
    Container for portfolio risk metrics.
    
    Attributes:
        sharpe_ratio: Annualized risk-adjusted return
        sortino_ratio: Downside risk-adjusted return
        max_drawdown: Maximum peak-to-trough decline
        current_drawdown: Current decline from peak
        kelly_growth: Expected log growth rate per bet
        win_rate: Historical win rate
        avg_odds: Average decimal odds on bets
    """
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    kelly_growth: float
    win_rate: float
    avg_odds: float
    
    def __str__(self) -> str:
        return (
            f"Risk Metrics:\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"  Sortino Ratio: {self.sortino_ratio:.2f}\n"
            f"  Max Drawdown: {self.max_drawdown:.1%}\n"
            f"  Current Drawdown: {self.current_drawdown:.1%}\n"
            f"  Kelly Growth: {self.kelly_growth:.4f}\n"
            f"  Win Rate: {self.win_rate:.1%}\n"
            f"  Avg Odds: {self.avg_odds:.2f}"
        )


@dataclass
class GBMForecast:
    """
    Forecast from Geometric Brownian Motion model.
    
    Attributes:
        expected_wealth: E[W(t)]
        median_wealth: Median W(t) (50th percentile)
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        prob_above_initial: P(W(t) > W_0)
        prob_below_floor: P(W(t) < floor)
        time_horizon: Forecast horizon (in bets or time units)
    """
    expected_wealth: float
    median_wealth: float
    lower_bound: float
    upper_bound: float
    prob_above_initial: float
    prob_below_floor: float
    time_horizon: float
    confidence_level: float = 0.95


class GBMForecaster:
    """
    Geometric Brownian Motion forecaster for bankroll evolution.
    
    Models long-term bankroll growth using continuous-time approximation.
    Useful for:
        - Confidence intervals on future wealth
        - Probability of reaching goals
        - Probability of hitting drawdown limits
    
    The Core Model:
    ---------------
    
    Under GBM, log-wealth follows:
    
    .. math::
    
        \\log W(t) \\sim N\\left(\\log W_0 + (\\mu - \\frac{\\sigma^2}{2})t, \\sigma^2 t\\right)
    
    So W(t) is log-normally distributed.
    
    Key insight: The drift term (μ - σ²/2) is the **geometric** growth rate,
    which differs from the arithmetic mean return μ. This is why Kelly
    maximizes μ - σ²/2, not μ.
    
    Example:
        >>> # From historical bet returns
        >>> returns = [0.1, -0.05, 0.08, -0.03, 0.12, ...]
        >>> forecaster = GBMForecaster.from_returns(returns)
        >>>
        >>> # Forecast 100 bets ahead
        >>> forecast = forecaster.forecast(
        ...     initial_wealth=10000,
        ...     time_horizon=100,
        ...     floor=2000
        ... )
        >>> print(f"Expected: ${forecast.expected_wealth:.2f}")
        >>> print(f"95% CI: [${forecast.lower_bound:.2f}, ${forecast.upper_bound:.2f}]")
    
    Attributes:
        mu: Drift (expected log return per unit time)
        sigma: Volatility (std dev of log returns)
        geometric_growth: μ - σ²/2 (true growth rate)
    """
    
    def __init__(self, mu: float, sigma: float):
        """
        Initialize GBM forecaster with drift and volatility.
        
        Args:
            mu: Expected log return per time unit (per bet)
            sigma: Standard deviation of log returns
        """
        self.mu = mu
        self.sigma = sigma
    
    @classmethod
    def from_returns(cls, returns: List[float]) -> "GBMForecaster":
        """
        Estimate GBM parameters from historical returns.
        
        Args:
            returns: List of fractional returns (e.g., [0.1, -0.05, ...])
                    where 0.1 means +10% and -0.05 means -5%
        
        Returns:
            Fitted GBMForecaster
        """
        returns_arr = np.array(returns)
        
        # Convert to log returns
        log_returns = np.log(1 + returns_arr)
        
        # Estimate parameters
        mu = np.mean(log_returns)
        sigma = np.std(log_returns, ddof=1)
        
        return cls(mu=mu, sigma=sigma)
    
    @classmethod
    def from_kelly_params(
        cls,
        win_prob: float,
        net_odds: float,
        kelly_fraction: float
    ) -> "GBMForecaster":
        """
        Create GBM forecaster from Kelly betting parameters.
        
        This derives the theoretical drift and volatility from
        the Kelly strategy parameters.
        
        For a Kelly bet:
            - Win: wealth *= (1 + f*b)
            - Lose: wealth *= (1 - f)
        
        So:
            - E[log return] = p*log(1+fb) + q*log(1-f)
            - Var[log return] = p*q*(log(1+fb) - log(1-f))²
        
        Args:
            win_prob: Probability of winning
            net_odds: Net odds (decimal_odds - 1)
            kelly_fraction: Fraction of bankroll wagered
        
        Returns:
            GBMForecaster with theoretical parameters
        """
        p = win_prob
        q = 1 - p
        f = kelly_fraction
        b = net_odds
        
        # Log returns
        log_win = np.log(1 + f * b)
        log_lose = np.log(1 - f)
        
        # Expected log return (Kelly growth rate)
        mu = p * log_win + q * log_lose
        
        # Variance of log return
        var = p * q * (log_win - log_lose) ** 2
        sigma = np.sqrt(var)
        
        return cls(mu=mu, sigma=sigma)
    
    @property
    def geometric_growth(self) -> float:
        """
        Geometric growth rate per time unit.
        
        This is μ - σ²/2, which is the expected log-wealth growth rate.
        Under Kelly betting, this is maximized.
        """
        return self.mu - (self.sigma ** 2) / 2
    
    @property
    def doubling_time(self) -> float:
        """
        Expected number of bets to double the bankroll.
        
        .. math::
        
            t_{double} = \\frac{\\log 2}{\\mu - \\sigma^2/2}
        """
        g = self.geometric_growth
        if g <= 0:
            return float('inf')
        return np.log(2) / g
    
    def forecast(
        self,
        initial_wealth: float,
        time_horizon: float,
        floor: float = 0.0,
        confidence: float = 0.95
    ) -> GBMForecast:
        """
        Forecast future bankroll distribution.
        
        Args:
            initial_wealth: Starting bankroll (W_0)
            time_horizon: Number of time units (bets) to forecast
            floor: Wealth floor for probability calculation
            confidence: Confidence level for interval
        
        Returns:
            GBMForecast with distribution statistics
        """
        t = time_horizon
        W0 = initial_wealth
        
        # Log-normal parameters for log(W(t))
        # log(W(t)) ~ N(log(W0) + (μ - σ²/2)t, σ²t)
        mean_log = np.log(W0) + (self.mu - self.sigma**2 / 2) * t
        var_log = self.sigma**2 * t
        std_log = np.sqrt(var_log)
        
        # Expected value (arithmetic mean of log-normal)
        # E[W(t)] = W0 * exp(μt)
        expected = W0 * np.exp(self.mu * t)
        
        # Median (50th percentile of log-normal = exp(mean_log))
        median = np.exp(mean_log)
        
        # Confidence interval
        z = stats.norm.ppf((1 + confidence) / 2)
        lower = np.exp(mean_log - z * std_log)
        upper = np.exp(mean_log + z * std_log)
        
        # Probability above initial
        # P(W(t) > W0) = P(log(W(t)) > log(W0))
        z_initial = (np.log(W0) - mean_log) / std_log
        prob_above = 1 - stats.norm.cdf(z_initial)
        
        # Probability below floor
        if floor > 0:
            z_floor = (np.log(floor) - mean_log) / std_log
            prob_below = stats.norm.cdf(z_floor)
        else:
            prob_below = 0.0
        
        return GBMForecast(
            expected_wealth=expected,
            median_wealth=median,
            lower_bound=lower,
            upper_bound=upper,
            prob_above_initial=prob_above,
            prob_below_floor=prob_below,
            time_horizon=t,
            confidence_level=confidence
        )
    
    def simulate_paths(
        self,
        initial_wealth: float,
        time_horizon: int,
        n_paths: int = 1000
    ) -> np.ndarray:
        """
        Monte Carlo simulation of bankroll paths.
        
        Args:
            initial_wealth: Starting bankroll
            time_horizon: Number of time steps
            n_paths: Number of simulation paths
        
        Returns:
            Array of shape (n_paths, time_horizon+1) with wealth trajectories
        """
        dt = 1  # One time unit per step
        
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, time_horizon))
        
        # Log wealth increments
        # d(log W) = (μ - σ²/2)dt + σ dW
        d_log_W = (self.mu - self.sigma**2 / 2) * dt + self.sigma * dW
        
        # Cumulative log wealth
        log_W = np.zeros((n_paths, time_horizon + 1))
        log_W[:, 0] = np.log(initial_wealth)
        log_W[:, 1:] = np.log(initial_wealth) + np.cumsum(d_log_W, axis=1)
        
        return np.exp(log_W)
    
    def time_to_goal(
        self,
        initial_wealth: float,
        goal_wealth: float,
        confidence: float = 0.5
    ) -> float:
        """
        Estimate time to reach a wealth goal.
        
        Returns the time at which P(W(t) > goal) = confidence.
        
        For median (50%), this is simply:
        
        .. math::
        
            t = \\frac{\\log(goal/W_0)}{\\mu - \\sigma^2/2}
        
        Args:
            initial_wealth: Starting bankroll
            goal_wealth: Target bankroll
            confidence: Probability level (0.5 = median time)
        
        Returns:
            Estimated time in betting units
        """
        if goal_wealth <= initial_wealth:
            return 0.0
        
        g = self.geometric_growth
        if g <= 0:
            return float('inf')
        
        # For median (50%)
        if confidence == 0.5:
            return np.log(goal_wealth / initial_wealth) / g
        
        # General case requires solving for t in:
        # P(log(W(t)) > log(goal)) = confidence
        # 
        # log(goal) = mean_log + z * std_log = log(W0) + gt + z*σ*√t
        # 
        # This is a quadratic in √t, solve numerically
        
        log_ratio = np.log(goal_wealth / initial_wealth)
        z = stats.norm.ppf(confidence)
        
        # Solve: gt - z*σ*√t - log_ratio = 0
        # Let u = √t, then: g*u² - z*σ*u - log_ratio = 0
        a = g
        b = -z * self.sigma
        c = -log_ratio
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return float('inf')
        
        u = (-b + np.sqrt(discriminant)) / (2*a)
        return u**2 if u > 0 else float('inf')


def calculate_ruin_probability(
    win_prob: float,
    net_odds: float,
    kelly_fraction: float,
    n_bets: int = 1000,
    ruin_threshold: float = 0.01
) -> float:
    """
    Calculate probability of ruin over N bets.
    
    "Ruin" is defined as wealth falling below ruin_threshold times
    the initial wealth.
    
    This uses Monte Carlo simulation as the analytical formula is
    complex for fractional Kelly betting.
    
    Args:
        win_prob: Probability of winning each bet
        net_odds: Net odds (decimal_odds - 1)
        kelly_fraction: Fraction of bankroll wagered each bet
        n_bets: Number of bets to simulate
        ruin_threshold: Fraction of initial wealth that defines ruin
    
    Returns:
        Probability of hitting ruin (0 to 1)
    
    Example:
        >>> # 55% win rate, even money, quarter Kelly
        >>> ruin_prob = calculate_ruin_probability(0.55, 1.0, 0.025, n_bets=500)
        >>> print(f"Ruin probability: {ruin_prob:.2%}")
    """
    n_simulations = 10000
    initial_wealth = 1.0
    ruin_count = 0
    
    for _ in range(n_simulations):
        wealth = initial_wealth
        
        for _ in range(n_bets):
            stake = wealth * kelly_fraction
            
            if np.random.random() < win_prob:
                wealth += stake * net_odds
            else:
                wealth -= stake
            
            if wealth < initial_wealth * ruin_threshold:
                ruin_count += 1
                break
    
    return ruin_count / n_simulations


def calculate_risk_metrics(
    returns: List[float],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 365
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics from return history.
    
    Args:
        returns: List of fractional returns per bet
        risk_free_rate: Annual risk-free rate (for Sharpe/Sortino)
        annualization_factor: Number of bets per year (for annualization)
    
    Returns:
        RiskMetrics object
    """
    returns_arr = np.array(returns)
    n = len(returns_arr)
    
    if n == 0:
        return RiskMetrics(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            kelly_growth=0.0,
            win_rate=0.0,
            avg_odds=0.0
        )
    
    # Basic stats
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1) if n > 1 else 0.0
    
    # Sharpe Ratio (annualized)
    excess_return = mean_return - risk_free_rate / annualization_factor
    sharpe = (excess_return / std_return * np.sqrt(annualization_factor)
             if std_return > 0 else 0.0)
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns_arr[returns_arr < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
    sortino = (excess_return / downside_std * np.sqrt(annualization_factor)
              if downside_std > 0 else 0.0)
    
    # Drawdown analysis
    cumulative = np.cumprod(1 + returns_arr)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / peak
    max_drawdown = np.max(drawdowns)
    current_drawdown = drawdowns[-1]
    
    # Kelly growth (expected log return)
    log_returns = np.log(1 + returns_arr)
    kelly_growth = np.mean(log_returns)
    
    # Win rate
    win_rate = np.mean(returns_arr > 0)
    
    # Average odds (inferred from returns)
    # For wins: return = stake * (odds - 1), assume stake = 1
    winning_returns = returns_arr[returns_arr > 0]
    avg_odds = np.mean(winning_returns) + 1 if len(winning_returns) > 0 else 2.0
    
    return RiskMetrics(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        current_drawdown=current_drawdown,
        kelly_growth=kelly_growth,
        win_rate=win_rate,
        avg_odds=avg_odds
    )


def optimal_kelly_for_target_drawdown(
    win_prob: float,
    net_odds: float,
    max_acceptable_drawdown: float = 0.20,
    drawdown_probability: float = 0.05
) -> float:
    """
    Find Kelly fraction that limits drawdown to acceptable level.
    
    Rather than using full or fixed fractional Kelly, this finds the
    fraction that ensures P(drawdown > max_acceptable) ≈ drawdown_probability.
    
    Uses the approximation that max drawdown scales with Kelly fraction.
    
    Args:
        win_prob: Win probability
        net_odds: Net odds
        max_acceptable_drawdown: Maximum acceptable drawdown (e.g., 0.20 = 20%)
        drawdown_probability: Acceptable probability of exceeding max drawdown
    
    Returns:
        Recommended Kelly fraction
    """
    # Full Kelly fraction
    q = 1 - win_prob
    full_kelly = (win_prob * net_odds - q) / net_odds
    
    if full_kelly <= 0:
        return 0.0
    
    # Approximate max drawdown at full Kelly
    # Rough approximation: max_dd ≈ 2 * σ² / μ where σ and μ are from Kelly
    log_win = np.log(1 + full_kelly * net_odds)
    log_lose = np.log(1 - full_kelly)
    
    mu = win_prob * log_win + q * log_lose
    var = win_prob * q * (log_win - log_lose) ** 2
    
    # Expected max drawdown roughly proportional to variance
    expected_max_dd = 2 * var / max(mu, 0.001)
    
    # Scale Kelly to achieve target drawdown
    if expected_max_dd > 0:
        scale = min(1.0, max_acceptable_drawdown / expected_max_dd)
    else:
        scale = 1.0
    
    # Further reduce by drawdown probability confidence
    confidence_scale = 1 - drawdown_probability
    
    return full_kelly * scale * confidence_scale

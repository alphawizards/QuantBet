"""
Statistical Robustness Module.

Provides rigorous statistical validation for backtesting results:
- Non-parametric significance tests (appropriate for non-normal betting returns)
- Block bootstrap confidence intervals (preserves temporal dependence)
- Power analysis for sample size validation
- Multiple testing correction

Based on data-scientist.md standards for statistical rigor.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum


# ============================================================================
# Configuration and Types
# ============================================================================

class SignificanceLevel(Enum):
    """Common significance levels."""
    ALPHA_10 = 0.10
    ALPHA_05 = 0.05
    ALPHA_01 = 0.01


@dataclass
class SampleSizeRequirements:
    """Minimum sample sizes for reliable metrics."""
    # Based on power analysis for detecting 5% edge with 80% power
    roi_significance: int = 385
    win_rate_reliable: int = 96
    sharpe_comparison: int = 400
    brier_score: int = 100
    calibration: int = 200
    
    @classmethod
    def validate(cls, n_samples: int, metric: str) -> Tuple[bool, str]:
        """Check if sample size is sufficient for a metric."""
        requirements = cls()
        thresholds = {
            'roi': requirements.roi_significance,
            'win_rate': requirements.win_rate_reliable,
            'sharpe': requirements.sharpe_comparison,
            'brier': requirements.brier_score,
            'calibration': requirements.calibration,
        }
        
        min_required = thresholds.get(metric.lower(), 100)
        
        if n_samples < min_required:
            return False, f"{metric} unreliable: {n_samples} bets (need {min_required}+)"
        return True, f"{metric} sample size adequate: {n_samples} bets"


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant_at_05: bool
    significant_at_01: bool
    effect_size: Optional[float] = None
    effect_size_interpretation: str = ""
    sample_size: int = 0
    sample_adequate: bool = True
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def __str__(self) -> str:
        sig_str = "✓" if self.significant_at_05 else "✗"
        ci_str = f" (95% CI: {self.confidence_interval[0]:.4f} to {self.confidence_interval[1]:.4f})" if self.confidence_interval else ""
        return f"{self.test_name}: p={self.p_value:.4f} {sig_str}{ci_str}"


@dataclass 
class RobustnessReport:
    """Complete statistical robustness assessment."""
    profitability_test: StatisticalTestResult
    roi_confidence_interval: Tuple[float, float]
    sample_size_checks: Dict[str, Tuple[bool, str]]
    effect_sizes: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_statistically_significant(self) -> bool:
        return self.profitability_test.significant_at_05
    
    @property
    def is_sample_adequate(self) -> bool:
        return all(check[0] for check in self.sample_size_checks.values())
    
    def __str__(self) -> str:
        lines = [
            "═" * 60,
            "         STATISTICAL ROBUSTNESS REPORT",
            "═" * 60,
            "",
            "📊 PROFITABILITY SIGNIFICANCE",
            f"   {self.profitability_test}",
            f"   Effect Size (Cohen's d): {self.profitability_test.effect_size:.3f} ({self.profitability_test.effect_size_interpretation})",
            "",
            "📈 ROI CONFIDENCE INTERVAL (95%)",
            f"   Lower: {self.roi_confidence_interval[0]:.2%}",
            f"   Upper: {self.roi_confidence_interval[1]:.2%}",
            "",
            "📏 SAMPLE SIZE ADEQUACY",
        ]
        
        for metric, (adequate, msg) in self.sample_size_checks.items():
            status = "✓" if adequate else "⚠"
            lines.append(f"   {status} {msg}")
        
        if self.warnings:
            lines.append("")
            lines.append("⚠️ WARNINGS")
            for w in self.warnings:
                lines.append(f"   - {w}")
        
        lines.extend(["", "═" * 60])
        return "\n".join(lines)


# ============================================================================
# Non-Parametric Significance Tests
# ============================================================================

def test_profitability_wilcoxon(returns: pd.Series) -> StatisticalTestResult:
    """
    Test if strategy is statistically profitable using Wilcoxon signed-rank test.
    
    Non-parametric alternative to t-test - does NOT assume normality.
    Appropriate for betting returns which have heavy tails.
    
    H0: Median return = 0 (no profitability)
    H1: Median return > 0 (profitable)
    
    Args:
        returns: Series of per-bet returns (profit/stake)
    
    Returns:
        StatisticalTestResult with significance information
    """
    if len(returns) < 10:
        return StatisticalTestResult(
            test_name="Wilcoxon Signed-Rank",
            statistic=0,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
            sample_size=len(returns),
            sample_adequate=False
        )
    
    # Remove zeros for Wilcoxon
    non_zero = returns[returns != 0]
    
    if len(non_zero) < 10:
        return StatisticalTestResult(
            test_name="Wilcoxon Signed-Rank",
            statistic=0,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
            sample_size=len(returns),
            sample_adequate=False
        )
    
    # One-sided test (greater than 0)
    statistic, p_value_two_sided = stats.wilcoxon(non_zero, alternative='greater')
    
    # Effect size: r = Z / sqrt(N)
    # Z approximation for Wilcoxon
    n = len(non_zero)
    mean = n * (n + 1) / 4
    std = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (statistic - mean) / std if std > 0 else 0
    effect_size = abs(z) / np.sqrt(n)
    
    # Interpret effect size (r)
    if effect_size < 0.1:
        interpretation = "negligible"
    elif effect_size < 0.3:
        interpretation = "small"
    elif effect_size < 0.5:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    sample_ok, _ = SampleSizeRequirements.validate(len(returns), 'roi')
    
    return StatisticalTestResult(
        test_name="Wilcoxon Signed-Rank",
        statistic=statistic,
        p_value=p_value_two_sided,
        significant_at_05=p_value_two_sided < 0.05,
        significant_at_01=p_value_two_sided < 0.01,
        effect_size=effect_size,
        effect_size_interpretation=interpretation,
        sample_size=len(returns),
        sample_adequate=sample_ok
    )


def test_profitability_sign(returns: pd.Series) -> StatisticalTestResult:
    """
    Sign test for profitability - even more robust than Wilcoxon.
    
    Only tests if there are more positive than negative returns.
    Makes minimal assumptions about return distribution.
    
    Args:
        returns: Series of per-bet returns
    
    Returns:
        StatisticalTestResult
    """
    n_positive = (returns > 0).sum()
    n_negative = (returns < 0).sum()
    n_total = n_positive + n_negative
    
    if n_total < 10:
        return StatisticalTestResult(
            test_name="Sign Test",
            statistic=n_positive,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
            sample_size=len(returns),
            sample_adequate=False
        )
    
    # Binomial test: H0 is p=0.5, H1 is p>0.5
    result = stats.binomtest(n_positive, n_total, p=0.5, alternative='greater')
    
    # Effect size: proportion difference from 0.5
    effect_size = (n_positive / n_total) - 0.5 if n_total > 0 else 0
    
    if abs(effect_size) < 0.05:
        interpretation = "negligible"
    elif abs(effect_size) < 0.15:
        interpretation = "small"
    elif abs(effect_size) < 0.25:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return StatisticalTestResult(
        test_name="Sign Test",
        statistic=n_positive,
        p_value=result.pvalue,
        significant_at_05=result.pvalue < 0.05,
        significant_at_01=result.pvalue < 0.01,
        effect_size=effect_size,
        effect_size_interpretation=interpretation,
        sample_size=len(returns),
        sample_adequate=n_total >= 50
    )


def test_runs_randomness(returns: pd.Series) -> StatisticalTestResult:
    """
    Wald-Wolfowitz runs test for randomness of returns.
    
    Tests if the sequence of wins/losses is random.
    Non-random patterns might indicate model issues or market adaptation.
    
    Args:
        returns: Series of per-bet returns
    
    Returns:
        StatisticalTestResult indicating if returns are random
    """
    # Convert to binary: 1 = profit, 0 = loss
    binary = (returns > 0).astype(int).values
    
    if len(binary) < 20:
        return StatisticalTestResult(
            test_name="Runs Test",
            statistic=0,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
            sample_size=len(returns),
            sample_adequate=False
        )
    
    # Count runs
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1
    
    n1 = binary.sum()  # Number of 1s
    n0 = len(binary) - n1  # Number of 0s
    
    if n1 == 0 or n0 == 0:
        return StatisticalTestResult(
            test_name="Runs Test",
            statistic=runs,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
            sample_size=len(returns),
            sample_adequate=False
        )
    
    # Expected runs and variance under randomness
    expected_runs = (2 * n1 * n0) / (n1 + n0) + 1
    variance = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
    
    if variance <= 0:
        return StatisticalTestResult(
            test_name="Runs Test",
            statistic=runs,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
            sample_size=len(returns),
            sample_adequate=False
        )
    
    z = (runs - expected_runs) / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed
    
    return StatisticalTestResult(
        test_name="Runs Test (Randomness)",
        statistic=runs,
        p_value=p_value,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        effect_size=z,
        effect_size_interpretation="non-random pattern" if p_value < 0.05 else "random",
        sample_size=len(returns),
        sample_adequate=len(returns) >= 30
    )


# ============================================================================
# Block Bootstrap Confidence Intervals
# ============================================================================

def block_bootstrap_ci(
    data: pd.Series,
    metric_func: Callable[[pd.Series], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    block_size: int = 10
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval using block bootstrap.
    
    Block bootstrap preserves temporal dependence in the data,
    which is critical for betting returns (winning/losing streaks).
    
    Args:
        data: Time series data (betting returns)
        metric_func: Function to compute the metric (e.g., np.mean for ROI)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (0.95 = 95%)
        block_size: Size of contiguous blocks to resample
    
    Returns:
        Tuple of (lower_bound, upper_bound, point_estimate)
    """
    if len(data) < block_size * 2:
        block_size = max(3, len(data) // 4)
    
    n = len(data)
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        # Generate block bootstrap sample
        indices = []
        while len(indices) < n:
            # Random starting point for block
            start = np.random.randint(0, max(1, n - block_size + 1))
            # Add block indices (with wrap-around)
            for j in range(block_size):
                indices.append((start + j) % n)
        
        # Truncate to original length
        indices = indices[:n]
        sample = data.iloc[indices]
        
        try:
            estimate = metric_func(sample)
            if np.isfinite(estimate):
                bootstrap_estimates.append(estimate)
        except Exception:
            continue
    
    if len(bootstrap_estimates) < 100:
        # Fallback to point estimate if bootstrap fails
        point_est = metric_func(data)
        return point_est, point_est, point_est
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
    upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)
    point = metric_func(data)
    
    return lower, upper, point


def bootstrap_roi_ci(
    returns: pd.Series,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    block_size: int = 10
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for ROI using block bootstrap.
    
    Args:
        returns: Series of per-bet returns (profit/stake)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        block_size: Block size for temporal dependence
    
    Returns:
        Tuple of (lower_ci, upper_ci, point_estimate)
    """
    return block_bootstrap_ci(
        data=returns,
        metric_func=lambda x: x.mean(),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        block_size=block_size
    )


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    annualization: float = 150
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for Sharpe ratio.
    
    Args:
        returns: Series of per-bet returns
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        annualization: Annualization factor
    
    Returns:
        Tuple of (lower_ci, upper_ci, point_estimate)
    """
    def sharpe_func(r):
        if r.std() == 0:
            return 0
        return (r.mean() / r.std()) * np.sqrt(annualization)
    
    return block_bootstrap_ci(
        data=returns,
        metric_func=sharpe_func,
        n_bootstrap=n_bootstrap,
        confidence=confidence
    )


# ============================================================================
# Power Analysis and Sample Size
# ============================================================================

def calculate_required_sample_size(
    effect_size: float = 0.05,
    power: float = 0.80,
    alpha: float = 0.05,
    std_estimate: float = 0.15
) -> int:
    """
    Calculate required sample size to detect a given effect with specified power.
    
    For betting, effect_size is the expected ROI (e.g., 5% = 0.05).
    
    Uses approximation: n ≈ ((z_alpha + z_beta) * std / effect)^2
    
    Args:
        effect_size: Expected effect (ROI) to detect
        power: Statistical power (1 - Type II error rate)
        alpha: Significance level (Type I error rate)
        std_estimate: Estimated standard deviation of returns
    
    Returns:
        Required sample size
    """
    if effect_size <= 0:
        return float('inf')
    
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)
    
    n = ((z_alpha + z_beta) * std_estimate / effect_size) ** 2
    
    return int(np.ceil(n))


def power_for_sample_size(
    n: int,
    effect_size: float = 0.05,
    alpha: float = 0.05,
    std_estimate: float = 0.15
) -> float:
    """
    Calculate statistical power for a given sample size.
    
    Args:
        n: Sample size (number of bets)
        effect_size: Expected effect (ROI) to detect  
        alpha: Significance level
        std_estimate: Estimated standard deviation of returns
    
    Returns:
        Statistical power (0-1)
    """
    if n <= 0 or effect_size <= 0:
        return 0.0
    
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se = std_estimate / np.sqrt(n)
    
    if se == 0:
        return 1.0
    
    z = effect_size / se
    power = stats.norm.cdf(z - z_alpha) + stats.norm.cdf(-z - z_alpha)
    
    return min(1.0, max(0.0, power))


# ============================================================================
# Multiple Testing Correction
# ============================================================================

def correct_multiple_tests(
    p_values: List[float],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction to p-values.
    
    Important when comparing multiple models/strategies to avoid false positives.
    
    Args:
        p_values: List of p-values from multiple tests
        method: Correction method:
            - 'bonferroni': Conservative, controls FWER
            - 'fdr_bh': Benjamini-Hochberg, controls FDR (recommended)
            - 'holm': Holm-Bonferroni step-down
        alpha: Significance level
    
    Returns:
        Tuple of (reject_array, corrected_p_values)
    """
    try:
        from statsmodels.stats.multitest import multipletests
        reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
        return reject, p_corrected
    except ImportError:
        # Fallback: simple Bonferroni
        n = len(p_values)
        corrected = np.array(p_values) * n
        corrected = np.minimum(corrected, 1.0)
        reject = corrected < alpha
        return reject, corrected


def compare_strategies_statistical(
    returns_a: pd.Series,
    returns_b: pd.Series
) -> Dict:
    """
    Statistical comparison of two betting strategies.
    
    Uses Mann-Whitney U test (non-parametric) to compare distributions.
    
    Args:
        returns_a: Returns from strategy A
        returns_b: Returns from strategy B
    
    Returns:
        Dictionary with comparison results
    """
    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(returns_a, returns_b, alternative='two-sided')
    
    # Effect size (Vargha-Delaney A measure)
    n1, n2 = len(returns_a), len(returns_b)
    a12 = statistic / (n1 * n2)  # Probability A > B
    
    if a12 < 0.44:
        effect_interpretation = "B is better (medium-large)"
    elif a12 < 0.50:
        effect_interpretation = "B is slightly better"
    elif a12 < 0.56:
        effect_interpretation = "negligible difference"
    elif a12 < 0.64:
        effect_interpretation = "A is slightly better"
    else:
        effect_interpretation = "A is better (medium-large)"
    
    return {
        'test_name': 'Mann-Whitney U',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size_a12': a12,
        'interpretation': effect_interpretation,
        'mean_a': returns_a.mean(),
        'mean_b': returns_b.mean(),
        'sample_a': len(returns_a),
        'sample_b': len(returns_b)
    }


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_robustness(
    returns: pd.Series,
    stakes: Optional[pd.Series] = None,
    metric_name: str = "Strategy"
) -> RobustnessReport:
    """
    Complete statistical robustness analysis of backtesting results.
    
    Performs:
    1. Non-parametric profitability test (Wilcoxon)
    2. Block bootstrap confidence intervals for ROI
    3. Sample size adequacy checks
    4. Effect size calculations
    
    Args:
        returns: Series of per-bet returns (profit/stake)
        stakes: Optional series of stake amounts
        metric_name: Name for reporting
    
    Returns:
        RobustnessReport with full analysis
    """
    warnings_list = []
    
    # 1. Test profitability significance
    profit_test = test_profitability_wilcoxon(returns)
    
    # 2. Bootstrap CI for ROI
    roi_lower, roi_upper, roi_point = bootstrap_roi_ci(returns)
    
    # Add CI to test result
    profit_test.confidence_interval = (roi_lower, roi_upper)
    
    # 3. Sample size checks
    sample_checks = {}
    for metric in ['roi', 'win_rate', 'sharpe', 'brier']:
        adequate, msg = SampleSizeRequirements.validate(len(returns), metric)
        sample_checks[metric] = (adequate, msg)
        if not adequate:
            warnings_list.append(f"Sample size may be insufficient for reliable {metric}")
    
    # 4. Calculate power
    current_power = power_for_sample_size(
        n=len(returns),
        effect_size=0.05,  # Detect 5% ROI
        std_estimate=returns.std() if returns.std() > 0 else 0.15
    )
    
    if current_power < 0.80:
        warnings_list.append(
            f"Statistical power is only {current_power:.1%} (need 80%+). "
            f"May miss real effects."
        )
    
    # 5. Check if CI includes zero
    if roi_lower < 0 < roi_upper:
        warnings_list.append(
            "Confidence interval includes zero - profitability uncertain"
        )
    
    # 6. Effect sizes
    effect_sizes = {
        'profitability': profit_test.effect_size or 0,
        'roi_width': roi_upper - roi_lower,
        'statistical_power': current_power
    }
    
    return RobustnessReport(
        profitability_test=profit_test,
        roi_confidence_interval=(roi_lower, roi_upper),
        sample_size_checks=sample_checks,
        effect_sizes=effect_sizes,
        warnings=warnings_list
    )


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_significance_check(returns: pd.Series) -> bool:
    """
    Quick check if returns are statistically significant at p < 0.05.
    
    Args:
        returns: Series of per-bet returns
    
    Returns:
        True if statistically significant profit
    """
    result = test_profitability_wilcoxon(returns)
    return result.significant_at_05 and result.sample_adequate


def format_metric_with_ci(
    point_estimate: float,
    ci_lower: float,
    ci_upper: float,
    as_percentage: bool = True,
    decimals: int = 2
) -> str:
    """
    Format a metric with its confidence interval.
    
    Args:
        point_estimate: Central estimate
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        as_percentage: Format as percentage
        decimals: Decimal places
    
    Returns:
        Formatted string like "8.2% (95% CI: 2.1% - 14.3%)"
    """
    if as_percentage:
        return f"{point_estimate:.{decimals}%} (95% CI: {ci_lower:.{decimals}%} - {ci_upper:.{decimals}%})"
    else:
        return f"{point_estimate:.{decimals}f} (95% CI: {ci_lower:.{decimals}f} - {ci_upper:.{decimals}f})"

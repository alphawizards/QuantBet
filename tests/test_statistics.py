"""
Tests for Statistical Robustness Module.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.statistics import (
    test_profitability_wilcoxon as calc_wilcoxon,
    test_profitability_sign as calc_sign,
    test_runs_randomness as calc_runs,
    block_bootstrap_ci,
    bootstrap_roi_ci,
    calculate_required_sample_size,
    power_for_sample_size,
    correct_multiple_tests,
    compare_strategies_statistical,
    analyze_robustness,
    quick_significance_check,
    format_metric_with_ci,
    SampleSizeRequirements
)


class TestNonParametricTests:
    """Tests for non-parametric significance tests."""
    
    @pytest.fixture
    def returns(self):
        """Standard returns fixture if needed by tests implicitly."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.05, 0.15, 200))

    def test_wilcoxon_profitable_strategy(self):
        """Clearly profitable returns should be significant."""
        np.random.seed(42)
        # Profitable: mean return ~5%
        returns = pd.Series(np.random.normal(0.05, 0.15, 500))
        
        result = calc_wilcoxon(returns)
        
        assert result.test_name == "Wilcoxon Signed-Rank"
        assert result.p_value < 0.05  # Should be significant
        assert result.significant_at_05 == True
        assert result.sample_size == 500
    
    def test_wilcoxon_unprofitable_strategy(self):
        """Zero-mean returns should NOT be significant."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0, 0.15, 200))
        
        result = calc_wilcoxon(returns)
        
        # p-value should be high (not significant)
        assert result.p_value > 0.10
        assert result.significant_at_05 == False
    
    def test_wilcoxon_small_sample(self):
        """Small sample should return inadequate."""
        returns = pd.Series([0.1, 0.2, 0.05])
        
        result = calc_wilcoxon(returns)
        
        assert result.sample_adequate == False
    
    def test_sign_test_profitable(self):
        """Sign test on profitable returns."""
        np.random.seed(42)
        # 70% positive returns
        returns = pd.Series([0.1, -0.05, 0.08, 0.12, -0.02, 0.15, 0.03] * 20)
        
        result = calc_sign(returns)
        
        assert result.test_name == "Sign Test"
        assert result.significant_at_05 == True
    
    def test_runs_test_random(self):
        """Random sequence should pass runs test."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.02, 0.15, 200))
        
        result = calc_runs(returns)
        
        # Random should NOT be significant (fail to reject H0: random)
        assert result.test_name == "Runs Test (Randomness)"
        # High p-value = random (as expected)


class TestBlockBootstrap:
    """Tests for block bootstrap confidence intervals."""
    
    def test_bootstrap_ci_basic(self):
        """Basic bootstrap should return CI."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0.05, 0.10, 200))
        
        lower, upper, point = block_bootstrap_ci(
            data=data,
            metric_func=np.mean,
            n_bootstrap=500
        )
        
        assert lower < point < upper
        assert lower < 0.05 < upper  # True mean should be in CI
    
    def test_bootstrap_roi_ci(self):
        """ROI bootstrap should work."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.03, 0.12, 300))
        
        lower, upper, point = bootstrap_roi_ci(returns, n_bootstrap=500)
        
        assert lower < point < upper
        assert upper - lower > 0  # CI has width
    
    def test_bootstrap_preserves_blocks(self):
        """Block bootstrap should handle small data."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        lower, upper, point = block_bootstrap_ci(
            data=data,
            metric_func=np.mean,
            n_bootstrap=100,
            block_size=3
        )
        
        assert lower <= point <= upper


class TestPowerAnalysis:
    """Tests for power analysis and sample size."""
    
    def test_required_sample_size(self):
        """Sample size calculation for detecting 5% ROI."""
        n = calculate_required_sample_size(
            effect_size=0.05,
            power=0.80,
            alpha=0.05,
            std_estimate=0.15
        )
        
        # Formula: n = ((z_alpha + z_beta) * std / effect)^2
        # With given params: n ≈ ((1.96 + 0.84) * 0.15 / 0.05)^2 ≈ 71
        assert 50 < n < 150

    
    def test_power_increases_with_sample(self):
        """Power should increase with sample size."""
        power_100 = power_for_sample_size(100, effect_size=0.05)
        power_500 = power_for_sample_size(500, effect_size=0.05)
        
        assert power_500 > power_100
    
    def test_power_80_at_calculated_n(self):
        """Power should be ~80% at calculated sample size."""
        n = calculate_required_sample_size(effect_size=0.05, power=0.80)
        actual_power = power_for_sample_size(n, effect_size=0.05)
        
        assert 0.75 < actual_power < 0.85


class TestMultipleTesting:
    """Tests for multiple testing correction."""
    
    def test_bonferroni_correction(self):
        """Bonferroni should be conservative."""
        p_values = [0.03, 0.04, 0.06, 0.10]
        
        reject, corrected = correct_multiple_tests(p_values, method='bonferroni')
        
        # With 4 tests, need p < 0.0125 for Bonferroni
        assert reject[0] == False  # 0.03 * 4 = 0.12 > 0.05
    
    def test_fdr_correction(self):
        """FDR should be less conservative."""
        p_values = [0.01, 0.02, 0.04, 0.10]
        
        reject, corrected = correct_multiple_tests(p_values, method='fdr_bh')
        
        # FDR is less strict
        assert isinstance(reject, np.ndarray)
        assert isinstance(corrected, np.ndarray)


class TestStrategyComparison:
    """Tests for strategy comparison."""
    
    def test_compare_different_strategies(self):
        """Comparison of clearly different strategies."""
        np.random.seed(42)
        returns_a = pd.Series(np.random.normal(0.08, 0.15, 200))  # Better
        returns_b = pd.Series(np.random.normal(0.02, 0.15, 200))  # Worse
        
        result = compare_strategies_statistical(returns_a, returns_b)
        
        assert 'p_value' in result
        assert 'effect_size_a12' in result
        assert result['mean_a'] > result['mean_b']
    
    def test_compare_similar_strategies(self):
        """Similar strategies should have high p-value."""
        np.random.seed(42)
        returns_a = pd.Series(np.random.normal(0.05, 0.15, 200))
        returns_b = pd.Series(np.random.normal(0.05, 0.15, 200))
        
        result = compare_strategies_statistical(returns_a, returns_b)
        
        # Should not be significant
        assert result['p_value'] > 0.10 or abs(result['effect_size_a12'] - 0.5) < 0.1


class TestRobustnessReport:
    """Tests for complete robustness analysis."""
    
    def test_analyze_profitable_strategy(self):
        """Robustness report for profitable strategy."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.06, 0.12, 500))
        
        report = analyze_robustness(returns)
        
        assert report.is_statistically_significant == True
        assert report.roi_confidence_interval[0] < 0.06 < report.roi_confidence_interval[1]
    
    def test_analyze_insufficient_sample(self):
        """Robustness report should warn on small sample."""
        returns = pd.Series(np.random.normal(0.05, 0.15, 50))
        
        report = analyze_robustness(returns)
        
        # Should have sample size warnings
        assert len(report.warnings) > 0
        assert not report.is_sample_adequate
    
    def test_report_string_format(self):
        """Report should format nicely."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.04, 0.12, 300))
        
        report = analyze_robustness(returns)
        report_str = str(report)
        
        assert "STATISTICAL ROBUSTNESS REPORT" in report_str
        assert "ROI CONFIDENCE INTERVAL" in report_str


class TestSampleSizeValidation:
    """Tests for sample size requirements."""
    
    def test_validate_adequate(self):
        """Adequate sample should pass."""
        adequate, msg = SampleSizeRequirements.validate(500, 'roi')
        
        assert adequate == True
        assert "adequate" in msg.lower()
    
    def test_validate_inadequate(self):
        """Inadequate sample should fail."""
        adequate, msg = SampleSizeRequirements.validate(50, 'roi')
        
        assert adequate == False
        assert "unreliable" in msg.lower()


class TestConvenienceFunctions:
    """Tests for utility functions."""
    
    def test_quick_significance_check(self):
        """Quick check should return boolean."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.06, 0.12, 500))
        
        result = quick_significance_check(returns)
        
        assert isinstance(result, bool)
    
    def test_format_metric_with_ci(self):
        """Format should produce readable string."""
        formatted = format_metric_with_ci(0.082, 0.021, 0.143)
        
        assert "8.20%" in formatted
        assert "95% CI" in formatted
        assert "2.10%" in formatted

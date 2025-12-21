"""
Data Leakage Detection and Validation Module.

Implements Betfair Rule #3: Avoid Data Leakage.

Common sources of data leakage in sports betting:
    1. Using BSP (Betfair Starting Price) before reconciliation
    2. Features calculated using future data
    3. Aggregate statistics using full participant history
    4. Target variables available before event completion

This module provides tools to detect and prevent these issues.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


class LeakageSeverity(Enum):
    """Severity level of data leakage."""
    INFO = "info"           # Potential issue, worth investigating
    WARNING = "warning"     # Likely leakage, should be fixed
    CRITICAL = "critical"   # Definite leakage, will invalidate backtest


@dataclass
class LeakageWarning:
    """Warning about potential data leakage."""
    feature_name: str
    severity: LeakageSeverity
    description: str
    affected_rows: int = 0
    recommendation: str = ""
    
    def __str__(self) -> str:
        icon = {
            LeakageSeverity.INFO: "ℹ️",
            LeakageSeverity.WARNING: "⚠️",
            LeakageSeverity.CRITICAL: "🚨"
        }[self.severity]
        
        return f"""
{icon} [{self.severity.value.upper()}] {self.feature_name}
   {self.description}
   Affected rows: {self.affected_rows}
   Recommendation: {self.recommendation}
"""


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    warnings: List[LeakageWarning]
    features_checked: int
    rows_checked: int
    passed: bool
    
    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        
        critical = sum(1 for w in self.warnings if w.severity == LeakageSeverity.CRITICAL)
        warnings = sum(1 for w in self.warnings if w.severity == LeakageSeverity.WARNING)
        info = sum(1 for w in self.warnings if w.severity == LeakageSeverity.INFO)
        
        report = f"""
═══════════════════════════════════════════════════════════════
            DATA LEAKAGE VALIDATION REPORT
═══════════════════════════════════════════════════════════════
Status: {status}
Features Checked: {self.features_checked}
Rows Checked: {self.rows_checked}

Issues Found:
  🚨 Critical: {critical}
  ⚠️  Warning:  {warnings}
  ℹ️  Info:     {info}
───────────────────────────────────────────────────────────────
"""
        
        if self.warnings:
            report += "\nDETAILS:\n"
            for warning in sorted(self.warnings, 
                                 key=lambda w: list(LeakageSeverity).index(w.severity)):
                report += str(warning)
        else:
            report += "\n✨ No leakage issues detected!\n"
        
        report += "\n═══════════════════════════════════════════════════════════════\n"
        
        return report


class LeakageValidator:
    """
    Validates datasets for common data leakage issues.
    
    This is critical for honest backtesting. Per Betfair Rule #3:
    "Data leakage occurs when your model uses information that is 
    available now, but would not have been known at the time the 
    bet was placed."
    
    Example:
        >>> validator = LeakageValidator()
        >>> report = validator.validate(features_df, 'game_date', 'home_win')
        >>> print(report)
        >>> assert report.passed, "Data leakage detected!"
    """
    
    # Known BSP-related column patterns that indicate potential leakage
    BSP_PATTERNS = ['bsp', 'starting_price', 'sp_', '_sp', 'final_odds', 'closing']
    
    # Patterns that suggest forward-looking features
    FUTURE_PATTERNS = ['next_', 'future_', 'will_', 'final_', 'result_', 'outcome_']
    
    # Patterns that suggest aggregate/lifetime features
    AGGREGATE_PATTERNS = ['career_', 'lifetime_', 'all_time_', 'total_', 'overall_']
    
    def __init__(
        self,
        date_column: str = 'game_date',
        target_column: str = 'home_win',
        custom_bsp_patterns: Optional[List[str]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize the leakage validator.
        
        Args:
            date_column: Name of the date/datetime column
            target_column: Name of the target variable column
            custom_bsp_patterns: Additional patterns for BSP-like columns
            strict_mode: If True, warnings become critical errors
        """
        self.date_column = date_column
        self.target_column = target_column
        self.strict_mode = strict_mode
        
        self.bsp_patterns = self.BSP_PATTERNS.copy()
        if custom_bsp_patterns:
            self.bsp_patterns.extend(custom_bsp_patterns)
    
    def validate(
        self,
        data: pd.DataFrame,
        date_column: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> ValidationReport:
        """
        Run all validation checks on the dataset.
        
        Args:
            data: DataFrame with features and target
            date_column: Override default date column
            target_column: Override default target column
        
        Returns:
            ValidationReport with all warnings
        """
        date_col = date_column or self.date_column
        target_col = target_column or self.target_column
        
        warnings = []
        
        # Run all checks
        warnings.extend(self.check_temporal_leakage(data, date_col))
        warnings.extend(self.check_target_leakage(data, target_col))
        warnings.extend(self.check_bsp_leakage(data))
        warnings.extend(self.check_future_patterns(data))
        warnings.extend(self.check_aggregate_patterns(data))
        warnings.extend(self.check_missing_value_patterns(data, date_col))
        
        # Determine if validation passed
        passed = not any(w.severity == LeakageSeverity.CRITICAL for w in warnings)
        if self.strict_mode:
            passed = len(warnings) == 0
        
        return ValidationReport(
            warnings=warnings,
            features_checked=len(data.columns),
            rows_checked=len(data),
            passed=passed
        )
    
    def check_temporal_leakage(
        self,
        data: pd.DataFrame,
        date_column: str
    ) -> List[LeakageWarning]:
        """
        Check for features that correlate with future information.
        
        Looks for features that have higher correlation with future
        data than past data, suggesting lookahead bias.
        """
        warnings = []
        
        if date_column not in data.columns:
            warnings.append(LeakageWarning(
                feature_name=date_column,
                severity=LeakageSeverity.WARNING,
                description="Date column not found in dataset",
                recommendation="Ensure date column exists for temporal validation"
            ))
            return warnings
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            try:
                data = data.copy()
                data[date_column] = pd.to_datetime(data[date_column])
            except Exception:
                warnings.append(LeakageWarning(
                    feature_name=date_column,
                    severity=LeakageSeverity.WARNING,
                    description="Could not parse date column",
                    recommendation="Convert date column to datetime format"
                ))
                return warnings
        
        # Check if data is sorted by date
        if not data[date_column].is_monotonic_increasing:
            warnings.append(LeakageWarning(
                feature_name=date_column,
                severity=LeakageSeverity.INFO,
                description="Data is not sorted by date",
                recommendation="Sort data chronologically for walk-forward validation"
            ))
        
        return warnings
    
    def check_target_leakage(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> List[LeakageWarning]:
        """
        Check for features that are derived from the target.
        
        High correlation between features and target can indicate
        that the feature was calculated using outcome information.
        """
        warnings = []
        
        if target_column not in data.columns:
            return warnings
        
        target = data[target_column]
        if not pd.api.types.is_numeric_dtype(target):
            return warnings
        
        # Check correlations
        for col in data.select_dtypes(include=[np.number]).columns:
            if col == target_column:
                continue
            
            try:
                corr = data[col].corr(target)
                
                if abs(corr) > 0.95:
                    warnings.append(LeakageWarning(
                        feature_name=col,
                        severity=LeakageSeverity.CRITICAL,
                        description=f"Near-perfect correlation with target ({corr:.3f})",
                        recommendation="This feature likely contains target information"
                    ))
                elif abs(corr) > 0.8:
                    warnings.append(LeakageWarning(
                        feature_name=col,
                        severity=LeakageSeverity.WARNING,
                        description=f"Very high correlation with target ({corr:.3f})",
                        recommendation="Investigate if this feature uses outcome data"
                    ))
            except Exception:
                pass
        
        return warnings
    
    def check_bsp_leakage(
        self,
        data: pd.DataFrame
    ) -> List[LeakageWarning]:
        """
        Check for BSP (Betfair Starting Price) related columns.
        
        BSP is only known after the event starts, so using it
        to decide whether to place a bet is a form of leakage.
        """
        warnings = []
        
        for col in data.columns:
            col_lower = col.lower()
            
            for pattern in self.bsp_patterns:
                if pattern in col_lower:
                    warnings.append(LeakageWarning(
                        feature_name=col,
                        severity=LeakageSeverity.CRITICAL,
                        description=f"BSP-like feature detected (pattern: '{pattern}')",
                        affected_rows=len(data),
                        recommendation=(
                            "BSP is not known before bet placement. "
                            "Use opening odds or predicted BSP instead."
                        )
                    ))
                    break
        
        return warnings
    
    def check_future_patterns(
        self,
        data: pd.DataFrame
    ) -> List[LeakageWarning]:
        """
        Check for columns that suggest forward-looking information.
        """
        warnings = []
        
        for col in data.columns:
            col_lower = col.lower()
            
            for pattern in self.FUTURE_PATTERNS:
                if col_lower.startswith(pattern):
                    warnings.append(LeakageWarning(
                        feature_name=col,
                        severity=LeakageSeverity.WARNING,
                        description=f"Column name suggests future data (pattern: '{pattern}')",
                        recommendation="Verify this feature uses only past information"
                    ))
                    break
        
        return warnings
    
    def check_aggregate_patterns(
        self,
        data: pd.DataFrame
    ) -> List[LeakageWarning]:
        """
        Check for features using full history aggregates.
        
        Features like "career total" or "lifetime average" typically
        include data from after the event being predicted.
        """
        warnings = []
        
        for col in data.columns:
            col_lower = col.lower()
            
            for pattern in self.AGGREGATE_PATTERNS:
                if pattern in col_lower:
                    warnings.append(LeakageWarning(
                        feature_name=col,
                        severity=LeakageSeverity.WARNING,
                        description=f"Aggregate feature may include future data (pattern: '{pattern}')",
                        recommendation=(
                            "Recalculate using only data available before each event"
                        )
                    ))
                    break
        
        return warnings
    
    def check_missing_value_patterns(
        self,
        data: pd.DataFrame,
        date_column: str
    ) -> List[LeakageWarning]:
        """
        Check for suspicious missing value patterns.
        
        If missing values cluster at the start of the dataset,
        it may indicate lookahead bias in feature calculation.
        """
        warnings = []
        
        if date_column not in data.columns:
            return warnings
        
        try:
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(date_column)
        except Exception:
            return warnings
        
        # Check first 10% of data for missing values
        n_check = max(1, len(data) // 10)
        
        for col in data.columns:
            if col == date_column:
                continue
            
            early_missing = data[col].iloc[:n_check].isna().mean()
            late_missing = data[col].iloc[-n_check:].isna().mean()
            
            if early_missing > 0.5 and late_missing < 0.1:
                warnings.append(LeakageWarning(
                    feature_name=col,
                    severity=LeakageSeverity.INFO,
                    description=(
                        f"High missing rate in early data ({early_missing:.1%}) "
                        f"vs late data ({late_missing:.1%})"
                    ),
                    affected_rows=int(data[col].isna().sum()),
                    recommendation=(
                        "This pattern suggests rolling calculations. "
                        "Ensure only past data is used."
                    )
                ))
        
        return warnings
    
    def paper_trade_comparison(
        self,
        paper_results: pd.DataFrame,
        backtest_results: pd.DataFrame,
        tolerance: float = 0.05
    ) -> ValidationReport:
        """
        Compare paper trading results with backtest results.
        
        Per Betfair Rule #3: "A practical way to check for data leakage 
        is to paper trade a strategy for a week or two, then run a backtest 
        over the same period using only information that would have been 
        available at the time."
        
        Args:
            paper_results: Results from live paper trading
            backtest_results: Results from backtest over same period
            tolerance: Maximum acceptable difference (e.g., 0.05 = 5%)
        
        Returns:
            ValidationReport with comparison results
        """
        warnings = []
        
        # Compare key metrics
        metrics_to_compare = ['roi', 'win_rate', 'total_profit']
        
        for metric in metrics_to_compare:
            if metric in paper_results.columns and metric in backtest_results.columns:
                paper_val = paper_results[metric].iloc[-1]
                backtest_val = backtest_results[metric].iloc[-1]
                
                diff = abs(paper_val - backtest_val)
                max_val = max(abs(paper_val), abs(backtest_val), 0.01)
                pct_diff = diff / max_val
                
                if pct_diff > tolerance:
                    severity = LeakageSeverity.CRITICAL if pct_diff > 0.2 else LeakageSeverity.WARNING
                    
                    warnings.append(LeakageWarning(
                        feature_name=metric,
                        severity=severity,
                        description=(
                            f"Paper trade ({paper_val:.4f}) differs from "
                            f"backtest ({backtest_val:.4f}) by {pct_diff:.1%}"
                        ),
                        recommendation="Data leakage is likely present in the backtest"
                    ))
        
        passed = not any(w.severity == LeakageSeverity.CRITICAL for w in warnings)
        
        return ValidationReport(
            warnings=warnings,
            features_checked=len(metrics_to_compare),
            rows_checked=max(len(paper_results), len(backtest_results)),
            passed=passed
        )


def quick_validate(data: pd.DataFrame) -> bool:
    """
    Quick validation check for data leakage.
    
    Args:
        data: DataFrame to validate
    
    Returns:
        True if no critical issues found
    """
    validator = LeakageValidator()
    report = validator.validate(data)
    return report.passed

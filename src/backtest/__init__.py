"""
Backtesting Framework for NBL/WNBL Betting Models.

This package provides a rigorous backtesting system implementing
Betfair's 10 Golden Rules of Automation.

Components:
    - engine: Walk-forward backtesting orchestrator
    - simulator: Realistic bet execution simulation
    - metrics: Performance metrics calculation
    - validator: Data leakage detection
    - comparison: Model A/B testing framework
    - report: Report generation
"""

from .engine import BacktestEngine, BacktestResult, BacktestConfig
from .simulator import (
    BetSimulator, 
    SessionResult, 
    StakingStrategy,
    FlatStake,
    PercentageStake,
    FractionalKellyStake,
    BoundedKellyStake
)
from .metrics import BacktestMetrics, calculate_metrics
from .validator import LeakageValidator, LeakageWarning, ValidationReport
from .comparison import ModelComparison, ModelRegistry, ComparisonResult, TournamentResult
from .report import BacktestReport, save_report

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestResult",
    "BacktestConfig",
    # Simulator
    "BetSimulator",
    "SessionResult",
    "StakingStrategy",
    "FlatStake",
    "PercentageStake",
    "FractionalKellyStake",
    "BoundedKellyStake",
    # Metrics
    "BacktestMetrics",
    "calculate_metrics",
    # Validator
    "LeakageValidator",
    "LeakageWarning",
    "ValidationReport",
    # Comparison
    "ModelComparison",
    "ModelRegistry",
    "ComparisonResult",
    "TournamentResult",
    # Report
    "BacktestReport",
    "save_report",
]

"""
Model Comparison and A/B Testing Framework.

Provides tools for comparing multiple betting models with
proper statistical testing to avoid Betfair Rule #4: Do Not Overfit.

Features:
    - Paired statistical tests (t-test, Wilcoxon)
    - Tournament-style model comparison
    - Confidence intervals for metrics
    - Leaderboard generation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Protocol
import logging
import numpy as np
import pandas as pd
from scipy import stats

from .engine import BacktestEngine, BacktestResult, BacktestConfig
from .metrics import BacktestMetrics, compare_metrics
from .simulator import StakingStrategy, FractionalKellyStake


logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Protocol for prediction models."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Predictor":
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class ComparisonResult:
    """
    Result of comparing two models.
    
    Attributes:
        model_a_name: Name of first model
        model_b_name: Name of second model
        model_a_metrics: Metrics for first model
        model_b_metrics: Metrics for second model
        winner: Name of winning model (or "tie")
        p_value: Statistical significance of difference
        confidence_level: Confidence level of test (e.g., 0.95)
        test_used: Name of statistical test used
        metric_differences: Dict of metric name -> (diff, is_significant)
    """
    model_a_name: str
    model_b_name: str
    model_a_metrics: BacktestMetrics
    model_b_metrics: BacktestMetrics
    winner: str
    p_value: float
    confidence_level: float = 0.95
    test_used: str = "paired_t_test"
    metric_differences: Dict[str, Tuple[float, bool]] = field(default_factory=dict)
    
    def __str__(self) -> str:
        sig = "significant" if self.p_value < (1 - self.confidence_level) else "not significant"
        
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MODEL COMPARISON RESULT
╠══════════════════════════════════════════════════════════════════════╣
║
║  {self.model_a_name} vs {self.model_b_name}
║
║  Winner: {self.winner}
║  p-value: {self.p_value:.4f} ({sig})
║  Test: {self.test_used}
║
╚══════════════════════════════════════════════════════════════════════╝

{compare_metrics(self.model_a_metrics, self.model_b_metrics, self.model_a_name, self.model_b_name)}
"""


@dataclass
class LeaderboardEntry:
    """Entry in the model leaderboard."""
    model_name: str
    rank: int
    metrics: BacktestMetrics
    wins: int = 0
    losses: int = 0
    ties: int = 0
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.ties
        return self.wins / total if total > 0 else 0


@dataclass
class TournamentResult:
    """
    Result of tournament-style model comparison.
    
    Attributes:
        leaderboard: Ordered list of models by performance
        pairwise_comparisons: All pairwise comparison results
        best_model: Name of the best performing model
        ranking_metric: Metric used for ranking
    """
    leaderboard: List[LeaderboardEntry]
    pairwise_comparisons: Dict[Tuple[str, str], ComparisonResult]
    best_model: str
    ranking_metric: str = "roi"
    
    def __str__(self) -> str:
        header = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MODEL TOURNAMENT RESULTS
║                    Ranking by: {self.ranking_metric}
╠══════════════════════════════════════════════════════════════════════╣
"""
        
        rows = []
        for entry in self.leaderboard:
            m = entry.metrics
            rows.append(
                f"║ {entry.rank:2d}. {entry.model_name:<20} │ "
                f"ROI: {m.roi:+.2%} │ Sharpe: {m.sharpe_ratio:.2f} │ "
                f"W/L: {entry.wins}/{entry.losses}"
            )
        
        footer = f"""
╠══════════════════════════════════════════════════════════════════════╣
║  🏆 Best Model: {self.best_model}
╚══════════════════════════════════════════════════════════════════════╝
"""
        
        return header + "\n".join(rows) + "\n" + footer


class ModelRegistry:
    """
    Registry for managing multiple prediction models.
    
    Provides a central place to register, retrieve, and manage
    different model versions for comparison.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, Predictor] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        model: Predictor,
        description: str = "",
        version: str = "1.0",
        **metadata
    ) -> None:
        """
        Register a model.
        
        Args:
            name: Unique model name
            model: Prediction model instance
            description: Human-readable description
            version: Model version string
            **metadata: Additional metadata
        """
        self._models[name] = model
        self._metadata[name] = {
            "description": description,
            "version": version,
            "registered_at": pd.Timestamp.now(),
            **metadata
        }
        logger.info(f"Registered model: {name} (v{version})")
    
    def get(self, name: str) -> Predictor:
        """Get a registered model by name."""
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a model."""
        return self._metadata.get(name, {})
    
    def unregister(self, name: str) -> None:
        """Remove a model from the registry."""
        if name in self._models:
            del self._models[name]
            del self._metadata[name]
            logger.info(f"Unregistered model: {name}")


class ModelComparison:
    """
    Framework for comparing betting model performance.
    
    Supports:
        - Pairwise model comparison with statistical tests
        - Tournament-style comparison across multiple models
        - Comparison across different staking strategies
    
    Implements safeguards against Betfair Rule #4 (overfitting)
    by requiring out-of-sample testing and reporting significance.
    """
    
    def __init__(
        self,
        engine: Optional[BacktestEngine] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize model comparison framework.
        
        Args:
            engine: Backtest engine to use
            confidence_level: Confidence level for statistical tests
        """
        self.engine = engine or BacktestEngine()
        self.confidence_level = confidence_level
    
    def compare_models(
        self,
        model_a: Predictor,
        model_b: Predictor,
        name_a: str,
        name_b: str,
        data: pd.DataFrame,
        features: List[str],
        target: str = "home_win",
        staking: Optional[StakingStrategy] = None
    ) -> ComparisonResult:
        """
        Compare two models on the same test data.
        
        Args:
            model_a: First model
            model_b: Second model
            name_a: Name for first model
            name_b: Name for second model
            data: Dataset for backtesting
            features: Feature column names
            target: Target column name
            staking: Staking strategy to use
        
        Returns:
            ComparisonResult with statistical analysis
        """
        staking = staking or FractionalKellyStake(0.25)
        
        # Run backtests
        result_a = self.engine.run_backtest(
            model=model_a,
            data=data,
            features=features,
            target=target,
            staking=staking
        )
        
        result_b = self.engine.run_backtest(
            model=model_b,
            data=data,
            features=features,
            target=target,
            staking=staking
        )
        
        # Extract per-bet returns for statistical test
        returns_a = result_a.session.to_dataframe()['profit']
        returns_b = result_b.session.to_dataframe()['profit']
        
        # Perform paired t-test on returns
        p_value, winner, test_used = self._statistical_comparison(
            returns_a, returns_b, name_a, name_b
        )
        
        # Calculate metric differences
        metric_diffs = self._calculate_metric_differences(
            result_a.metrics, result_b.metrics
        )
        
        return ComparisonResult(
            model_a_name=name_a,
            model_b_name=name_b,
            model_a_metrics=result_a.metrics,
            model_b_metrics=result_b.metrics,
            winner=winner,
            p_value=p_value,
            confidence_level=self.confidence_level,
            test_used=test_used,
            metric_differences=metric_diffs
        )
    
    def tournament(
        self,
        models: Dict[str, Predictor],
        data: pd.DataFrame,
        features: List[str],
        target: str = "home_win",
        staking: Optional[StakingStrategy] = None,
        ranking_metric: str = "roi"
    ) -> TournamentResult:
        """
        Run a tournament comparing multiple models.
        
        Each model is compared against all others, and a leaderboard
        is generated based on the specified ranking metric.
        
        Args:
            models: Dictionary of model_name -> model
            data: Dataset for backtesting
            features: Feature column names
            target: Target column name
            staking: Staking strategy
            ranking_metric: Metric to rank by ('roi', 'sharpe_ratio', etc.)
        
        Returns:
            TournamentResult with leaderboard and comparisons
        """
        staking = staking or FractionalKellyStake(0.25)
        
        # Run backtest for each model
        results: Dict[str, BacktestResult] = {}
        for name, model in models.items():
            logger.info(f"Testing model: {name}")
            results[name] = self.engine.run_backtest(
                model=model,
                data=data,
                features=features,
                target=target,
                staking=staking
            )
        
        # Pairwise comparisons
        comparisons: Dict[Tuple[str, str], ComparisonResult] = {}
        win_counts: Dict[str, Dict[str, int]] = {
            name: {"wins": 0, "losses": 0, "ties": 0}
            for name in models.keys()
        }
        
        model_names = list(models.keys())
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i+1:]:
                # Get returns for comparison
                returns_a = results[name_a].session.to_dataframe()['profit']
                returns_b = results[name_b].session.to_dataframe()['profit']
                
                p_value, winner, test_used = self._statistical_comparison(
                    returns_a, returns_b, name_a, name_b
                )
                
                comparison = ComparisonResult(
                    model_a_name=name_a,
                    model_b_name=name_b,
                    model_a_metrics=results[name_a].metrics,
                    model_b_metrics=results[name_b].metrics,
                    winner=winner,
                    p_value=p_value,
                    test_used=test_used
                )
                
                comparisons[(name_a, name_b)] = comparison
                
                # Update win counts
                if winner == name_a:
                    win_counts[name_a]["wins"] += 1
                    win_counts[name_b]["losses"] += 1
                elif winner == name_b:
                    win_counts[name_b]["wins"] += 1
                    win_counts[name_a]["losses"] += 1
                else:
                    win_counts[name_a]["ties"] += 1
                    win_counts[name_b]["ties"] += 1
        
        # Create leaderboard
        leaderboard = []
        for name, result in results.items():
            entry = LeaderboardEntry(
                model_name=name,
                rank=0,  # Will be set after sorting
                metrics=result.metrics,
                wins=win_counts[name]["wins"],
                losses=win_counts[name]["losses"],
                ties=win_counts[name]["ties"]
            )
            leaderboard.append(entry)
        
        # Sort by ranking metric
        leaderboard.sort(
            key=lambda e: getattr(e.metrics, ranking_metric, 0),
            reverse=True
        )
        
        # Assign ranks
        for i, entry in enumerate(leaderboard):
            entry.rank = i + 1
        
        best_model = leaderboard[0].model_name if leaderboard else ""
        
        return TournamentResult(
            leaderboard=leaderboard,
            pairwise_comparisons=comparisons,
            best_model=best_model,
            ranking_metric=ranking_metric
        )
    
    def _statistical_comparison(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        name_a: str,
        name_b: str
    ) -> Tuple[float, str, str]:
        """
        Perform statistical comparison between two return series.
        
        Returns:
            Tuple of (p_value, winner_name, test_name)
        """
        # Ensure equal length by taking minimum
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a.iloc[:min_len]
        returns_b = returns_b.iloc[:min_len]
        
        if min_len < 30:
            # Use Wilcoxon for small samples
            try:
                stat, p_value = stats.wilcoxon(returns_a, returns_b)
                test_used = "wilcoxon_signed_rank"
            except ValueError:
                # Fallback if Wilcoxon fails
                p_value = 1.0
                test_used = "insufficient_data"
        else:
            # Use paired t-test for larger samples
            stat, p_value = stats.ttest_rel(returns_a, returns_b)
            test_used = "paired_t_test"
        
        # Determine winner
        alpha = 1 - self.confidence_level
        
        if p_value < alpha:
            mean_a = returns_a.mean()
            mean_b = returns_b.mean()
            winner = name_a if mean_a > mean_b else name_b
        else:
            winner = "tie"
        
        return p_value, winner, test_used
    
    def _calculate_metric_differences(
        self,
        metrics_a: BacktestMetrics,
        metrics_b: BacktestMetrics
    ) -> Dict[str, Tuple[float, bool]]:
        """
        Calculate differences between metrics with significance.
        
        Returns:
            Dictionary of metric_name -> (difference, is_significant)
        """
        diffs = {}
        
        compare_metrics_list = [
            'roi', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct',
            'win_rate', 'brier_score'
        ]
        
        for metric in compare_metrics_list:
            val_a = getattr(metrics_a, metric, 0)
            val_b = getattr(metrics_b, metric, 0)
            diff = val_a - val_b
            
            # Simple heuristic for significance
            if abs(diff) > abs(val_a) * 0.1:  # >10% relative difference
                is_sig = True
            else:
                is_sig = False
            
            diffs[metric] = (diff, is_sig)
        
        return diffs


def compare_quick(
    model_a: Predictor,
    model_b: Predictor,
    data: pd.DataFrame,
    features: List[str]
) -> ComparisonResult:
    """
    Quick model comparison with default settings.
    
    Args:
        model_a: First model
        model_b: Second model
        data: Dataset
        features: Feature columns
    
    Returns:
        ComparisonResult
    """
    comparison = ModelComparison()
    return comparison.compare_models(
        model_a=model_a,
        model_b=model_b,
        name_a="Model A",
        name_b="Model B",
        data=data,
        features=features
    )

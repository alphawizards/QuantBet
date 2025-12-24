"""
Four Factors Research Pipeline.

Comprehensive research pipeline for analyzing NBL game prediction metrics
using Dean Oliver's Four Factors. Generates correlation analysis, feature
importance, and model comparison reports.

Usage:
    python scripts/run_four_factors_research.py

Output:
    - data/research/four_factors_nbl_analysis.json
    - data/research/correlation_heatmap.png (if matplotlib available)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy import stats

from src.data.scraper import NBLDataScraper
from src.features.four_factors import (
    FourFactorsAnalyzer,
    FourFactorsCalculator,
    PaceCalculator,
    RollingFourFactors,
    generate_research_output,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Research Pipeline
# =============================================================================

class FourFactorsResearchPipeline:
    """
    End-to-end research pipeline for NBL Four Factors analysis.
    
    Implements the research protocol:
        1. Data Acquisition: Load historical game data
        2. Feature Engineering: Calculate Four Factors differentials
        3. Correlation Analysis: Pearson/Spearman vs win outcomes
        4. Model Fitting: Logistic regression and Random Forest
        5. Backtesting: Rolling window validation
    """
    
    def __init__(
        self,
        output_dir: str = "data/research",
        min_season: str = "2018-2019"
    ):
        """
        Initialize research pipeline.
        
        Args:
            output_dir: Directory for output files
            min_season: Earliest season to include in analysis
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_season = min_season
        
        self.scraper = NBLDataScraper()
        self.analyzer = FourFactorsAnalyzer()
        self.pace_calc = PaceCalculator()
        self.rolling_ff = RollingFourFactors(window=5)
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete research analysis pipeline.
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("=" * 60)
        logger.info("NBL Four Factors Research Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Data Acquisition
        logger.info("\n[1/5] Data Acquisition...")
        games_df = self._load_data()
        
        if games_df is None or len(games_df) == 0:
            logger.error("Failed to load data")
            return {"error": "Data acquisition failed"}
        
        # Step 2: Feature Engineering
        logger.info("\n[2/5] Feature Engineering...")
        games_df = self._engineer_features(games_df)
        
        # Step 3: Correlation Analysis
        logger.info("\n[3/5] Correlation Analysis...")
        correlations = self._analyze_correlations(games_df)
        
        # Step 4: Model Fitting
        logger.info("\n[4/5] Model Fitting...")
        models = self._fit_models(games_df)
        
        # Step 5: Pace Analysis
        logger.info("\n[5/5] NBL Pace Analysis...")
        pace_analysis = self._analyze_pace(games_df)
        
        # Compile results
        results = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "min_season": self.min_season,
                "total_games": len(games_df),
                "seasons_included": sorted(games_df['season'].unique().tolist()) if 'season' in games_df.columns else []
            },
            "data_quality": self._get_data_quality_summary(games_df),
            "correlations": correlations,
            "models": models,
            "pace_analysis": pace_analysis,
            "insights": self._generate_insights(correlations, models, pace_analysis)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load and prepare game data."""
        try:
            games_df = self.scraper.get_four_factors_data(min_season=self.min_season)
            logger.info(f"Loaded {len(games_df)} games")
            return games_df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Four Factors and derived features."""
        # Compute raw Four Factors
        df = self.analyzer.compute_four_factors_for_games(df)
        
        # Add rolling features (this can be slow for large datasets)
        logger.info("  - Calculating rolling Four Factors (may take a moment)...")
        df = self.rolling_ff.add_rolling_features_to_df(df)
        
        # Add pace metrics
        if all(c in df.columns for c in ['home_fga', 'home_orb', 'home_turnovers', 'home_fta']):
            df['game_pace'] = df.apply(
                lambda r: self.pace_calc.calculate_game_pace(
                    r['home_fga'], r['home_orb'], r['home_turnovers'], r['home_fta']
                ).pace,
                axis=1
            )
        
        logger.info(f"  - Engineered {len([c for c in df.columns if 'delta_' in c or 'roll_' in c])} new features")
        return df
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Run correlation analysis on Four Factors."""
        correlations = self.analyzer.calculate_correlation_with_wins(df)
        
        results = {
            "four_factors_vs_wins": [
                {
                    "metric": c.metric_name,
                    "pearson_r": round(c.pearson_r, 4),
                    "pearson_p": round(c.pearson_p, 4),
                    "spearman_rho": round(c.spearman_rho, 4),
                    "significant": c.is_significant
                }
                for c in correlations
            ]
        }
        
        # Print summary
        logger.info("\n  Correlation with Wins:")
        for c in correlations:
            sig = "✓" if c.is_significant else "✗"
            logger.info(f"    {c.metric_name}: r={c.pearson_r:+.3f} {sig}")
        
        # Oliver's weights comparison
        results["olivers_weights"] = {
            "efg_pct": 0.40,
            "tov_pct": 0.25,
            "orb_pct": 0.20,
            "ft_rate": 0.15
        }
        
        # Calculate NBL-specific weights from data
        if correlations:
            total_abs_r = sum(abs(c.pearson_r) for c in correlations)
            if total_abs_r > 0:
                results["nbl_empirical_weights"] = {
                    c.metric_name.replace(" Differential", ""): round(abs(c.pearson_r) / total_abs_r, 3)
                    for c in correlations
                }
        
        return results
    
    def _fit_models(self, df: pd.DataFrame) -> Dict:
        """Fit prediction models and extract coefficients."""
        try:
            logreg = self.analyzer.fit_logistic_model(df, use_cross_validation=True)
            rf = self.analyzer.fit_random_forest_model(df)
            
            return {
                "logistic_regression": {
                    "coefficients": {k: round(v, 4) for k, v in logreg['coefficients'].items()},
                    "intercept": round(logreg['intercept'], 4),
                    "brier_score": round(logreg['brier_score'], 4),
                    "cv_accuracy": round(logreg['cv_accuracy'], 4) if logreg['cv_accuracy'] else None,
                    "cv_brier": round(logreg['cv_brier'], 4) if logreg['cv_brier'] else None,
                    "n_samples": logreg['n_samples'],
                    "feature_importance": [
                        {"feature": k, "importance": round(abs(v), 4)}
                        for k, v in logreg['feature_importance']
                    ]
                },
                "random_forest": {
                    "feature_importance": [
                        {"feature": k, "importance": round(v, 4)}
                        for k, v in rf['feature_importance']
                    ],
                    "n_samples": rf['n_samples']
                }
            }
        except Exception as e:
            logger.warning(f"Model fitting failed: {e}")
            return {"error": str(e)}
    
    def _analyze_pace(self, df: pd.DataFrame) -> Dict:
        """Analyze NBL-specific pace metrics."""
        pace_results = {}
        
        if 'game_pace' in df.columns:
            pace_data = df['game_pace'].dropna()
            
            pace_results = {
                "nbl_average_pace": round(pace_data.mean(), 2),
                "pace_std_dev": round(pace_data.std(), 2),
                "pace_min": round(pace_data.min(), 2),
                "pace_max": round(pace_data.max(), 2),
                "high_pace_threshold": round(pace_data.mean() + pace_data.std(), 2),
                "low_pace_threshold": round(pace_data.mean() - pace_data.std(), 2),
            }
            
            # Pace vs win correlation
            if 'home_win' in df.columns:
                mask = ~(df['game_pace'].isna() | df['home_win'].isna())
                if mask.sum() > 30:
                    r, p = stats.pearsonr(df.loc[mask, 'game_pace'], df.loc[mask, 'home_win'])
                    pace_results["pace_win_correlation"] = {
                        "pearson_r": round(r, 4),
                        "p_value": round(p, 4),
                        "significant": p < 0.05
                    }
            
            logger.info(f"\n  NBL Average Pace: {pace_results['nbl_average_pace']:.1f} possessions/40min")
        
        return pace_results
    
    def _get_data_quality_summary(self, df: pd.DataFrame) -> Dict:
        """Generate data quality summary."""
        quality = self.analyzer.assess_data_quality(df)
        
        return {
            "total_games": quality.total_games,
            "complete_data_pct": round(quality.completeness_pct, 1),
            "date_range": list(quality.date_range),
            "num_teams": len(quality.teams_with_data),
            "teams": quality.teams_with_data,
            "warnings": quality.warnings
        }
    
    def _generate_insights(
        self, 
        correlations: Dict, 
        models: Dict, 
        pace: Dict
    ) -> List[str]:
        """Generate human-readable insights from analysis."""
        insights = []
        
        # Top predictor
        if "four_factors_vs_wins" in correlations and correlations["four_factors_vs_wins"]:
            top = correlations["four_factors_vs_wins"][0]
            insights.append(
                f"Strongest predictor: {top['metric']} (r={top['pearson_r']:+.3f})"
            )
        
        # Model performance
        if "logistic_regression" in models and "brier_score" in models["logistic_regression"]:
            brier = models["logistic_regression"]["brier_score"]
            insights.append(
                f"Logistic regression Brier Score: {brier:.3f} "
                f"({'Good' if brier < 0.22 else 'Moderate' if brier < 0.25 else 'Poor'} calibration)"
            )
        
        # NBL pace context
        if "nbl_average_pace" in pace:
            nbl_pace = pace["nbl_average_pace"]
            insights.append(
                f"NBL average pace: {nbl_pace:.1f} poss/40min "
                f"(compare to NBA ~100 poss/48min)"
            )
        
        # Oliver's weights comparison
        if "nbl_empirical_weights" in correlations:
            weights = correlations["nbl_empirical_weights"]
            if weights:
                top_factor = max(weights.items(), key=lambda x: x[1])
                insights.append(
                    f"NBL empirical top factor: {top_factor[0]} ({top_factor[1]:.0%} of signal)"
                )
        
        return insights
    
    def _save_results(self, results: Dict):
        """Save analysis results to JSON."""
        output_path = self.output_dir / "four_factors_nbl_analysis.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n✓ Results saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the Four Factors research pipeline."""
    print("\n" + "=" * 60)
    print("NBL Four Factors Research Pipeline")
    print("Based on Dean Oliver's Basketball on Paper")
    print("=" * 60 + "\n")
    
    pipeline = FourFactorsResearchPipeline(
        output_dir="data/research",
        min_season="2018-2019"
    )
    
    results = pipeline.run_full_analysis()
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)
    
    if "insights" in results:
        print("\n📊 Key Insights:")
        for insight in results["insights"]:
            print(f"  • {insight}")
    
    if "correlations" in results and "four_factors_vs_wins" in results["correlations"]:
        print("\n📈 Four Factors Correlation with Wins:")
        for factor in results["correlations"]["four_factors_vs_wins"]:
            sig = "✓" if factor["significant"] else "✗"
            print(f"  {factor['metric']}: r={factor['pearson_r']:+.4f} {sig}")
    
    if "models" in results and "logistic_regression" in results["models"]:
        logreg = results["models"]["logistic_regression"]
        if "cv_accuracy" in logreg and logreg["cv_accuracy"]:
            print(f"\n🎯 Model Performance:")
            print(f"  CV Accuracy: {logreg['cv_accuracy']:.1%}")
            print(f"  Brier Score: {logreg['brier_score']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Full results saved to: data/research/four_factors_nbl_analysis.json")
    print("=" * 60 + "\n")
    
    return results


if __name__ == "__main__":
    main()

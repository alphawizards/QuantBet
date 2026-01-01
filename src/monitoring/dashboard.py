"""
Model Monitoring Dashboard for QuantBet.

Real-time monitoring of prediction model performance including:
- Calibration curves
- Brier score tracking
- ROI and Sharpe ratio
- Model drift detection
- Alert thresholds
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Current model performance metrics."""
    timestamp: str
    n_predictions: int
    n_resolved: int           # Predictions with known outcomes
    
    # Calibration metrics
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    
    # Drift indicators (non-optional - must come before optional params)
    prediction_mean: float      # Average predicted probability
    prediction_std: float       # Std of predictions
    calibration_slope: float    # Slope of calibration curve
    
    # Optional metrics with defaults (must come after non-default params)
    auc_roc: Optional[float] = None
    
    # Betting metrics (optional)
    roi: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'n_predictions': self.n_predictions,
            'n_resolved': self.n_resolved,
            'brier_score': round(self.brier_score, 4),
            'log_loss': round(self.log_loss, 4),
            'ece': round(self.expected_calibration_error, 4),
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'auc_roc': round(self.auc_roc, 4) if self.auc_roc else None,
            'roi': round(self.roi, 4) if self.roi else None,
            'sharpe': round(self.sharpe_ratio, 2) if self.sharpe_ratio else None,
            'win_rate': round(self.win_rate, 4) if self.win_rate else None,
            'pred_mean': round(self.prediction_mean, 3),
            'pred_std': round(self.prediction_std, 3),
            'cal_slope': round(self.calibration_slope, 3)
        }


class ModelMonitor:
    """
    Monitors production model performance in real-time.
    
    Loads predictions from production logger and calculates
    comprehensive metrics for dashboard display.
    """
    
    def __init__(
        self,
        prediction_log_dir: str = "data/predictions",
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize model monitor.
        
        Args:
            prediction_log_dir: Directory with prediction logs
            alert_thresholds: Dict of metric -> threshold for alerts
        """
        self.log_dir = Path(prediction_log_dir)
        
        # Default alert thresholds
        self.thresholds = alert_thresholds or {
            'brier_score': 0.25,      # Alert if > 0.25
            'accuracy': 0.48,          # Alert if < 48%
            'ece': 0.10,              # Alert if > 10%
            'roi': -0.05,             # Alert if < -5%
        }
        
        logger.info(f"Model monitor initialized: {self.log_dir}")
    
    def load_predictions(
        self,
        days: int = 7
    ) -> pd.DataFrame:
        """
        Load prediction logs from recent days.
        
        Args:
            days: Number of days to load
        
        Returns:
            DataFrame with all predictions
        """
        all_predictions = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            log_file = self.log_dir / f"{date_str}.jsonl"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        all_predictions.append(json.loads(line))
        
        if not all_predictions:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_predictions)
        logger.info(f"Loaded {len(df)} predictions from last {days} days")
        
        return df
    
    def calculate_expected_calibration_error(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted probabilities
        and actual frequencies across probability bins.
        
        Args:
            predictions: Predicted probabilities
            outcomes: True binary outcomes
            n_bins: Number of probability bins
        
        Returns:
            ECE value (0 = perfect calibration)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            # Find predictions in this bin
            mask = (predictions >= bins[i]) & (predictions < bins[i+1])
            
            if mask.sum() > 0:
                avg_pred = predictions[mask].mean()
                avg_outcome = outcomes[mask].mean()
                bin_weight = mask.sum() / len(predictions)
                
                ece += bin_weight * abs(avg_pred - avg_outcome)
        
        return ece
    
    def calculate_calibration_curve(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate calibration curve data points.
        
        Returns:
            Tuple of (bin_centers, observed_frequencies)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        observed_freqs = []
        
        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i+1])
            
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                observed_freqs.append(outcomes[mask].mean())
        
        return np.array(bin_centers), np.array(observed_freqs)
    
    def calculate_metrics(
        self,
        days: int = 7,
        min_resolved: int = 20
    ) -> Optional[ModelMetrics]:
        """
        Calculate comprehensive model metrics.
        
        Args:
            days: Number of days to analyze
            min_resolved: Minimum resolved predictions needed
        
        Returns:
            ModelMetrics object or None if insufficient data
        """
        df = self.load_predictions(days)
        
        if df.empty:
            logger.warning("No predictions found")
            return None
        
        # Filter to resolved predictions
        resolved = df[df['home_won'].notna()].copy()
        
        if len(resolved) < min_resolved:
            logger.warning(
                f"Insufficient resolved predictions: {len(resolved)} < {min_resolved}"
            )
            return None
        
        # Extract arrays
        y_pred = resolved['predicted_home_prob'].values
        y_true = resolved['home_won'].astype(int).values
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        # Calibration metrics
        try:
            from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
            
            brier = brier_score_loss(y_true, y_pred)
            logloss = log_loss(y_true, y_pred)
            ece = self.calculate_expected_calibration_error(y_pred, y_true)
            
            # Classification metrics
            tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
            fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
            tn = ((y_pred_binary == 0) & (y_true == 0)).sum()
            fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
            
            accuracy = (tp + tn) / len(y_true)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            try:
                auc = roc_auc_score(y_true, y_pred)
            except:
                auc = None
            
            # Betting metrics (if available)
            roi = None
            sharpe = None
            win_rate = None
            
            if 'bet_result' in resolved.columns:
                bet_results = resolved['bet_result'].dropna()
                if len(bet_results) > 0:
                    roi = bet_results.mean()
                    win_rate = (bet_results > 0).mean()
                    
                    if len(bet_results) > 1:
                        sharpe = bet_results.mean() / bet_results.std() * np.sqrt(252)
            
            # Drift indicators
            pred_mean = y_pred.mean()
            pred_std = y_pred.std()
            
            # Calibration slope (perfect = 1.0)
            bin_centers, obs_freqs = self.calculate_calibration_curve(y_pred, y_true)
            if len(bin_centers) > 1:
                slope, _ = np.polyfit(bin_centers, obs_freqs, 1)
            else:
                slope = 1.0
            
            metrics = ModelMetrics(
                timestamp=datetime.now().isoformat(),
                n_predictions=len(df),
                n_resolved=len(resolved),
                brier_score=brier,
                log_loss=logloss,
                expected_calibration_error=ece,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                auc_roc=auc,
                roi=roi,
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                prediction_mean=pred_mean,
                prediction_std=pred_std,
                calibration_slope=slope
            )
            
            # Check thresholds
            self._check_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return None
    
    def _check_alerts(self, metrics: ModelMetrics) -> None:
        """Check if any metrics breach alert thresholds."""
        alerts = []
        
        if metrics.brier_score > self.thresholds['brier_score']:
            alerts.append(
                f"⚠️  HIGH BRIER SCORE: {metrics.brier_score:.3f} > {self.thresholds['brier_score']}"
            )
        
        if metrics.accuracy < self.thresholds['accuracy']:
            alerts.append(
                f"⚠️  LOW ACCURACY: {metrics.accuracy:.1%} < {self.thresholds['accuracy']:.1%}"
            )
        
        if metrics.expected_calibration_error > self.thresholds['ece']:
            alerts.append(
                f"⚠️  POOR CALIBRATION: ECE {metrics.expected_calibration_error:.3f} > {self.thresholds['ece']}"
            )
        
        if metrics.roi is not None and metrics.roi < self.thresholds['roi']:
            alerts.append(
                f"⚠️  NEGATIVE ROI: {metrics.roi:.1%} < {self.thresholds['roi']:.1%}"
            )
        
        if alerts:
            for alert in alerts:
                logger.warning(alert)
        else:
            logger.info("✅ All metrics within acceptable thresholds")
    
    def get_dashboard_data(
        self,
        days: int = 7
    ) -> Dict:
        """
        Get complete dashboard data.
        
        Returns:
            Dictionary with metrics and visualizations
        """
        metrics = self.calculate_metrics(days)
        
        if metrics is None:
            return {'error': 'Insufficient data'}
        
        df = self.load_predictions(days)
        resolved = df[df['home_won'].notna()]
        
        return {
            'metrics': metrics.to_dict(),
            'calibration_curve': self._get_calibration_data(resolved),
            'performance_trend': self._get_trend_data(resolved),
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _get_calibration_data(self, df: pd.DataFrame) -> Dict:
        """Get calibration curve data for plotting."""
        if df.empty:
            return {}
        
        y_pred = df['predicted_home_prob'].values
        y_true = df['home_won'].astype(int).values
        
        bin_centers, obs_freqs = self.calculate_calibration_curve(y_pred, y_true)
        
        return {
            'predicted': bin_centers.tolist(),
            'observed': obs_freqs.tolist()
        }
    
    def _get_trend_data(self, df: pd.DataFrame) -> Dict:
        """Get performance trend over time."""
        if df.empty:
            return {}
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        daily_metrics = []
        
        for date in sorted(df['date'].unique()):
            day_data = df[df['date'] == date]
            
            if len(day_data) >= 5:  # Minimum for meaningful metrics
                y_pred = day_data['predicted_home_prob'].values
                y_true = day_data['home_won'].astype(int).values
                
                try:
                    from sklearn.metrics import brier_score_loss
                    brier = brier_score_loss(y_true, y_pred)
                    accuracy = ((y_pred >= 0.5).astype(int) == y_true).mean()
                    
                    daily_metrics.append({
                        'date': str(date),
                        'brier': brier,
                        'accuracy': accuracy
                    })
                except:
                    pass
        
        return {'daily': daily_metrics}
    
    def _generate_recommendations(self, metrics: ModelMetrics) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        if metrics.brier_score > 0.22:
            recommendations.append(
                "Consider retraining model - Brier score above target"
            )
        
        if metrics.expected_calibration_error > 0.08:
            recommendations.append(
                "Apply isotonic regression calibration to improve probability estimates"
            )
        
        if abs(metrics.calibration_slope - 1.0) > 0.2:
            recommendations.append(
                f"Calibration slope {metrics.calibration_slope:.2f} - model may be over/under-confident"
            )
        
        if metrics.roi is not None and metrics.roi < 0:
            recommendations.append(
                "Negative ROI detected - review bet sizing and edge thresholds"
            )
        
        if not recommendations:
            recommendations.append("✅ Model performance is healthy")
        
        return recommendations


def monitor_production_models():
    """
    Command-line utility to monitor production models.
    
    Usage:
        python -m src.monitoring.dashboard
    """
    monitor = ModelMonitor()
    
    dashboard_data = monitor.get_dashboard_data(days=7)
    
    if 'error' in dashboard_data:
        print(f"❌ {dashboard_data['error']}")
        return
    
    metrics = dashboard_data['metrics']
    
    print("=" * 60)
    print("QUANTBET MODEL MONITORING DASHBOARD")
    print("=" * 60)
    print(f"\nLast Updated: {metrics['timestamp']}")
    print(f"Predictions: {metrics['n_predictions']} total, {metrics['n_resolved']} resolved")
    
    print("\n📊 CALIBRATION METRICS:")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    
    print("\n🎯 CLASSIFICATION METRICS:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    if metrics['auc_roc']:
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    
    if metrics['roi'] is not None:
        print("\n💰 BETTING METRICS:")
        print(f"  ROI: {metrics['roi']:.2%}")
        if metrics['sharpe']:
            print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
        if metrics['win_rate']:
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
    
    print("\n📈 MODEL HEALTH:")
    print(f"  Prediction Mean: {metrics['pred_mean']:.3f}")
    print(f"  Prediction Std: {metrics['pred_std']:.3f}")
    print(f"  Calibration Slope: {metrics['cal_slope']:.3f} (target: 1.0)")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in dashboard_data['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    monitor_production_models()

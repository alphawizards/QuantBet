"""
Backtest Report Generation Module.

Generates HTML and Markdown reports for backtest results.
Useful for documenting model performance and sharing results.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import pandas as pd
import numpy as np

from .engine import BacktestResult
from .comparison import TournamentResult, ComparisonResult
from .metrics import BacktestMetrics


class BacktestReport:
    """
    Generate comprehensive backtest reports.
    
    Supports HTML and Markdown output formats.
    """
    
    def __init__(self):
        """Initialize report generator."""
        pass
    
    def generate_markdown(
        self,
        result: BacktestResult,
        title: str = "Backtest Report"
    ) -> str:
        """
        Generate a Markdown report for a single backtest.
        
        Args:
            result: BacktestResult to report on
            title: Report title
        
        Returns:
            Markdown formatted string
        """
        m = result.metrics
        
        report = f"""# {title}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

| Metric | Value |
|--------|-------|
| **ROI** | {m.roi:+.2%} |
| **Total P/L** | ${m.profit_loss:+,.2f} |
| **Sharpe Ratio** | {m.sharpe_ratio:.2f} |
| **Max Drawdown** | {m.max_drawdown_pct:.2%} |
| **Total Bets** | {m.total_bets:,} |
| **Win Rate** | {m.win_rate:.2%} |

## Training & Testing Periods

- **Train:** {', '.join(result.train_periods)}
- **Test:** {', '.join(result.test_periods)}

## Detailed Metrics

### Profitability
- ROI: {m.roi:+.2%}
- Profit/Loss: ${m.profit_loss:+,.2f}
- Yield per Bet: {m.yield_per_bet:+.2%}
- Total Staked: ${m.total_staked:,.2f}

### Risk-Adjusted Returns
- Sharpe Ratio: {m.sharpe_ratio:.2f}
- Sortino Ratio: {m.sortino_ratio:.2f}
- Calmar Ratio: {m.calmar_ratio:.2f}

### Drawdown Analysis
- Max Drawdown: {m.max_drawdown_pct:.2%} (${m.max_drawdown:,.2f})
- Max DD Duration: {m.max_drawdown_duration} bets
- Current Drawdown: {m.current_drawdown_pct:.2%}

### Win/Loss Statistics
- Win Rate: {m.win_rate:.2%}
- Avg Odds (Winners): {m.avg_odds_winner:.2f}
- Avg Odds (Losers): {m.avg_odds_loser:.2f}
- Profit Factor: {m.profit_factor:.2f}

### Prediction Quality
- Brier Score: {m.brier_score:.4f}
- Log Loss: {m.log_loss:.4f}
- Calibration Error: {m.calibration_error:.4f}

## Equity Curve

The equity curve shows the bankroll evolution over the test period.

```
Start: ${result.session.initial_bankroll:,.2f}
Final: ${result.session.final_bankroll:,.2f}
Peak:  ${result.equity_curve.max():,.2f}
Min:   ${result.equity_curve.min():,.2f}
```
"""
        
        # Add validation report if available
        if result.validation_report:
            if result.validation_report.passed:
                report += "\n## Data Validation ✅\n\nNo data leakage issues detected.\n"
            else:
                report += f"\n## Data Validation ⚠️\n\nIssues found:\n{str(result.validation_report)}\n"
        
        return report
    
    def generate_comparison_markdown(
        self,
        comparison: ComparisonResult
    ) -> str:
        """
        Generate Markdown report for model comparison.
        
        Args:
            comparison: ComparisonResult to report on
        
        Returns:
            Markdown formatted string
        """
        sig = "statistically significant" if comparison.p_value < 0.05 else "not statistically significant"
        
        ma = comparison.model_a_metrics
        mb = comparison.model_b_metrics
        
        report = f"""# Model Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

**Winner: {comparison.winner}**

- p-value: {comparison.p_value:.4f} ({sig})
- Test used: {comparison.test_used}
- Confidence level: {comparison.confidence_level:.0%}

## Side-by-Side Comparison

| Metric | {comparison.model_a_name} | {comparison.model_b_name} | Difference |
|--------|---------------------------|---------------------------|------------|
| ROI | {ma.roi:+.2%} | {mb.roi:+.2%} | {ma.roi - mb.roi:+.2%} |
| Sharpe | {ma.sharpe_ratio:.2f} | {mb.sharpe_ratio:.2f} | {ma.sharpe_ratio - mb.sharpe_ratio:+.2f} |
| Max DD | {ma.max_drawdown_pct:.2%} | {mb.max_drawdown_pct:.2%} | {ma.max_drawdown_pct - mb.max_drawdown_pct:+.2%} |
| Win Rate | {ma.win_rate:.2%} | {mb.win_rate:.2%} | {ma.win_rate - mb.win_rate:+.2%} |
| Brier | {ma.brier_score:.4f} | {mb.brier_score:.4f} | {ma.brier_score - mb.brier_score:+.4f} |

## Interpretation

"""
        
        if comparison.winner != "tie":
            report += f"**{comparison.winner}** outperforms with statistical significance (p={comparison.p_value:.4f}).\n"
        else:
            report += "No statistically significant difference between models.\n"
        
        return report
    
    def generate_tournament_markdown(
        self,
        tournament: TournamentResult
    ) -> str:
        """
        Generate Markdown report for tournament results.
        
        Args:
            tournament: TournamentResult to report on
        
        Returns:
            Markdown formatted string
        """
        report = f"""# Model Tournament Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏆 Winner: {tournament.best_model}

Ranking metric: {tournament.ranking_metric}

## Leaderboard

| Rank | Model | ROI | Sharpe | Max DD | W/L/T |
|------|-------|-----|--------|--------|-------|
"""
        
        for entry in tournament.leaderboard:
            m = entry.metrics
            report += f"| {entry.rank} | {entry.model_name} | {m.roi:+.2%} | {m.sharpe_ratio:.2f} | {m.max_drawdown_pct:.2%} | {entry.wins}/{entry.losses}/{entry.ties} |\n"
        
        report += "\n## Pairwise Comparisons\n\n"
        
        for (name_a, name_b), comp in tournament.pairwise_comparisons.items():
            sig = "✓" if comp.p_value < 0.05 else "✗"
            report += f"- **{name_a}** vs **{name_b}**: Winner = {comp.winner} (p={comp.p_value:.4f}) {sig}\n"
        
        return report
    
    def generate_html(
        self,
        result: BacktestResult,
        title: str = "Backtest Report"
    ) -> str:
        """
        Generate HTML report for a backtest.
        
        Args:
            result: BacktestResult to report on
            title: Report title
        
        Returns:
            HTML formatted string
        """
        m = result.metrics
        
        # Determine status colors
        roi_color = "green" if m.roi > 0 else "red"
        sharpe_color = "green" if m.sharpe_ratio > 1 else "orange" if m.sharpe_ratio > 0 else "red"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #1a1a2e; margin-bottom: 30px; }}
        h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 10px; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 8px;
            color: white;
        }}
        .metric-card.positive {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .metric-card.negative {{ background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card {'positive' if m.roi > 0 else 'negative'}">
                <div class="metric-label">Return on Investment</div>
                <div class="metric-value">{m.roi:+.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{m.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{m.max_drawdown_pct:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{m.win_rate:.2%}</div>
            </div>
        </div>
        
        <h2>Training & Testing</h2>
        <table>
            <tr><th>Period</th><th>Seasons</th></tr>
            <tr><td>Training</td><td>{', '.join(result.train_periods)}</td></tr>
            <tr><td>Testing</td><td>{', '.join(result.test_periods)}</td></tr>
        </table>
        
        <h2>Detailed Metrics</h2>
        <table>
            <tr><th>Category</th><th>Metric</th><th>Value</th></tr>
            <tr><td>Profitability</td><td>ROI</td><td>{m.roi:+.2%}</td></tr>
            <tr><td>Profitability</td><td>Profit/Loss</td><td>${m.profit_loss:+,.2f}</td></tr>
            <tr><td>Profitability</td><td>Total Bets</td><td>{m.total_bets:,}</td></tr>
            <tr><td>Risk</td><td>Sharpe Ratio</td><td>{m.sharpe_ratio:.2f}</td></tr>
            <tr><td>Risk</td><td>Sortino Ratio</td><td>{m.sortino_ratio:.2f}</td></tr>
            <tr><td>Drawdown</td><td>Max Drawdown</td><td>{m.max_drawdown_pct:.2%}</td></tr>
            <tr><td>Drawdown</td><td>Max DD Duration</td><td>{m.max_drawdown_duration} bets</td></tr>
            <tr><td>Prediction</td><td>Brier Score</td><td>{m.brier_score:.4f}</td></tr>
            <tr><td>Prediction</td><td>Calibration Error</td><td>{m.calibration_error:.4f}</td></tr>
        </table>
        
        <h2>Equity Curve</h2>
        <p>Start: <strong>${result.session.initial_bankroll:,.2f}</strong> → 
           Final: <strong>${result.session.final_bankroll:,.2f}</strong></p>
    </div>
</body>
</html>
"""
        return html


def save_report(
    result: BacktestResult,
    filepath: str,
    format: str = "markdown"
) -> None:
    """
    Save backtest report to file.
    
    Args:
        result: BacktestResult to report
        filepath: Output file path
        format: 'markdown' or 'html'
    """
    reporter = BacktestReport()
    
    if format == "html":
        content = reporter.generate_html(result)
    else:
        content = reporter.generate_markdown(result)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

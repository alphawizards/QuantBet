# Model Monitoring & Validation

This directory contains tools for production model monitoring and validation.

## Components

### 1. Production Prediction Logger (`prediction_logger.py`)
Logs all predictions with metadata for tracking and evaluation.

**Usage**:
```python
from src.monitoring.prediction_logger import get_production_logger, create_prediction_log

# Create a prediction log
pred_log = create_prediction_log(
    game_id="NBL_2026_01_05_MEL_SYD",
    home_team="MEL",
    away_team="SYD",
    game_datetime="2026-01-05T19:00:00",
    model_name="Bayesian ELO",
    predicted_home_prob=0.62,
    home_odds=1.85,
    away_odds=2.10,
    bookmaker="Sportsbet",
    recommended_bet="BET_HOME",
    kelly_stake_pct=2.5,
    edge=0.047
)

# Log it
logger = get_production_logger()
logger.log_prediction(pred_log)
```

**Log Location**: `data/predictions/{YYYY-MM-DD}.jsonl`

---

### 2. Walk-Forward Validation (`walk_forward.py`)
Prevents overfitting with rolling window validation.

**Usage**:
```python
from src.backtest.walk_forward import WalkForwardValidator

def model_factory():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, max_depth=5)

validator = WalkForwardValidator(
    train_window_days=365,  # Train on 1 year
    test_window_days=30,    # Test on 1 month
    step_days=30            # Step forward 1 month
)

results = validator.validate(model_factory, data)
print(results['summary'])
```

**Key Metrics**:
- Mean Brier Score across windows
- Stability (std of Brier scores)
- Trend analysis

---

### 3. Model Monitoring Dashboard (`dashboard.py`)
Real-time performance monitoring and alerting.

**CLI Usage**:
```bash
python -m src.monitoring.dashboard
```

**Output**:
```
QUANTBET MODEL MONITORING DASHBOARD
================================================
Last Updated: 2026-01-05T10:30:00

📊 CALIBRATION METRICS:
  Brier Score: 0.1823
  Log Loss: 0.5234
  ECE: 0.0456

🎯 CLASSIFICATION METRICS:
  Accuracy: 58.3%
  Precision: 61.2%
  Recall: 55.1%

💰 BETTING METRICS:
  ROI: 4.2%
  Sharpe Ratio: 1.85
  Win Rate: 54.1%

💡 RECOMMENDATIONS:
  ✅ Model performance is healthy
```

**Alert Thresholds**:
- Brier Score > 0.25 → ⚠️
- Accuracy < 48% → ⚠️
- ECE > 0.10 → ⚠️
- ROI < -5% → ⚠️

---

## Integration with API

Add to prediction endpoints:

```python
from src.monitoring.prediction_logger import get_production_logger, create_prediction_log

@app.get("/predictions/today")
async def get_predictions():
    # ... make predictions ...
    
    # Log each prediction
    logger = get_production_logger()
    
    for game in games:
        pred_log = create_prediction_log(
            game_id=game.id,
            # ... other fields ...
        )
        logger.log_prediction(pred_log)
    
    return predictions
```

---

## Best Practices

### 1. Always Log Predictions
Every prediction made should be logged before being returned to users.

### 2. Update Outcomes
After games complete, update prediction logs with actual results:
```python
logger.update_outcome(prediction_id, home_score, away_score)
```

### 3. Monitor Daily
Run the dashboard CLI daily to check model health:
```bash
python -m src.monitoring.dashboard
```

### 4. Walk-Forward Before Deployment
Always validate with walk-forward before deploying model changes:
```bash
python scripts/validate_model.py --walk-forward
```

### 5. Set Up Alerts
Configure email/Slack alerts for threshold breaches in production.

---

## Metrics Glossary

**Brier Score**: Mean squared error of probability predictions. Lower is better (0 = perfect, 0.25 = random for binary classification).

**Expected Calibration Error (ECE)**: Average difference between predicted probabilities and observed frequencies. Lower is better (0 = perfect calibration).

**Calibration Slope**: Slope of calibration curve. 1.0 = perfect, <1.0 = overconfident, >1.0 = underconfident.

**Sharpe Ratio**: Risk-adjusted return. >1.5 is good, >2.0 is excellent.

**ROI**: Return on investment (profit / stake). >3% is good for sports betting.

---

## Files Generated

```
data/
├── predictions/          # Daily prediction logs
│   ├── 2026-01-01.jsonl
│   ├── 2026-01-02.jsonl
│   └── ...
├── monitoring/          # Monitoring outputs
│   ├── metrics.json
│   └── alerts.log
└── validation/          # Walk-forward results
    └── results.json
```

---

## Troubleshooting

**No predictions logged**:
- Check `data/predictions/` directory exists
- Verify logger is initialized in API startup

**Insufficient data for metrics**:
- Need at least 20 resolved predictions (games with known outcomes)
- Wait for more games to complete

**High Brier Score**:
- Model may be poorly calibrated
- Consider isotonic regression calibration
- Check for model drift

**Negative ROI**:
- Review edge threshold (should be >3%)
- Check Kelly fraction (recommend 0.25)
- Verify odds are from sharp bookmakers

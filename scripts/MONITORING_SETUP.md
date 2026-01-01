# QuantBet Monitoring Setup Guide

## Quick Start

### 1. Manual Test Run
```powershell
# Test the monitoring dashboard
cd c:\Users\ckr_4\01 Projects\QuantBet\QuantBet
python -m src.monitoring.dashboard
```

### 2. Run Walk-Forward Validation
```powershell
# Validate models before production
python scripts\run_walk_forward_validation.py
```

### 3. Run Daily Monitoring
```powershell
# Execute daily monitoring check
.\scripts\daily_monitoring.ps1 -Verbose
```

---

## Automated Setup (Windows Task Scheduler)

### Create Daily Monitoring Task

1. **Open Task Scheduler**:
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create New Task**:
   - Click "Create Basic Task..."
   - Name: `QuantBet Daily Monitoring`
   - Description: `Daily model performance monitoring`

3. **Set Trigger**:
   - Trigger: `Daily`
   - Start: `8:00 AM` (after markets close)
   - Recur every: `1 days`

4. **Set Action**:
   - Action: `Start a program`
   - Program/script: `powershell.exe`
   - Add arguments:
     ```
     -ExecutionPolicy Bypass -File "c:\Users\ckr_4\01 Projects\QuantBet\QuantBet\scripts\daily_monitoring.ps1" -Verbose
     ```
   - Start in: `c:\Users\ckr_4\01 Projects\QuantBet\QuantBet`

5. **Finish**:
   - Click "Finish"
   - Right-click task → "Run" to test

---

## API Integration Status

### Prediction Logging

**Status**: ⚠️ Needs manual integration  

The prediction logger code needs to be manually added to `src/api/app.py` around line 1027.

**Add this code block before line 1028**:

```python
# LOG PREDICTIONS TO PRODUCTION LOGGER
try:
    from src.monitoring.prediction_logger import get_production_logger, create_prediction_log
    import uuid
    
    prod_logger = get_production_logger()
    game_id = f"NBL_{datetime.now().strftime('%Y-%m-%d')}_{home_code}_{away_code}"
    
    # Log the best prediction (typically Bayesian ELO if available)
    if model_predictions:
        best_pred = model_predictions[0]  # Bayesian ELO
        
        pred_log = create_prediction_log(
            game_id=game_id,
            home_team=game.home_team,
            away_team=game.away_team,
            game_datetime=game.commence_time,
            model_name=best_pred.model_name,
            predicted_home_prob=best_pred.predicted_home_prob,
            home_odds=game.best_home_odds,
            away_odds=game.best_away_odds,
            bookmaker=game.bookmakers[0] if game.bookmakers else "Unknown",
            recommended_bet=best_pred.recommended_bet,
            kelly_stake_pct=best_pred.kelly_stake_pct,
            edge=best_pred.edge,
            uncertainty=0.05 if best_pred.confidence == "HIGH" else 0.10  # Approximate
        )
        
        prod_logger.log_prediction(pred_log)
        logger.info(f"Logged prediction for {game_id}")
        
except Exception as e:
    logger.error(f"Failed to log prediction: {e}")
    # Don't fail the request if logging fails
```

**Location**: Insert before creating `MultiModelGame` object

---

## Monitoring Outputs

### Daily Logs
```
data/
├── predictions/              # Daily prediction logs
│   └── YYYY-MM-DD.jsonl     # One file per day
│
├── monitoring/
│   ├── daily_monitor_YYYYMMDD.log      # Monitor execution logs
│   └── daily_report_YYYYMMDD.txt       # Daily summary reports
│
└── validation/
    └── walk_forward_results.json       # Validation results
```

### Metrics Dashboard Output
```
QUANTBET MODEL MONITORING DASHBOARD
====================================
Last Updated: 2025-12-31T18:00:00
Predictions: 45 total, 32 resolved

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

---

## Alert Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Brier Score | > 0.25 | ⚠️  Review calibration |
| Accuracy | < 48% | ⚠️  Check model drift |
| ECE | > 0.10 | ⚠️  Recalibrate probabilities |
| ROI | < -5% | 🚨 Stop betting, review strategy |

---

## Troubleshooting

### "No predictions found"
**Cause**: Prediction logger not integrated into API  
**Fix**: Manually add logging code to `app.py` (see above)

### "Insufficient data for metrics"
**Cause**: Need at least 20 resolved predictions  
**Fix**: Wait for games to complete, outcomes to be recorded

### "Module not found: src.monitoring"
**Cause**: Missing `__init__.py` file  
**Fix**: Create empty `src/monitoring/__init__.py`

### Task Scheduler not running
**Cause**: PowerShell execution policy  
**Fix**: Run as administrator: `Set-ExecutionPolicy RemoteSigned`

---

## Next Steps

1. **✅ Manual Integration**: Add prediction logging code to `app.py`

2. **✅ Test Logging**:
   ```powershell
   # Make API request
   curl http://localhost:8000/games/multi-model-predictions
   
   # Check log was created
   ls data\predictions\
   ```

3. **✅ Run Validation**:
   ```powershell
   python scripts\run_walk_forward_validation.py
   ```

4. **✅ Schedule Daily Monitoring**: Set up Task Scheduler as described above

5. **✅ Monitor for 1 Week**: Let system collect data

6. **Review Results**: After 1 week, review dashboard for model health

---

## Integration Checklist

- [ ] Add prediction logging to `src/api/app.py`
- [ ] Create `src/monitoring/__init__.py` (empty file)
- [ ] Test prediction logging works
- [ ] Run walk-forward validation
- [ ] Set up Task Scheduler for daily monitoring
- [ ] Verify daily monitoring runs successfully
- [ ] Review first weekly report

---

## Support

**Logs**: Check `data/monitoring/daily_monitor_*.log`  
**Reports**: Review `data/monitoring/daily_report_*.txt`  
**Manual Dashboard**: `python -m src.monitoring.dashboard`

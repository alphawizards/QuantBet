# QuantBet Production Deployment Checklist

## Pre-Deployment Checklist

### Infrastructure ✅
- [x] Docker containers running and healthy
- [x] Database credentials secured (environment variables)
- [x] API health checks enabled
- [x] Automated backups configured (7-day retention)
- [x] Prediction logging integrated into API
- [x] Monitoring dashboard operational

### Data Science ✅
- [x] Model review completed (A- grade)
- [x] Walk-forward validation passed (Brier: 0.26 ± 0.03, Stable)
- [x] Expected Calibration Error implemented
- [x] Statistical rigor validated
- [x] Risk management validated (Kelly, GBM, Sharpe)

### Monitoring ✅
- [x] Production prediction logger active
- [x] Daily monitoring script created
- [x] Alert thresholds configured
- [x] Comprehensive logging enabled

---

## Deployment Steps

### 1. Set Up Task Scheduler (Manual - 5 minutes)

**Run as Administrator**:
```powershell
cd "c:\Users\ckr_4\01 Projects\QuantBet\QuantBet"
.\scripts\setup_task_scheduler.ps1
```

**Verify**:
```powershell
Get-ScheduledTask -TaskName "QuantBet Daily Monitoring"
```

**Expected Output**:
```
TaskName: QuantBet Daily Monitoring
State: Ready
NextRunTime: Tomorrow at 08:00
```

**Status**: [ ] Complete

---

### 2. Paper Trading (1-2 weeks)

**Objective**: Validate system with 20+ predictions before risking capital

**Daily Tasks**:
- [ ] Check predictions: `curl http://localhost:8000/games/multi-model-predictions`
- [ ] Record paper trades in spreadsheet
- [ ] Update outcomes after games complete
- [ ] Review monitoring dashboard

**Weekly Review**:
- [ ] Calculate ROI, win rate, Sharpe ratio
- [ ] Check Brier score stability
- [ ] Review calibration curve
- [ ] Compare expected vs actual performance

**Success Criteria**:
- [ ] Brier Score < 0.25
- [ ] Accuracy > 48%
- [ ] ROI > 0%
- [ ] Sharpe Ratio > 1.0
- [ ] 20+ resolved predictions
- [ ] No system errors

**Documentation**: See `docs/PAPER_TRADING_GUIDE.md`

**Status**: [ ] Complete

---

### 3. System Validation

**Test Monitoring Dashboard**:
```powershell
python -m src.monitoring.dashboard
```

**Expected** (with sufficient data):
```
✅ Brier Score: < 0.25
✅ Accuracy: > 48%
✅ ECE: < 0.10
✅ ROI: Positive
```

**Check Prediction Logs**:
```powershell
ls data\predictions\
Get-Content data\predictions\$(Get-Date -Format 'yyyy-MM-dd').jsonl
```

**Verify API Integration**:
```powershell
# Make API call
curl http://localhost:8000/games/multi-model-predictions | ConvertFrom-Json

# Check log was created
ls data\predictions\
```

**Status**: [ ] Complete

---

### 4. Live Deployment with Minimal Stakes (After Paper Trading)

**Pre-Flight Checks**:
- [ ] Paper trading successful (ROI > 0%, 20+ predictions)
- [ ] Monitoring dashboard shows healthy metrics
- [ ] Daily automation working (Task Scheduler)
- [ ] Bankroll allocated (recommended: $1,000-$5,000)

**Week 1 Parameters**:
```
Max Stake: 0.5% of bankroll
Min Edge: 5% (conservative)
Max Bets/Day: 3
Stop Loss: -10% of bankroll
```

**Example**:
```
Bankroll: $5,000
Max Stake: $25 (0.5%)
If edge > 5%, Kelly says bet 3%:
  Actual bet = min($25, 3% × $5,000) = $25
```

**Daily Workflow**:
1. Check predictions at 6 PM (games usually 7-8 PM)
2. Only bet if:
   - Edge > 5%
   - Kelly > 0%
   - Confidence = HIGH
   - Bookmaker limits allow
3. Place bet immediately
4. Log bet manually in tracking spreadsheet
5. Update outcome after game

**Status**: [ ] Complete

---

### 5. Performance Monitoring (Ongoing)

**Daily**:
- [ ] Review new predictions
- [ ] Check dashboard for alerts
- [ ] Update bet tracking spreadsheet

**Weekly**:
- [ ] Run monitoring dashboard
- [ ] Calculate weekly ROI, Sharpe
- [ ] Review calibration metrics
- [ ] Check for model drift

**Monthly**:
- [ ] Full performance review
- [ ] Compare to expected performance
- [ ] Decide on stake scaling
- [ ] Consider model retraining if needed

**Alert Response**:
```
If Brier > 0.25: Review calibration, consider retraining
If ROI negative for 7 days: Stop betting, debug
If Accuracy < 45%: Model drift detected, retrain
If ECE > 0.10: Apply isotonic calibration
```

**Status**: [ ] Ongoing

---

### 6. Gradual Scaling (After Successful Week 1)

**Scaling Schedule** (if ROI > 3%, Sharpe > 1.5):

| Week | Max Stake % | Min Edge % | Max Bets/Day |
|------|-------------|------------|--------------|
| 1    | 0.5%        | 5%         | 3            |
| 2    | 1.0%        | 4%         | 4            |
| 3    | 1.5%        | 3.5%       | 5            |
| 4+   | 2.5%        | 3%         | No limit     |

**Stop Conditions**:
```
If ROI negative for 1 week: Revert to previous week's parameters
If drawdown > 15%: Reduce stakes by 50%
If Sharpe < 1.0 for 2 weeks: Stop and review
```

**Status**: [ ] Ongoing

---

## Post-Deployment Maintenance

### Weekly Maintenance
- [ ] Review prediction logs
- [ ] Check monitoring dashboard
- [ ] Update bet tracking
- [ ] Verify database backups exist

### Monthly Maintenance
- [ ] Run walk-forward validation on new data
- [ ] Review model performance vs baseline
- [ ] Consider model retraining
- [ ] Update expected performance metrics

### Quarterly Maintenance
- [ ] Full system audit
- [ ] Data quality review
- [ ] Model comparison (test new approaches)
- [ ] Infrastructure updates

---

## Emergency Procedures

### System Down
1. Check Docker: `docker ps`
2. Check logs: `docker logs quantbet_api`
3. Restart: `docker-compose restart`
4. If persists: Review error logs, contact support

### Poor Performance
1. Stop betting immediately
2. Run monitoring dashboard
3. Check calibration metrics
4. Review recent predictions vs outcomes
5. Consider model retraining

### Bookmaker Limits
1. Track max bet sizes by bookmaker
2. Distribute across multiple bookmakers
3. Consider using betting exchanges
4. Adjust Kelly sizing for limits

---

## Success Metrics

### Target Monthly Performance:
- **ROI**: 3-5% per bet
- **Sharpe Ratio**: 1.5-2.0
- **Win Rate**: 53-55%
- **Brier Score**: 0.18-0.22
- **Max Drawdown**: < 20%

### Target Annual Performance:
- **Total Return**: 20-30% (200 bets × 4% ROI)
- **Sharpe Ratio**: > 1.5
- **Max Drawdown**: < 25%
- **Probability of Ruin**: < 1%

---

## Support Resources

**Setup Guides**:
- `scripts/MONITORING_SETUP.md` - Monitoring setup
- `docs/PAPER_TRADING_GUIDE.md` - Paper trading workflow
- `scripts/BACKUP_README.md` - Backup procedures

**Commands**:
```powershell
# Dashboard
python -m src.monitoring.dashboard

# Walk-Forward Validation
python scripts\run_walk_forward_validation.py

# Daily Monitoring
.\scripts\daily_monitoring.ps1 -Verbose

# Database Backup
.\scripts\backup_db.ps1
```

**Logs**:
- Predictions: `data/predictions/YYYY-MM-DD.jsonl`
- Monitoring: `data/monitoring/daily_monitor_*.log`
- Reports: `data/monitoring/daily_report_*.txt`
- API: `docker logs quantbet_api`

---

## Final Checklist Before Going Live

- [ ] All infrastructure deployed and tested
- [ ] Paper trading completed (20+ predictions)
- [ ] Metrics meet targets (ROI > 0%, Brier < 0.25)
- [ ] Daily monitoring automated (Task Scheduler)
- [ ] Bankroll allocated and deposited
- [ ] Betting accounts opened and verified
- [ ] Stop-loss conditions defined
- [ ] Emergency procedures understood
- [ ] Support resources bookmarked

**When all checked**: 🚀 **READY FOR PRODUCTION**

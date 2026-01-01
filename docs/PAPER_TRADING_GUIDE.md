# QuantBet Paper Trading Guide

## What is Paper Trading?

Paper trading means tracking predictions and hypothetical bets WITHOUT risking real money. This validates your system works correctly before deploying capital.

---

## Duration: 1-2 Weeks

**Goal**: Collect 20+ resolved predictions to validate model performance

**Timeline**:
- Week 1: January 1-7, 2025 (ongoing NBL season)
- Week 2: January 8-14, 2025 (if needed)

---

## Daily Workflow

### Morning (Before Games)

1. **Check Upcoming Games**:
   ```bash
   curl http://localhost:8000/games/multi-model-predictions
   ```

2. **Review Predictions**:
   - Home/away win probabilities
   - Recommended bets
   - Kelly stake percentages
   - Edge calculations

3. **Record Paper Trades**:
   ```
   Date: 2025-01-05
   Game: Melbourne vs Sydney
   Prediction: MEL 62% (BET_HOME)
   Odds: 1.85
   Kelly Stake: 2.5% ($250 on $10K bankroll)
   Edge: 4.7%
   ```

4. **Paper Trade Decision**:
   - Only bet if edge > 3%
   - Only bet if Kelly > 0%
   - Max 2.5% of bankroll per bet

---

### Evening (After Games)

1. **Check Results**:
   - Visit NBL.com.au for final scores
   - Update your paper trading log

2. **Calculate Performance**:
   ```
   If MEL won:
   - Profit = $250 × (1.85 - 1) = $212.50
   - ROI = 212.50 / 250 = 85%
   
   If MEL lost:
   - Loss = -$250
   - ROI = -100%
   ```

3. **Update Tracking Spreadsheet**:
   | Date | Game | Bet | Odds | Stake | Result | P/L | ROI |
   |------|------|-----|------|-------|--------|-----|-----|
   | 1/5  | MEL-SYD | HOME | 1.85 | $250 | WIN | +$212 | +85% |

---

## Monitoring Dashboard

### Check Daily Metrics

```powershell
python -m src.monitoring.dashboard
```

**Expected Output** (after 20+ games):
```
📊 CALIBRATION METRICS:
  Brier Score: 0.18-0.22 (target: < 0.25) ✅
  ECE: 0.04-0.08 (target: < 0.10) ✅

🎯 CLASSIFICATION METRICS:
  Accuracy: 53-58% (target: > 48%) ✅

💰 BETTING METRICS:
  ROI: 3-5% (target: > 0%) ✅
  Sharpe Ratio: 1.5-2.0 (target: > 1.5) ✅
  Win Rate: 53-55% ✅
```

---

## Decision Criteria

### ✅ Proceed to Live Trading IF:
- [x] Brier Score < 0.25
- [x] Accuracy > 48%
- [x] ROI > 0% (profitable)
- [x] Sharpe Ratio > 1.0
- [x] At least 20 resolved predictions
- [x] No major system errors

### ⚠️ Review & Adjust IF:
- Brier Score > 0.25 (poor calibration)
- Accuracy < 48% (worse than random)
- ROI < -5% (losing money)
- System crashes or errors

### 🚨 Stop & Debug IF:
- ROI < -10%
- Brier Score > 0.30
- API errors on every request
- Dashboard shows critical alerts

---

## Tracking Template

### Create Excel/Google Sheets:

**Columns**:
1. Date
2. Game
3. Predicted Winner
4. Probability
5. Actual Winner
6. Bet Type (HOME/AWAY/SKIP)
7. Odds
8. Kelly Stake %
9. Paper Stake ($)
10. Result (WIN/LOSS/SKIP)
11. P/L ($)
12. Cumulative P/L ($)
13. ROI (%)

**Example Row**:
```
2025-01-05 | MEL-SYD | MEL | 62% | MEL | BET_HOME | 1.85 | 2.5% | $250 | WIN | +$212.50 | +$212.50 | +85%
```

---

## Weekly Review

### After Each Week:

1. **Calculate Metrics**:
   ```
   Total Bets: 15
   Wins: 8 (53.3%)
   Losses: 7
   Total Staked: $3,750
   Total P/L: +$156
   ROI: 4.2%
   ```

2. **Review Dashboard**:
   - Check Brier score trend
   - Review calibration curve
   - Compare expected vs actual performance

3. **Adjust if Needed**:
   - If underperforming: Lower Kelly fraction (0.25 → 0.10)
   - If Brier high: Review model calibration
   - If too many losses: Increase edge threshold (3% → 5%)

---

## Sample Paper Trading Log

### Week 1 Results (Hypothetical):

| Game | Prediction | Actual | Bet | Stake | Result | P/L |
|------|------------|--------|-----|-------|--------|-----|
| MEL-SYD | MEL 62% | MEL | HOME | $250 | WIN | +$212.50 |
| PER-BRI | PER 58% | BRI | HOME | $200 | LOSS | -$200 |
| SYD-ADL | SKIP | - | SKIP | $0 | - | $0 |
| NZB-PHX | NZB 65% | NZB | HOME | $300 | WIN | +$270 |
| MEL-PER | MEL 54% | PER | HOME | $100 | LOSS | -$100 |

**Week 1 Summary**:
- Bets: 4 (1 skipped)
- Wins: 2 (50%)
- Total Staked: $850
- Total P/L: +$182.50
- ROI: +21.5% (week)

---

## Red Flags to Watch For

### 🚨 Stop Paper Trading If:

1. **System Errors**:
   - API continuously returns 500 errors
   - Prediction logger not writing files
   - Dashboard crashes

2. **Model Drift**:
   - Accuracy drops below 45%
   - Brier score rises above 0.30
   - Predictions seem random

3. **Data Issues**:
   - Odds not updating
   - Wrong teams in predictions
   - Mismatched game times

---

## Transition to Live Trading

### After Successful Paper Trading:

1. **Start Small**:
   - First bet: 0.5% of bankroll (not 2.5%)
   - Max 3 bets per day
   - Only bet when edge > 5% (not 3%)

2. **Scale Gradually**:
   ```
   Week 1: 0.5% max stake, edge > 5%
   Week 2: 1.0% max stake, edge > 4%
   Week 3: 1.5% max stake, edge > 3.5%
   Week 4: 2.5% max stake, edge > 3%
   ```

3. **Monitor Closely**:
   - Check dashboard daily
   - Track every bet manually
   - Stop if ROI goes negative for 1 week

---

## Expected Paper Trading Results

### Realistic Expectations:

**Good Performance**:
- ROI: 3-7% per bet
- Win Rate: 52-56%
- Sharpe: 1.5-2.5
- Brier: 0.18-0.22

**Acceptable Performance**:
- ROI: 1-3% per bet
- Win Rate: 50-52%
- Sharpe: 1.0-1.5
- Brier: 0.22-0.25

**Poor Performance** (Don't go live):
- ROI: < 0% per bet
- Win Rate: < 48%
- Sharpe: < 1.0
- Brier: > 0.25

---

## Next Steps After Paper Trading

### If Results are Good (✅):
1. Open real betting accounts
2. Deposit minimal bankroll ($1,000-$5,000)
3. Start with 0.5% stakes
4. Monitor for 1 week
5. Scale to 2.5% stakes if profitable

### If Results are Mixed (⚠️):
1. Review model calibration
2. Adjust Kelly fraction lower
3. Increase edge threshold
4. Paper trade for another week
5. Re-evaluate

### If Results are Poor (🚨):
1. Stop paper trading
2. Review model predictions vs actual
3. Debug calibration issues
4. Retrain models with more data
5. Re-validate with walk-forward
6. Start paper trading again

---

## Support

**Questions**: Review `MONITORING_SETUP.md`  
**Dashboard**: `python -m src.monitoring.dashboard`  
**Logs**: `data/predictions/YYYY-MM-DD.jsonl`  
**Reports**: `data/monitoring/daily_report_*.txt`

# QuantBet - NBL/WNBL Quantitative Betting System

A rigorous portfolio construction approach to sports betting using **Advanced Kelly Criterion theory** for geometric growth optimization.

## Overview

QuantBet is a quantitative betting system targeting Australian Basketball (NBL/WNBL) markets, treating bankroll management as an asset allocation problem with:

- **Geometric growth optimization** via Kelly Criterion
- **Ruin-avoidance** through bounded-below models
- **Estimation error handling** via Bayesian shrinkage
- **Correlation-aware** multi-asset optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantBet.git
cd QuantBet

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Unix

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
QuantBet/
├── src/
│   ├── database/           # PostgreSQL schema & ORM models
│   │   ├── schema.sql      # CREATE TABLE statements
│   │   └── models.py       # SQLAlchemy models
│   ├── features/           # Feature engineering
│   │   └── engineer.py     # NBLFeatureEngineer class
│   ├── model/              # Probabilistic prediction
│   │   ├── predictor.py    # XGBoost walk-forward model
│   │   └── calibration.py  # Platt/Isotonic calibration
│   └── portfolio/          # Staking engine (core)
│       ├── kelly.py        # Kelly Criterion variants
│       ├── optimizer.py    # Multi-asset optimization
│       └── risk.py         # GBM & risk metrics
├── tests/
│   ├── test_kelly.py
│   ├── test_optimizer.py
│   └── test_features.py
└── requirements.txt
```

## Core Components

### 1. Data Infrastructure
PostgreSQL schema for games, player stats, and odds history with FIBA 40-minute scaling.

### 2. Feature Engineering
- **Rolling Efficiency**: Last 5 games ORtg/DRtg
- **Travel Fatigue**: Distance-based scoring with Perth-NZ weighting
- **Import Availability**: Key player tracking

### 3. Probabilistic Model
XGBoost with walk-forward validation (train 2021-2023, test 2024) and probability calibration.

### 4. Advanced Staking Engine

| Variant | Description |
|---------|-------------|
| Classic Kelly | `f* = (pb - q) / b` |
| Fractional Kelly | 0.25x multiplier for variance reduction |
| Bounded-Below | Floor constraint for tail risk protection |
| Bayesian Kelly | Shrinkage toward market implied probability |
| Multi-Asset | Simultaneous optimization with correlation handling |

## Usage

### Kelly Calculation

```python
from src.portfolio import kelly_criterion, fractional_kelly, bayesian_kelly

# Basic Kelly: 55% win probability at 2.0 decimal odds
result = kelly_criterion(0.55, 2.0)
print(f"Bet {result.fraction:.1%} of bankroll")  # 10%

# Fractional (quarter) Kelly
result = fractional_kelly(0.55, 2.0, fraction=0.25)
print(f"Bet {result.fraction:.1%} of bankroll")  # 2.5%

# Bayesian with shrinkage
result, p_shrunk = bayesian_kelly(
    prob_model=0.60,      # Our model says 60%
    prob_market=0.50,     # Market implies 50%
    decimal_odds=2.0,
    confidence=0.5        # 50/50 weight
)
print(f"Shrunk probability: {p_shrunk:.1%}")  # 55%
```

### Multi-Asset Optimization

```python
from src.portfolio import optimize_simultaneous_bets, BetOpportunity

bets = [
    BetOpportunity(prob=0.55, decimal_odds=2.0, bet_id="mel_win"),
    BetOpportunity(prob=0.52, decimal_odds=1.91, bet_id="syd_win"),
]

result = optimize_simultaneous_bets(bets, max_total_fraction=0.10)
print(result.stake_amounts(bankroll=10000))
```

## Running Tests

```bash
# Run all tests
python scripts/run_all_tests.py

# Run specific test suites
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run E2E tests (requires Playwright)
cd dashboard
npx playwright test
```

**Current Test Status** (as of 2026-01-04):
- ✅ Unit Tests: 170 passing
- 📊 Coverage: 5.1% (186/3636 lines)
- 🎯 Target Coverage: 80%+

## Bet Tracking

Track your actual placed bets and monitor real-time P/L performance:

```python
# Track a new bet
POST /api/bets/track
{
    "game_id": "mel_syd_20260104",
    "home_team": "Melbourne United",
    "away_team": "Sydney Kings",
    "game_date": "2026-01-04T19:00:00",
    "bet_on": "HOME",
    "prediction": 0.60,
    "odds": 2.15,
    "stake": 100.00,
    "confidence": "HIGH"
}

# Update bet result
PUT /api/bets/{bet_id}/result
{
    "actual_result": "HOME",
    "status": "WON"
}

# Get betting statistics
GET /api/bets/stats
# Returns: win_rate, roi, total_profit, total_staked
```

### Dashboard Features
- **Pending Bets Table**: View upcoming games with positive edge
- **Tracked Bets History**: Monitor placed bets with P/L tracking
- **Betting Statistics**: Win rate, ROI, and profit metrics



## Mathematical Foundation

The Kelly Criterion maximizes expected logarithmic growth:

```
f* = (p(b+1) - 1) / b = (pb - q) / b
```

Where:
- `f*` = optimal fraction of bankroll
- `p` = win probability
- `q` = 1 - p (lose probability)
- `b` = decimal odds - 1 (net odds)

> ⚠️ **Warning**: This implementation uses the exact logarithmic formulation, NOT the dangerous quadratic approximation often found online.

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Sports betting involves financial risk. Past performance does not guarantee future results. Always bet responsibly.

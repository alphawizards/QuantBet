# NBL/WNBL Quantitative Betting System - Technical Design Document

**Role:** Senior Quantitative Analyst / Lead Python Developer
**Domain:** Australian Basketball (NBL/WNBL) Inefficient Markets
**Date:** 2023-10-27

---

## 1. Executive Summary

This document outlines the technical architecture and implementation plan for a high-frequency automated betting system targeting the NBL and WNBL markets. Our thesis rests on the inefficiency of these markets due to lower liquidity, slower price discovery, and unique constraints (FIBA rules, travel distances, import player impact) compared to the NBA.

The system follows the Quantitative System Development Life Cycle (QSDLC):
1.  **Data Infrastructure:** Robust SQL schema for games, player stats, and odds.
2.  **Feature Engineering:** Domain-specific metrics (Travel Fatigue, Import Availability).
3.  **Modeling:** Walk-Forward XGBoost Regression for Margin of Victory.
4.  **Execution:** Fractional Kelly Criterion for position sizing.

---

## 2. Component 1: Data Infrastructure (SQL Schema)

We utilize PostgreSQL for its reliability and robust data typing. The schema is normalized to handle game metadata, granular player statistics, and odds history tracking.

### 2.1 Schema Design

The `games` table acts as the central fact table. `player_stats` links to games and players, handling the critical "Did Not Play" (DNP) scenarios common in lower-tier leagues. `odds_history` tracks line movement to analyze market efficiency.

**File:** `sql/schema.sql`

```sql
-- 1. Games Table
CREATE TABLE games (
    game_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    venue VARCHAR(255) NOT NULL,
    home_team_id INT NOT NULL,
    away_team_id INT NOT NULL,
    home_score INT,
    away_score INT,
    season VARCHAR(10) NOT NULL,
    is_playoff BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_home_team FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    CONSTRAINT fk_away_team FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

-- 2. Player Stats Table
CREATE TABLE player_stats (
    stat_id SERIAL PRIMARY KEY,
    game_id INT NOT NULL,
    player_id INT NOT NULL,
    team_id INT NOT NULL,
    minutes_played NUMERIC(4, 1),
    points INT,
    -- ... (other box score stats)
    dnp BOOLEAN DEFAULT FALSE,
    points_per_40 NUMERIC(5, 2), -- FIBA scaling
    CONSTRAINT fk_game FOREIGN KEY (game_id) REFERENCES games(game_id),
    CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- 3. Odds History Table
CREATE TABLE odds_history (
    odds_id SERIAL PRIMARY KEY,
    game_id INT NOT NULL,
    bookmaker_id INT NOT NULL,
    opening_spread NUMERIC(4, 1),
    closing_spread NUMERIC(4, 1),
    opening_moneyline NUMERIC(6, 3),
    closing_moneyline NUMERIC(6, 3),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_odds_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);
```

---

## 3. Component 2: Feature Engineering (Python)

We implement a dedicated `NBLFeatureEngineer` class to encapsulate domain logic. This ensures reproducibility and clean separation of concerns.

### 3.1 Key Features

1.  **Rolling Efficiency:** Captures recent team form adjusted for pace (possessions).
2.  **Travel Fatigue:** A custom metric specifically penalizing the Perth-New Zealand leg, which involves significant flight time and time zone shifts.
3.  **Import Availability:** Quantifies the impact of key American/Import players, whose absence disproportionately affects NBL teams.

**File:** `src/features.py`

```python
class NBLFeatureEngineer:
    def calculate_rolling_efficiency(self, team_id: int, window: int = 5) -> pd.DataFrame:
        """Calculates Last 5 games Offensive/Defensive Rating."""
        # Implementation details...

    def calculate_travel_fatigue(self, team_id: int, current_game_date: pd.Timestamp,
                               prev_game_venue: str, current_venue: str) -> int:
        """
        Calculates Travel_Fatigue_Score.
        Specific weighting: Perth (PER) to New Zealand (NZL) leg is weighted higher.
        """
        # ...

    def assess_import_availability(self, game_id: int, team_id: int) -> float:
        """
        Generates Import_Availability score based on active status of key imports.
        """
        # ...
```

---

## 4. Component 3: The Model (XGBoost)

We employ an XGBoost Regressor to predict the **Margin of Victory**. This approach allows us to capture non-linear relationships between features (e.g., fatigue compounding with opponent quality).

### 4.1 Validation Strategy
To avoid look-ahead bias, we strictly use **Walk-Forward Validation**:
- **Train:** 2021-2023 Seasons
- **Test:** 2024 Season

### 4.2 Probability Conversion
The model outputs a point margin. We convert this to a probability of covering the spread using the Normal Distribution Cumulative Distribution Function (CDF), assuming a standard deviation of margins typical for the NBL (~11.5 points).

**File:** `src/model.py`

```python
class NBLModel:
    def prepare_data(self, df: pd.DataFrame):
        """Splits data into Walk-Forward Validation sets."""
        train_mask = (df['date'] >= '2021-01-01') & (df['date'] < '2024-01-01')
        test_mask = (df['date'] >= '2024-01-01')
        # ...

    def predict_spread_probability(self, predicted_margin: float, spread: float, std_dev: float = 11.5) -> float:
        """
        Converts predicted margin to probability of covering the spread.
        P(Actual > -Spread) using Normal CDF.
        """
        threshold = -spread
        z_score = (threshold - predicted_margin) / std_dev
        prob_cover = 1 - norm.cdf(z_score)
        return prob_cover
```

---

## 5. Component 4: Staking Strategy (Kelly Criterion)

Capital allocation is managed via the **Kelly Criterion**, scaled down to a "Fractional Kelly" (e.g., 1/4) to reduce variance and risk of ruin.

### 5.1 Logic
The function `calculate_bet_size` computes the optimal percentage of bankroll to wager based on the edge (assessed probability vs. implied odds).

**File:** `src/staking.py`

```python
def calculate_bet_size(bankroll: float, assessed_probability: float, decimal_odds: float, kelly_fraction: float = 0.25) -> float:
    """
    Calculates bet size using Fractional Kelly.
    f* = (bp - q) / b
    """
    b = decimal_odds - 1
    p = assessed_probability
    q = 1 - p
    full_kelly = (b * p - q) / b

    return max(0.0, round(bankroll * full_kelly * kelly_fraction, 2))
```

---

## 6. Project Structure

```
.
├── DESIGN.md           # This document
├── README.md           # Project entry point
├── sql
│   └── schema.sql      # Database definitions
├── src
│   ├── __init__.py
│   ├── features.py     # Feature engineering logic
│   ├── model.py        # XGBoost model pipeline
│   └── staking.py      # Kelly criterion logic
└── tests
    └── test_components.py # Unit tests
```

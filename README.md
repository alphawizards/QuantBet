# NBL/WNBL Betting System

This repository contains the implementation of a quantitative betting system for Australian Basketball (NBL/WNBL).

## Overview

The system is designed to identify inefficiencies in the NBL market using a data-driven approach. It includes components for data ingestion, feature engineering, modeling, and staking.

## Documentation

For a comprehensive technical design document and implementation details, please refer to [DESIGN.md](DESIGN.md).

## Usage

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy xgboost scipy scikit-learn
    ```

2.  **Run Tests:**
    ```bash
    python3 -m unittest tests/test_components.py
    ```

3.  **Explore Code:**
    - `src/features.py`: Feature engineering logic (Travel fatigue, Import availability).
    - `src/model.py`: XGBoost model for margin prediction.
    - `src/staking.py`: Fractional Kelly Criterion implementation.
    - `sql/schema.sql`: Database schema.

# Data Fetching & Storage Review

## Overview
This document summarizes the findings from an audit of the data pipeline, identifying data sources, storage mechanisms, and potential issues.

## 1. Data Sources

| Source | Description | Collector Module | Status |
| :--- | :--- | :--- | :--- |
| **nblR (GitHub)** | Historical match results, box scores (player/team), pbp, shots. | `src.collectors.scraper` | ✅ Functional |
| **The Odds API** | Live betting odds from Australian bookmakers. | `src.collectors.odds_api` | ✅ Functional |
| **XLSX History** | Historical odds data from AusSportsBetting. | `src.collectors.integration` | ✅ Functional |
| **SpatialJam** | Premium analytics (Shot Machine, BPM, etc.). | `src.collectors.spatialjam_scraper` | 🚧 Stubbed |

## 2. Storage Architecture

The system exhibits a **split storage architecture**:

1.  **Production Pipeline (In-Memory/Parquet):**
    *   Scrapers fetch data and return Pandas DataFrames.
    *   `NBLDataScraper` caches these DataFrames locally as Parquet files (`data/cache/*.parquet`).
    *   `FeatureStore` and strategies consume these DataFrames directly.
    *   **Verdict:** This is the *active* pipeline used for analysis and backtesting.

2.  **Database (PostgreSQL):**
    *   **Schema:** A sophisticated schema (`src/core/database/schema.sql`) exists, defining tables for `games`, `teams`, `player_stats`, `odds_history`, etc., with normalized relationships and triggers.
    *   **Loading:** The script `scripts/load_data.py` populates a *different, simplified* set of tables (`nbl_games`, `nbl_odds`) using raw SQL.
    *   **Usage:** There appears to be no code currently reading from the sophisticated DB schema for strategy input. The `load_data.py` script seems to be a standalone utility not integrated with the core application flow.

## 3. Key Issues

*   **Schema Mismatch:** The database schema defined in SQL/ORM (`games`) matches the domain model but is **empty/unused**. The script `load_data.py` writes to `nbl_games` which is not referenced by the ORM or strategies.
*   **Missing Ingestion Logic:** There is no code path that takes the rich data from `NBLDataScraper` (box scores, pbp) and writes it to the PostgreSQL `player_stats` table.
*   **SpatialJam Integration:** The scraper is a stub. Premium data is not being ingested.

## 4. Recommendations

1.  **Unify Storage:**
    *   Update `scripts/load_data.py` (or create a new `scripts/ingest_pipeline.py`) to populate the *actual* ORM models (`src.core.database.models`) instead of raw SQL tables.
    *   Ingest data from `NBLDataScraper` into the DB, not just the XLSX file.

2.  **Feature Store Backend:**
    *   Update `FeatureStore` to optionally read from the DB (`FeatureStore(source='db')`) in addition to DataFrames, enabling production serving without re-scraping/loading Parquet files.

3.  **Clean Up:**
    *   Deprecate the `nbl_games`/`nbl_odds` tables if they are redundant with the core schema.

## 5. Verification

An end-to-end integration test `tests/test_data_pipeline.py` has been created to verify the current "In-Memory" pipeline. It confirms that:
*   Scraping and Caching works.
*   Data Integration (merging sources) works.
*   Feature Engineering works on the merged data.

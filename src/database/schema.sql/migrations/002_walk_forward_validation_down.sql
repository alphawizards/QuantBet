-- Migration 002 Rollback: Walk-Forward Validation Tables
-- Description: Removes walk-forward validation tables
-- Date: 2026-01-01

BEGIN;

-- Drop tables in reverse order (child first, parent second)
DROP TABLE IF EXISTS backtest_window_results CASCADE;
DROP TABLE IF EXISTS backtest_validations CASCADE;

COMMIT;

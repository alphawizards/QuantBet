-- Migration 003 Rollback: Parameter Sensitivity Analysis
-- Description: Removes parameter sensitivity results table
-- Date: 2026-01-01

BEGIN;

DROP TABLE IF EXISTS parameter_sensitivity_results CASCADE;

COMMIT;

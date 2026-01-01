-- Migration 005 Rollback: Production Monitoring Tables
-- Description: Removes all production monitoring tables
-- Date: 2026-01-01

BEGIN;

-- Drop tables in reverse order
DROP TABLE IF EXISTS retraining_jobs CASCADE;
DROP TABLE IF EXISTS performance_alerts CASCADE;
DROP TABLE IF EXISTS drift_alerts CASCADE;
DROP TABLE IF EXISTS drift_baselines CASCADE;
DROP TABLE IF EXISTS calibration_history CASCADE;

COMMIT;

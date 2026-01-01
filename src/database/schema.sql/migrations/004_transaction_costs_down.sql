-- Migration 004 Rollback: Transaction Costs
-- Description: Removes transaction cost columns from backtest_validations
-- Date: 2026-01-01

BEGIN;

-- Remove constraints first
ALTER TABLE backtest_validations
    DROP CONSTRAINT IF EXISTS valid_execution_rate,
    DROP CONSTRAINT IF EXISTS valid_commissions,
    DROP CONSTRAINT IF EXISTS gross_vs_net_roi;

-- Drop index
DROP INDEX IF EXISTS idx_backtest_roi;

-- Remove columns
ALTER TABLE backtest_validations
    DROP COLUMN IF EXISTS gross_roi,
    DROP COLUMN IF EXISTS net_roi,
    DROP COLUMN IF EXISTS total_commissions,
    DROP COLUMN IF EXISTS execution_rate;

COMMIT;

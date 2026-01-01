-- Migration 004: Transaction Costs
-- Description: Adds transaction cost columns to backtest_validations table
-- Dependencies: 002_walk_forward_validation.sql MUST BE APPLIED FIRST
-- Date: 2026-01-01

BEGIN;

-- Verify backtest_validations exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE tablename = 'backtest_validations') THEN
        RAISE EXCEPTION 'Migration 002 must be applied before migration 004. Table backtest_validations does not exist.';
    END IF;
END $$;

-- ============================================================================
-- Add Transaction Cost Columns
-- ============================================================================

ALTER TABLE backtest_validations
    ADD COLUMN IF NOT EXISTS gross_roi NUMERIC(6, 4),
    ADD COLUMN IF NOT EXISTS net_roi NUMERIC(6, 4),
    ADD COLUMN IF NOT EXISTS total_commissions NUMERIC(10, 2) DEFAULT 0,
    ADD COLUMN IF NOT EXISTS execution_rate NUMERIC(5, 4);

-- Add constraints with idempotent error handling
DO $$
BEGIN
    ALTER TABLE backtest_validations
        ADD CONSTRAINT valid_execution_rate CHECK (execution_rate IS NULL OR (execution_rate >= 0 AND execution_rate <= 1));
EXCEPTION
    WHEN duplicate_object THEN
        NULL; -- Constraint already exists, ignore
END $$;

DO $$
BEGIN
    ALTER TABLE backtest_validations
        ADD CONSTRAINT valid_commissions CHECK (total_commissions >= 0);
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE backtest_validations
        ADD CONSTRAINT gross_vs_net_roi CHECK (gross_roi IS NULL OR net_roi IS NULL OR net_roi <= gross_roi);
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

-- Add index for cost analysis queries
CREATE INDEX IF NOT EXISTS idx_backtest_roi ON backtest_validations(model_name, net_roi DESC NULLS LAST);

COMMENT ON COLUMN backtest_validations.gross_roi IS 'ROI before transaction costs';
COMMENT ON COLUMN backtest_validations.net_roi IS 'ROI after transaction costs (commissions + slippage)';
COMMENT ON COLUMN backtest_validations.total_commissions IS 'Total commissions paid in currency units';
COMMENT ON COLUMN backtest_validations.execution_rate IS 'Percentage of bets successfully filled (0-1)';

COMMIT;

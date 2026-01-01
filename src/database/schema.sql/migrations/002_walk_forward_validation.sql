-- Migration 002: Walk-Forward Validation Tables
-- Description: Adds tables for storing backtest validation results using walk-forward methodology
-- Dependencies: 001_initial_schema.sql (base schema)
-- Date: 2026-01-01

BEGIN;

-- ============================================================================
-- Backtest Validations Table
-- ============================================================================
-- Stores summary results from walk-forward validation runs

CREATE TABLE IF NOT EXISTS backtest_validations (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    validation_type VARCHAR(50) NOT NULL CHECK (validation_type IN ('walk_forward', 'holdout', 'cv')),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    n_windows INTEGER,
    mean_brier_score NUMERIC(6, 4),
    std_brier_score NUMERIC(6, 4),
    mean_accuracy NUMERIC(5, 4),
    mean_roi NUMERIC(6, 4),
    config JSONB,  -- stores window sizes, step size, etc.
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_brier CHECK (mean_brier_score >= 0 AND mean_brier_score <= 1),
    CONSTRAINT valid_accuracy CHECK (mean_accuracy >= 0 AND mean_accuracy <= 1),
    CONSTRAINT valid_windows CHECK (n_windows > 0)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_backtest_val_model ON backtest_validations(model_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_val_type ON backtest_validations(validation_type, model_name);
CREATE INDEX IF NOT EXISTS idx_backtest_val_date ON backtest_validations(created_at DESC);

COMMENT ON TABLE backtest_validations IS 'Stores walk-forward and holdout validation results';
COMMENT ON COLUMN backtest_validations.validation_type IS 'Type: walk_forward, holdout, or cv (cross-validation)';
COMMENT ON COLUMN backtest_validations.config IS 'JSONB config: {train_days: 365, test_days: 30, step_days: 30}';

-- ============================================================================
-- Backtest Window Results Table
-- ============================================================================
-- Stores per-window results from walk-forward validation

CREATE TABLE IF NOT EXISTS backtest_window_results (
    window_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_id UUID NOT NULL REFERENCES backtest_validations(validation_id) ON DELETE CASCADE,
    window_number INTEGER NOT NULL,
    train_start TIMESTAMP NOT NULL,
    train_end TIMESTAMP NOT NULL,
    test_start TIMESTAMP NOT NULL,
    test_end TIMESTAMP NOT NULL,
    n_train_samples INTEGER NOT NULL,
    n_test_samples INTEGER NOT NULL,
    brier_score NUMERIC(6, 4),
    log_loss NUMERIC(8, 6),
    accuracy NUMERIC(5, 4),
    roi NUMERIC(6, 4),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_window_dates CHECK (train_start < train_end AND test_start < test_end),
    CONSTRAINT valid_window_samples CHECK (n_train_samples > 0 AND n_test_samples > 0),
    CONSTRAINT valid_window_brier CHECK (brier_score >= 0 AND brier_score <= 1),
    CONSTRAINT valid_window_accuracy CHECK (accuracy >= 0 AND accuracy <= 1)
);

-- Unique constraint: one result per window per validation
CREATE UNIQUE INDEX IF NOT EXISTS idx_window_unique ON backtest_window_results(validation_id, window_number);

-- Indexes for queries
CREATE INDEX IF NOT EXISTS idx_window_validation ON backtest_window_results(validation_id, window_number);
CREATE INDEX IF NOT EXISTS idx_window_dates ON backtest_window_results(test_start, test_end);

COMMENT ON TABLE backtest_window_results IS 'Stores metrics for each window in walk-forward validation';
COMMENT ON COLUMN backtest_window_results.window_number IS 'Sequential window number (1, 2, 3...)';
COMMENT ON COLUMN backtest_window_results.brier_score IS 'Brier score for this window (0-1, lower is better)';

COMMIT;

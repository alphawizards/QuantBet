-- Migration 003: Parameter Sensitivity Analysis
-- Description: Adds table for tracking parameter sensitivity test results
-- Dependencies: None (standalone)
-- Date: 2026-01-01

BEGIN;

-- ============================================================================
-- Parameter Sensitivity Results Table
-- ============================================================================
-- Stores results from grid search parameter sensitivity analysis

CREATE TABLE IF NOT EXISTS parameter_sensitivity_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    parameter_name VARCHAR(50) NOT NULL,
    parameter_value NUMERIC(10, 4) NOT NULL,
    brier_score NUMERIC(6, 4),
    roi NUMERIC(6, 4),
    sharpe_ratio NUMERIC(6, 3),
    accuracy NUMERIC(5, 4),
    tested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_param_brier CHECK (brier_score >= 0 AND brier_score <= 1),
    CONSTRAINT valid_param_accuracy CHECK (accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1))
);

-- Unique constraint: prevent duplicate test results
-- Format: one row per (model, parameter, value) combination per day
CREATE UNIQUE INDEX IF NOT EXISTS idx_param_unique ON parameter_sensitivity_results(
    model_name, 
    parameter_name, 
    parameter_value,
    CAST(tested_at AS DATE)
);

-- Indexes for querying specific parameters
CREATE INDEX IF NOT EXISTS idx_param_sensitivity ON parameter_sensitivity_results(model_name, parameter_name);
CREATE INDEX IF NOT EXISTS idx_param_tested_at ON parameter_sensitivity_results(tested_at DESC);

COMMENT ON TABLE parameter_sensitivity_results IS 'Stores grid search results for parameter robustness testing';
COMMENT ON COLUMN parameter_sensitivity_results.parameter_name IS 'e.g., k_factor, kelly_fraction, home_advantage';
COMMENT ON COLUMN parameter_sensitivity_results.parameter_value IS 'The specific parameter value tested';
COMMENT ON COLUMN parameter_sensitivity_results.sharpe_ratio IS 'Risk-adjusted return metric';

COMMIT;

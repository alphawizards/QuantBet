-- Migration 005: Production Monitoring Tables
-- Description: Adds tables for calibration monitoring, drift detection, alerts, and retraining
-- Dependencies: None (standalone)
-- Date: 2026-01-01

BEGIN;

-- ============================================================================
-- Calibration History Table
-- ============================================================================
-- Stores time-series calibration metrics

CREATE TABLE IF NOT EXISTS calibration_history (
    calibration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,  -- FIXED: was calculated_at_timestamp, now with timezone
    window_size INTEGER NOT NULL,  -- number of predictions in window
    expected_calibration_error NUMERIC(6, 4),
    max_calibration_error NUMERIC(6, 4),
    brier_score NUMERIC(6, 4),
    calibration_bins JSONB,  -- stores predicted vs actual per bin
    
    CONSTRAINT valid_window_size CHECK (window_size > 0),
    CONSTRAINT valid_ece CHECK (expected_calibration_error >= 0 AND expected_calibration_error <= 1),
    CONSTRAINT valid_mce CHECK (max_calibration_error >= 0 AND max_calibration_error <= 1)
);

CREATE INDEX IF NOT EXISTS idx_calibration_model_time ON calibration_history(model_name, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_calibration_time ON calibration_history(calculated_at DESC);

COMMENT ON TABLE calibration_history IS 'Time-series tracking of model calibration quality';
COMMENT ON COLUMN calibration_history.expected_calibration_error IS 'ECE: mean absolute difference between predicted and observed probabilities';
COMMENT ON COLUMN calibration_history.calibration_bins IS 'JSONB array: [{predicted: 0.1, observed: 0.12, count: 50}, ...]';

-- ============================================================================
-- Drift Baselines Table
-- ============================================================================
-- Manages baseline periods for drift detection

CREATE TABLE IF NOT EXISTS drift_baselines (
    baseline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    baseline_start TIMESTAMPTZ NOT NULL,
    baseline_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    sample_size INTEGER,
    metadata JSONB,
    
    CONSTRAINT valid_baseline_dates CHECK (baseline_start < baseline_end)
);

-- Only one active baseline per model
CREATE UNIQUE INDEX IF NOT EXISTS idx_baseline_active ON drift_baselines(model_name, is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_baseline_model ON drift_baselines(model_name, is_active);

COMMENT ON TABLE drift_baselines IS 'Manages baseline periods for comparing production drift';
COMMENT ON COLUMN drift_baselines.is_active IS 'Only one active baseline allowed per model';
COMMENT ON COLUMN drift_baselines.metadata IS 'Optional: feature statistics, distribution summaries';

-- ============================================================================
-- Drift Alerts Table
-- ============================================================================
-- Stores detected drift events

CREATE TABLE IF NOT EXISTS drift_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    drift_type VARCHAR(50) NOT NULL CHECK (drift_type IN ('feature_drift', 'prediction_drift')),
    feature_name VARCHAR(100),  -- NULL for prediction drift
    psi_score NUMERIC(6, 4),
    ks_statistic NUMERIC(6, 4),
    p_value NUMERIC(8, 6),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH')),
    detected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    baseline_start TIMESTAMPTZ,
    baseline_end TIMESTAMPTZ,
    current_start TIMESTAMPTZ,
    current_end TIMESTAMPTZ,
    acknowledged BOOLEAN DEFAULT FALSE,
    
    CONSTRAINT valid_psi CHECK (psi_score >= 0),
    CONSTRAINT valid_ks CHECK (ks_statistic >= 0 AND ks_statistic <= 1),
    CONSTRAINT valid_p_value CHECK (p_value >= 0 AND p_value <= 1)
);

CREATE INDEX IF NOT EXISTS idx_drift_alerts_time ON drift_alerts(detected_at DESC, severity);
CREATE INDEX IF NOT EXISTS idx_drift_model ON drift_alerts(model_name, acknowledged);

COMMENT ON TABLE drift_alerts IS 'Detected drift events (feature/prediction distribution changes)';
COMMENT ON COLUMN drift_alerts.psi_score IS 'Population Stability Index: <0.1=low, 0.1-0.25=medium, >0.25=high';
COMMENT ON COLUMN drift_alerts.ks_statistic IS 'Kolmogorov-Smirnov test statistic';

-- ============================================================================
-- Performance Alerts Table
-- ============================================================================
-- Stores performance degradation alerts

CREATE TABLE IF NOT EXISTS performance_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN ('brier_score', 'roi', 'accuracy', 'ece')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'CRITICAL')),
    metric_value NUMERIC(10, 4),
    threshold_value NUMERIC(10, 4),
    message TEXT,
    triggered_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    notified_via VARCHAR(255),  -- e.g., 'email,slack'
    
    CONSTRAINT valid_severity_order CHECK (severity IN ('INFO', 'WARNING', 'CRITICAL'))
);

CREATE INDEX IF NOT EXISTS idx_perf_alerts_time ON performance_alerts(triggered_at DESC, severity);
CREATE INDEX IF NOT EXISTS idx_perf_model ON performance_alerts(model_name, acknowledged);

COMMENT ON TABLE performance_alerts IS 'Performance degradation alerts (Brier, ROI, accuracy, calibration)';
COMMENT ON COLUMN performance_alerts.alert_type IS 'Metric that triggered alert';
COMMENT ON COLUMN performance_alerts.notified_via IS 'Comma-separated list of notification channels';

-- ============================================================================
-- Retraining Jobs Table
-- ============================================================================
-- Tracks automated model retraining jobs

CREATE TABLE IF NOT EXISTS retraining_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL CHECK (trigger_type IN ('scheduled', 'performance', 'drift', 'manual')),
    trigger_reason TEXT,
    status VARCHAR(50) NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'VALIDATING', 'COMPLETED', 'FAILED')),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    champion_brier NUMERIC(6, 4),  -- old model performance
    challenger_brier NUMERIC(6, 4),  -- new model performance
    improvement_pct NUMERIC(6, 2),
    promoted BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_completion CHECK (completed_at IS NULL OR completed_at >= started_at),
    CONSTRAINT valid_brier_scores CHECK (
        (champion_brier IS NULL OR (champion_brier >= 0 AND champion_brier <= 1)) AND
        (challenger_brier IS NULL OR (challenger_brier >= 0 AND challenger_brier <= 1))
    )
);

CREATE INDEX IF NOT EXISTS idx_retraining_jobs_status ON retraining_jobs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retraining_model ON retraining_jobs(model_name, created_at DESC);

COMMENT ON TABLE retraining_jobs IS 'Automated model retraining job tracking';
COMMENT ON COLUMN retraining_jobs.trigger_type IS 'What triggered retraining: scheduled, performance degradation, drift, or manual';
COMMENT ON COLUMN retraining_jobs.promoted IS 'Whether challenger model was promoted to production';

COMMIT;

-- Migration: Add User Bets Tracking Table
-- Created: 2026-01-03
-- Purpose: Track user's actual placed bets for real P/L monitoring

CREATE TABLE IF NOT EXISTS user_bets (
    -- Primary identification
    id SERIAL PRIMARY KEY,
    bet_id VARCHAR(100) UNIQUE NOT NULL,  -- Unique bet identifier
    user_id VARCHAR(50) DEFAULT 'default_user',
    
    -- Game information
    game_id VARCHAR(100) NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    game_date TIMESTAMP NOT NULL,
    
    -- Bet details
    bet_on VARCHAR(10) NOT NULL CHECK (bet_on IN ('HOME', 'AWAY')),
    prediction DECIMAL(5,4) NOT NULL CHECK (prediction >= 0 AND prediction <= 1),
    odds DECIMAL(6,3) NOT NULL CHECK (odds > 1.0),
    stake DECIMAL(10,2) NOT NULL CHECK (stake > 0),
    edge DECIMAL(5,4),
    
    -- Model information
    model_id VARCHAR(50),
    confidence VARCHAR(20) CHECK (confidence IN ('HIGH', 'MEDIUM', 'LOW')),
    bookmaker VARCHAR(100),
    
    -- Outcome tracking
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'WON', 'LOST', 'VOID', 'CANCELLED')),
    actual_result VARCHAR(10) CHECK (actual_result IN ('HOME', 'AWAY', NULL)),
    profit DECIMAL(10,2),  -- NULL until settled
    settled_at TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Indexes for performance
CREATE INDEX idx_user_bets_user_id ON user_bets(user_id);
CREATE INDEX idx_user_bets_game_date ON user_bets(game_date DESC);
CREATE INDEX idx_user_bets_status ON user_bets(status);
CREATE INDEX idx_user_bets_created_at ON user_bets(created_at DESC);
CREATE INDEX idx_user_bets_game_id ON user_bets(game_id);

-- Composite index for common query patterns
CREATE INDEX idx_user_bets_user_status_date ON user_bets(user_id, status, game_date DESC);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_user_bets_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_user_bets_updated_at
    BEFORE UPDATE ON user_bets
    FOR EACH ROW
    EXECUTE FUNCTION update_user_bets_updated_at();

-- Add comment for documentation
COMMENT ON TABLE user_bets IS 'Tracks user placed bets for real-time P/L monitoring and performance analysis';
COMMENT ON COLUMN user_bets.bet_id IS 'Unique identifier for this bet (UUID format recommended)';
COMMENT ON COLUMN user_bets.status IS 'PENDING = bet placed but not settled, WON/LOST = outcome known, VOID = cancelled bet';
COMMENT ON COLUMN user_bets.profit IS 'Calculated profit/loss: (stake * odds - stake) if won, -stake if lost';

-- ============================================================================
-- QuantBet: NBL/WNBL Quantitative Betting System
-- PostgreSQL Schema Definition
-- ============================================================================
-- 
-- This schema supports:
--   1. Game metadata with venue and team information
--   2. Player box scores with FIBA 40-minute scaling and DNP handling
--   3. Odds history tracking opening vs closing lines
--
-- ============================================================================

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

CREATE TYPE league_type AS ENUM ('NBL', 'WNBL');
CREATE TYPE game_status AS ENUM ('scheduled', 'in_progress', 'completed', 'postponed', 'cancelled');
CREATE TYPE dnp_reason AS ENUM ('injury', 'rest', 'coach_decision', 'personal', 'suspension', 'not_with_team');
CREATE TYPE bet_type AS ENUM ('moneyline', 'spread', 'total', 'player_prop');
CREATE TYPE bet_outcome AS ENUM ('pending', 'win', 'loss', 'push', 'void');

-- ============================================================================
-- TABLE: teams
-- Reference table for NBL/WNBL teams
-- ============================================================================

CREATE TABLE teams (
    team_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_code       VARCHAR(10) NOT NULL UNIQUE,  -- e.g., 'MEL', 'SYD', 'PER'
    team_name       VARCHAR(100) NOT NULL,         -- e.g., 'Melbourne United'
    league          league_type NOT NULL,
    city            VARCHAR(50) NOT NULL,
    state           VARCHAR(50),
    country         VARCHAR(50) NOT NULL DEFAULT 'Australia',
    venue_name      VARCHAR(100),
    latitude        DECIMAL(10, 7),                -- For travel distance calculations
    longitude       DECIMAL(10, 7),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for common lookups
CREATE INDEX idx_teams_league ON teams(league) WHERE is_active = TRUE;
CREATE INDEX idx_teams_code ON teams(team_code);

-- ============================================================================
-- TABLE: games
-- Stores game metadata including date, venue, home/away teams
-- ============================================================================

CREATE TABLE games (
    game_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_game_id    VARCHAR(50) UNIQUE,            -- ID from data provider
    league              league_type NOT NULL,
    season              VARCHAR(10) NOT NULL,           -- e.g., '2023-24'
    round_number        INTEGER,
    
    -- Scheduling
    scheduled_datetime  TIMESTAMP WITH TIME ZONE NOT NULL,
    actual_start_time   TIMESTAMP WITH TIME ZONE,
    actual_end_time     TIMESTAMP WITH TIME ZONE,
    
    -- Teams
    home_team_id        UUID NOT NULL REFERENCES teams(team_id),
    away_team_id        UUID NOT NULL REFERENCES teams(team_id),
    
    -- Venue (may differ from home team's default venue)
    venue_name          VARCHAR(100),
    venue_city          VARCHAR(50),
    venue_country       VARCHAR(50) DEFAULT 'Australia',
    is_neutral_site     BOOLEAN DEFAULT FALSE,
    
    -- Results
    status              game_status NOT NULL DEFAULT 'scheduled',
    home_score          INTEGER CHECK (home_score >= 0),
    away_score          INTEGER CHECK (away_score >= 0),
    home_q1_score       INTEGER,
    home_q2_score       INTEGER,
    home_q3_score       INTEGER,
    home_q4_score       INTEGER,
    home_ot_score       INTEGER DEFAULT 0,
    away_q1_score       INTEGER,
    away_q2_score       INTEGER,
    away_q3_score       INTEGER,
    away_q4_score       INTEGER,
    away_ot_score       INTEGER DEFAULT 0,
    
    -- Metadata
    attendance          INTEGER,
    notes               TEXT,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT different_teams CHECK (home_team_id != away_team_id),
    CONSTRAINT valid_scores CHECK (
        (status != 'completed') OR 
        (home_score IS NOT NULL AND away_score IS NOT NULL)
    )
);

-- Indexes for common query patterns
CREATE INDEX idx_games_datetime ON games(scheduled_datetime DESC);
CREATE INDEX idx_games_season ON games(league, season);
CREATE INDEX idx_games_home_team ON games(home_team_id, scheduled_datetime DESC);
CREATE INDEX idx_games_away_team ON games(away_team_id, scheduled_datetime DESC);
CREATE INDEX idx_games_status ON games(status) WHERE status IN ('scheduled', 'in_progress');

-- ============================================================================
-- TABLE: players
-- Reference table for player information
-- ============================================================================

CREATE TABLE players (
    player_id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_player_id  VARCHAR(50) UNIQUE,
    first_name          VARCHAR(50) NOT NULL,
    last_name           VARCHAR(50) NOT NULL,
    display_name        VARCHAR(100) GENERATED ALWAYS AS (first_name || ' ' || last_name) STORED,
    
    -- Current team (NULL if free agent)
    current_team_id     UUID REFERENCES teams(team_id),
    jersey_number       VARCHAR(5),
    position            VARCHAR(20),                    -- e.g., 'PG', 'SG', 'SF', 'PF', 'C'
    
    -- Import status (critical for NBL analysis)
    is_import           BOOLEAN NOT NULL DEFAULT FALSE,
    import_tier         INTEGER CHECK (import_tier BETWEEN 1 AND 3),  -- NBL import tiers
    
    -- Physical attributes
    height_cm           INTEGER CHECK (height_cm BETWEEN 150 AND 250),
    weight_kg           DECIMAL(5, 2),
    birth_date          DATE,
    nationality         VARCHAR(50),
    
    -- Metadata
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    metadata            JSONB DEFAULT '{}',             -- Flexible additional data
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_players_team ON players(current_team_id) WHERE is_active = TRUE;
CREATE INDEX idx_players_import ON players(is_import, current_team_id) WHERE is_active = TRUE;
CREATE INDEX idx_players_name ON players(last_name, first_name);

-- ============================================================================
-- TABLE: player_stats
-- Box score statistics with FIBA 40-minute scaling and DNP handling
-- ============================================================================
-- 
-- FIBA games are 40 minutes (4 x 10-minute quarters) vs NBA's 48 minutes.
-- We store raw stats and provide per-40-minute scaled values for comparison.
--
-- DNP (Did Not Play) handling:
--   - seconds_played = 0 with dnp_reason populated
--   - All counting stats will be NULL or 0
--   - This distinguishes DNP from players not on roster
-- ============================================================================

CREATE TABLE player_stats (
    stat_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id             UUID NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player_id           UUID NOT NULL REFERENCES players(player_id),
    team_id             UUID NOT NULL REFERENCES teams(team_id),
    
    -- Playing time (in seconds for precision)
    seconds_played      INTEGER NOT NULL DEFAULT 0 CHECK (seconds_played >= 0),
    
    -- DNP handling
    did_not_play        BOOLEAN NOT NULL DEFAULT FALSE,
    dnp_reason          dnp_reason,
    
    -- Starter status
    is_starter          BOOLEAN DEFAULT FALSE,
    
    -- Scoring
    points              INTEGER DEFAULT 0 CHECK (points >= 0),
    field_goals_made    INTEGER DEFAULT 0,
    field_goals_attempted INTEGER DEFAULT 0,
    three_pointers_made INTEGER DEFAULT 0,
    three_pointers_attempted INTEGER DEFAULT 0,
    free_throws_made    INTEGER DEFAULT 0,
    free_throws_attempted INTEGER DEFAULT 0,
    
    -- Rebounding
    offensive_rebounds  INTEGER DEFAULT 0,
    defensive_rebounds  INTEGER DEFAULT 0,
    total_rebounds      INTEGER GENERATED ALWAYS AS (offensive_rebounds + defensive_rebounds) STORED,
    
    -- Playmaking & Defense
    assists             INTEGER DEFAULT 0,
    steals              INTEGER DEFAULT 0,
    blocks              INTEGER DEFAULT 0,
    turnovers           INTEGER DEFAULT 0,
    personal_fouls      INTEGER DEFAULT 0 CHECK (personal_fouls <= 6),
    
    -- Advanced (if available from source)
    plus_minus          INTEGER,
    efficiency_rating   DECIMAL(6, 2),  -- PER or similar
    
    -- Computed per-40-minute stats (FIBA scaling)
    -- These are computed via trigger or application layer
    pts_per_40          DECIMAL(5, 2),
    reb_per_40          DECIMAL(5, 2),
    ast_per_40          DECIMAL(5, 2),
    
    -- Metadata
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT unique_player_game UNIQUE (game_id, player_id),
    CONSTRAINT valid_fg CHECK (field_goals_made <= field_goals_attempted),
    CONSTRAINT valid_3pt CHECK (three_pointers_made <= three_pointers_attempted),
    CONSTRAINT valid_ft CHECK (free_throws_made <= free_throws_attempted),
    CONSTRAINT dnp_consistency CHECK (
        (did_not_play = FALSE) OR 
        (did_not_play = TRUE AND seconds_played = 0)
    )
);

-- Indexes for feature engineering queries
CREATE INDEX idx_player_stats_game ON player_stats(game_id);
CREATE INDEX idx_player_stats_player ON player_stats(player_id, game_id);
CREATE INDEX idx_player_stats_team ON player_stats(team_id, game_id);

-- ============================================================================
-- TABLE: odds_history
-- Tracks betting lines from opening to closing, with implied probabilities
-- ============================================================================
-- 
-- Key design decisions:
--   1. Store both decimal odds AND implied probability
--   2. Track line movement via recorded_at timestamp
--   3. Support multiple sportsbooks
--   4. Distinguish opening line, closing line, and interim snapshots
-- ============================================================================

CREATE TABLE odds_history (
    odds_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id             UUID NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    
    -- Sportsbook identification
    sportsbook          VARCHAR(50) NOT NULL,           -- e.g., 'Sportsbet', 'TAB', 'Bet365'
    
    -- Bet details
    bet_type            bet_type NOT NULL,
    selection           VARCHAR(100) NOT NULL,          -- e.g., 'Melbourne United', 'Over 175.5'
    
    -- For spread/total bets
    line_value          DECIMAL(6, 2),                  -- e.g., -4.5, 175.5
    
    -- Odds (stored in decimal format, Australian standard)
    decimal_odds        DECIMAL(8, 3) NOT NULL CHECK (decimal_odds >= 1.0),
    
    -- Implied probability (calculated from odds, before vig removal)
    -- Formula: implied_prob = 1 / decimal_odds
    implied_probability DECIMAL(6, 5) GENERATED ALWAYS AS (1.0 / decimal_odds) STORED,
    
    -- Line timing
    is_opening_line     BOOLEAN DEFAULT FALSE,
    is_closing_line     BOOLEAN DEFAULT FALSE,
    recorded_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Outcome (populated after game completion)
    outcome             bet_outcome DEFAULT 'pending',
    
    -- Metadata
    raw_data            JSONB,                          -- Original API response
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for odds analysis
CREATE INDEX idx_odds_game ON odds_history(game_id, bet_type);
CREATE INDEX idx_odds_sportsbook ON odds_history(sportsbook, game_id);
CREATE INDEX idx_odds_timing ON odds_history(game_id, recorded_at);
CREATE INDEX idx_odds_opening ON odds_history(game_id, bet_type) WHERE is_opening_line = TRUE;
CREATE INDEX idx_odds_closing ON odds_history(game_id, bet_type) WHERE is_closing_line = TRUE;

-- ============================================================================
-- TABLE: bets
-- Tracks our betting activity for performance analysis
-- ============================================================================

CREATE TABLE bets (
    bet_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id             UUID NOT NULL REFERENCES games(game_id),
    odds_id             UUID REFERENCES odds_history(odds_id),
    
    -- Bet details
    bet_type            bet_type NOT NULL,
    selection           VARCHAR(100) NOT NULL,
    line_value          DECIMAL(6, 2),
    decimal_odds        DECIMAL(8, 3) NOT NULL,
    
    -- Staking
    stake_amount        DECIMAL(12, 2) NOT NULL CHECK (stake_amount > 0),
    stake_fraction      DECIMAL(6, 5) NOT NULL,         -- Kelly fraction used
    bankroll_at_bet     DECIMAL(14, 2) NOT NULL,        -- Bankroll when bet placed
    
    -- Model inputs
    model_probability   DECIMAL(6, 5) NOT NULL,         -- Our estimated p
    market_probability  DECIMAL(6, 5) NOT NULL,         -- Implied from odds
    edge_estimate       DECIMAL(6, 5) GENERATED ALWAYS AS (model_probability - market_probability) STORED,
    
    -- Kelly calculation details
    kelly_fraction      DECIMAL(6, 5) NOT NULL,         -- Full Kelly
    kelly_multiplier    DECIMAL(4, 3) NOT NULL,         -- Fractional multiplier used
    
    -- Outcome
    outcome             bet_outcome DEFAULT 'pending',
    pnl                 DECIMAL(12, 2),                 -- Profit/Loss
    
    -- Timestamps
    placed_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    settled_at          TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    notes               TEXT,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance analysis
CREATE INDEX idx_bets_game ON bets(game_id);
CREATE INDEX idx_bets_placed ON bets(placed_at DESC);
CREATE INDEX idx_bets_outcome ON bets(outcome) WHERE outcome != 'pending';

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Function to calculate per-40-minute stats for FIBA scaling
CREATE OR REPLACE FUNCTION calculate_per_40_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Only calculate if player actually played
    IF NEW.seconds_played > 0 THEN
        -- Per-40-minute scaling: stat * (40 * 60 / seconds_played)
        NEW.pts_per_40 := ROUND((NEW.points::DECIMAL * 2400 / NEW.seconds_played), 2);
        NEW.reb_per_40 := ROUND(((COALESCE(NEW.offensive_rebounds, 0) + COALESCE(NEW.defensive_rebounds, 0))::DECIMAL * 2400 / NEW.seconds_played), 2);
        NEW.ast_per_40 := ROUND((NEW.assists::DECIMAL * 2400 / NEW.seconds_played), 2);
    ELSE
        NEW.pts_per_40 := 0;
        NEW.reb_per_40 := 0;
        NEW.ast_per_40 := 0;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-calculate per-40 stats
CREATE TRIGGER trg_calculate_per_40
    BEFORE INSERT OR UPDATE ON player_stats
    FOR EACH ROW
    EXECUTE FUNCTION calculate_per_40_stats();

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply timestamp triggers to relevant tables
CREATE TRIGGER trg_teams_timestamp
    BEFORE UPDATE ON teams FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    
CREATE TRIGGER trg_games_timestamp
    BEFORE UPDATE ON games FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    
CREATE TRIGGER trg_players_timestamp
    BEFORE UPDATE ON players FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    
CREATE TRIGGER trg_player_stats_timestamp
    BEFORE UPDATE ON player_stats FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    
CREATE TRIGGER trg_bets_timestamp
    BEFORE UPDATE ON bets FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for team game-by-game offensive/defensive ratings
CREATE VIEW vw_team_game_ratings AS
WITH game_possessions AS (
    -- Estimate possessions using the common formula
    -- Possessions ≈ FGA - OREB + TO + 0.44 * FTA
    SELECT 
        ps.game_id,
        ps.team_id,
        SUM(ps.field_goals_attempted) - SUM(ps.offensive_rebounds) + 
        SUM(ps.turnovers) + 0.44 * SUM(ps.free_throws_attempted) AS possessions,
        SUM(ps.points) AS points
    FROM player_stats ps
    GROUP BY ps.game_id, ps.team_id
)
SELECT 
    g.game_id,
    g.scheduled_datetime,
    g.season,
    t.team_code,
    gp.possessions,
    gp.points,
    -- Offensive Rating = 100 * (Points / Possessions)
    ROUND(100.0 * gp.points / NULLIF(gp.possessions, 0), 1) AS offensive_rating,
    -- Defensive Rating requires opponent's points
    ROUND(100.0 * opp.points / NULLIF(opp.possessions, 0), 1) AS defensive_rating
FROM game_possessions gp
JOIN games g ON gp.game_id = g.game_id
JOIN teams t ON gp.team_id = t.team_id
LEFT JOIN game_possessions opp ON opp.game_id = gp.game_id AND opp.team_id != gp.team_id
WHERE g.status = 'completed';

-- View for opening vs closing line comparison
CREATE VIEW vw_line_movement AS
SELECT 
    g.game_id,
    g.scheduled_datetime,
    g.home_score,
    g.away_score,
    oh_open.sportsbook,
    oh_open.bet_type,
    oh_open.selection,
    oh_open.decimal_odds AS opening_odds,
    oh_open.implied_probability AS opening_prob,
    oh_close.decimal_odds AS closing_odds,
    oh_close.implied_probability AS closing_prob,
    oh_close.decimal_odds - oh_open.decimal_odds AS odds_movement,
    oh_open.implied_probability - oh_close.implied_probability AS prob_movement
FROM games g
JOIN odds_history oh_open ON g.game_id = oh_open.game_id AND oh_open.is_opening_line = TRUE
JOIN odds_history oh_close ON g.game_id = oh_close.game_id 
    AND oh_close.is_closing_line = TRUE
    AND oh_close.sportsbook = oh_open.sportsbook
    AND oh_close.bet_type = oh_open.bet_type
    AND oh_close.selection = oh_open.selection
WHERE g.status = 'completed';

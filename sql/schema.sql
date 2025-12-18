-- Component 1: Data Infrastructure (SQL Schema)
-- Database: PostgreSQL

-- 1. Games Table
-- Metadata: date, venue, home/away team.
CREATE TABLE games (
    game_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    venue VARCHAR(255) NOT NULL,
    home_team_id INT NOT NULL,
    away_team_id INT NOT NULL,
    home_score INT,
    away_score INT,
    season VARCHAR(10) NOT NULL, -- e.g., '2023-2024'
    is_playoff BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_home_team FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    CONSTRAINT fk_away_team FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

-- 2. Player Stats Table
-- Box scores: must handle "Did Not Play" and 40-min scaling.
CREATE TABLE player_stats (
    stat_id SERIAL PRIMARY KEY,
    game_id INT NOT NULL,
    player_id INT NOT NULL,
    team_id INT NOT NULL,
    minutes_played NUMERIC(4, 1), -- e.g., 35.5
    points INT,
    rebounds INT,
    assists INT,
    steals INT,
    blocks INT,
    turnovers INT,
    fouls INT,

    -- "Did Not Play" (DNP) flag.
    -- If TRUE, minutes_played should be 0 or NULL.
    dnp BOOLEAN DEFAULT FALSE,
    dnp_reason VARCHAR(255), -- e.g., "Injury", "Coach's Decision"

    -- Derived metrics can be computed on the fly or stored.
    -- Storing scaled stats for 40-min FIBA game.
    -- Calculation: (Stat / Minutes) * 40
    points_per_40 NUMERIC(5, 2),
    rebounds_per_40 NUMERIC(5, 2),
    assists_per_40 NUMERIC(5, 2),

    CONSTRAINT fk_game FOREIGN KEY (game_id) REFERENCES games(game_id),
    CONSTRAINT fk_player FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- 3. Odds History Table
-- Must track Opening Line vs. Closing Line.
CREATE TABLE odds_history (
    odds_id SERIAL PRIMARY KEY,
    game_id INT NOT NULL,
    bookmaker_id INT NOT NULL,

    -- Spread (Handicap)
    opening_spread NUMERIC(4, 1), -- e.g., -5.5
    closing_spread NUMERIC(4, 1),

    -- Moneyline (decimal odds)
    opening_moneyline NUMERIC(6, 3), -- e.g., 1.909
    closing_moneyline NUMERIC(6, 3),

    -- Totals (Over/Under)
    opening_total NUMERIC(5, 1),
    closing_total NUMERIC(5, 1),

    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_odds_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Placeholder tables for foreign keys to make the schema valid in isolation
CREATE TABLE teams (
    team_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    code VARCHAR(10) -- e.g., 'PER', 'MEL'
);

CREATE TABLE players (
    player_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    is_import BOOLEAN DEFAULT FALSE -- Critical for NBL analysis
);

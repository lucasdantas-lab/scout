-- =============================================================================
-- SCOUT — Statistical Core for Outcome Understanding Tool
-- Database schema for Supabase (PostgreSQL)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- teams
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,   -- API-Football team id
    name        TEXT    NOT NULL,
    short_name  TEXT,
    city        TEXT
);

-- ---------------------------------------------------------------------------
-- matches
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matches (
    id              INTEGER     PRIMARY KEY,  -- API-Football fixture id
    season          INTEGER     NOT NULL,
    round           TEXT,
    match_date      TIMESTAMPTZ NOT NULL,
    home_team_id    INTEGER     REFERENCES teams(id),
    away_team_id    INTEGER     REFERENCES teams(id),
    home_goals      INTEGER,
    away_goals      INTEGER,
    status          TEXT,        -- 'FT', 'NS', 'LIVE', etc.
    venue           TEXT
);

CREATE INDEX IF NOT EXISTS idx_matches_season       ON matches (season);
CREATE INDEX IF NOT EXISTS idx_matches_match_date   ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_home_team_id ON matches (home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away_team_id ON matches (away_team_id);

-- ---------------------------------------------------------------------------
-- match_stats
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_stats (
    match_id                INTEGER PRIMARY KEY REFERENCES matches(id),
    home_shots              INTEGER,
    away_shots              INTEGER,
    home_shots_on_target    INTEGER,
    away_shots_on_target    INTEGER,
    home_possession         NUMERIC,
    away_possession         NUMERIC,
    home_xg                 NUMERIC,   -- nullable: not always available
    away_xg                 NUMERIC
);

-- ---------------------------------------------------------------------------
-- model_parameters
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_parameters (
    id              SERIAL      PRIMARY KEY,
    run_at          TIMESTAMPTZ DEFAULT NOW(),
    season          INTEGER,
    team_id         INTEGER     REFERENCES teams(id),
    attack          NUMERIC     NOT NULL,
    defense         NUMERIC     NOT NULL,
    parameter_type  TEXT        -- 'posterior_mean' | 'posterior_std'
);

-- ---------------------------------------------------------------------------
-- predictions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL      PRIMARY KEY,
    match_id        INTEGER     REFERENCES matches(id),
    generated_at    TIMESTAMPTZ DEFAULT NOW(),
    model_version   TEXT,
    prob_home       NUMERIC,
    prob_draw       NUMERIC,
    prob_away       NUMERIC,
    prob_btts       NUMERIC,
    prob_over25     NUMERIC,
    score_matrix    JSONB,      -- serialised (MAX_GOALS+1)x(MAX_GOALS+1) grid
    lambda_home     NUMERIC,
    lambda_away     NUMERIC,
    brier_score     NUMERIC     -- filled after the match is played
);

CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions (match_id);

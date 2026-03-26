-- =============================================================================
-- SCOUT — Statistical Core for Outcome Understanding Tool
-- Database schema for Supabase (PostgreSQL)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- teams
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    id              INTEGER PRIMARY KEY,   -- API-Football team id
    name            TEXT    NOT NULL,
    short_name      TEXT,
    city            TEXT,
    altitude_factor NUMERIC DEFAULT 1.0
);

-- ---------------------------------------------------------------------------
-- players
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS players (
    id              INTEGER PRIMARY KEY,   -- API-Football player id
    team_id         INTEGER REFERENCES teams(id),
    name            TEXT    NOT NULL,
    position        TEXT,
    overall_rating  NUMERIC DEFAULT 6.5
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
    home_xg         NUMERIC,     -- nullable: not always available
    away_xg         NUMERIC,
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
    away_possession         NUMERIC
);

-- ---------------------------------------------------------------------------
-- match_lineups
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_lineups (
    id              SERIAL  PRIMARY KEY,
    match_id        INTEGER REFERENCES matches(id),
    team_id         INTEGER REFERENCES teams(id),
    player_id       INTEGER REFERENCES players(id),
    is_starter      BOOLEAN,
    minutes_played  INTEGER,
    rating          NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_match_lineups_match_id ON match_lineups (match_id);
CREATE INDEX IF NOT EXISTS idx_match_lineups_team_id  ON match_lineups (team_id);

-- ---------------------------------------------------------------------------
-- match_events
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_events (
    id              SERIAL  PRIMARY KEY,
    match_id        INTEGER REFERENCES matches(id),
    team_id         INTEGER REFERENCES teams(id),
    event_type      TEXT,    -- 'goal', 'yellow_card', 'red_card', 'substitution'
    minute          INTEGER,
    player_id       INTEGER REFERENCES players(id)
);

-- ---------------------------------------------------------------------------
-- match_context (Claude agent-generated pre-match context)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_context (
    id                  SERIAL      PRIMARY KEY,
    match_id            INTEGER     REFERENCES matches(id),
    generated_at        TIMESTAMPTZ DEFAULT NOW(),
    raw_news            TEXT,
    processed_context   JSONB,
    -- processed_context structure:
    -- {
    --   "home": {"ausencias_confirmadas": [], "duvidas": [], "confirmados_importantes": [],
    --            "lambda_delta": float, "confianca": float, "notas": ""},
    --   "away": {"ausencias_confirmadas": [], "duvidas": [], "confirmados_importantes": [],
    --            "lambda_delta": float, "confianca": float, "notas": ""}
    -- }
    agent_model         TEXT
);

CREATE INDEX IF NOT EXISTS idx_match_context_match_id ON match_context (match_id);

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
    attack_std      NUMERIC,
    defense_std     NUMERIC,
    parameter_type  TEXT        -- 'posterior_mean', 'posterior_std'
);

-- ---------------------------------------------------------------------------
-- predictions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                      SERIAL      PRIMARY KEY,
    match_id                INTEGER     REFERENCES matches(id),
    generated_at            TIMESTAMPTZ DEFAULT NOW(),
    model_version           TEXT,
    prob_home               NUMERIC,
    prob_draw               NUMERIC,
    prob_away               NUMERIC,
    prob_btts               NUMERIC,
    prob_over25             NUMERIC,
    score_matrix            JSONB,      -- serialised (MAX_GOALS+1)x(MAX_GOALS+1) grid
    lambda_home             NUMERIC,
    lambda_away             NUMERIC,
    lambda_home_adjusted    NUMERIC,
    lambda_away_adjusted    NUMERIC,
    narrative               TEXT,
    brier_score             NUMERIC,    -- filled after the match is played
    rps_score               NUMERIC     -- filled after the match is played
);

CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions (match_id);

-- ---------------------------------------------------------------------------
-- calibration_log (Claude calibration agent history)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS calibration_log (
    id                      SERIAL      PRIMARY KEY,
    round_analyzed          TEXT,
    season                  INTEGER,
    generated_at            TIMESTAMPTZ DEFAULT NOW(),
    error_patterns          JSONB,
    suggested_adjustments   JSONB,
    agent_reasoning         TEXT,
    applied                 BOOLEAN     DEFAULT FALSE
);

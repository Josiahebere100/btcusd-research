CREATE TABLE IF NOT EXISTS research_state_observations (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    active_engine_count INTEGER NOT NULL,
    directional_engine_count INTEGER NOT NULL,
    atoms_json JSONB NOT NULL,
    state_json JSONB NOT NULL,
    next_round_id TEXT,
    next_outcome TEXT CHECK (next_outcome IS NULL OR next_outcome IN ('UP','DOWN')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    UNIQUE(session_id, round_id)
);

CREATE TABLE IF NOT EXISTS research_interaction_stats (
    interaction_hash TEXT PRIMARY KEY,
    interaction_key JSONB NOT NULL,
    observation_count BIGINT NOT NULL DEFAULT 0,
    up_count BIGINT NOT NULL DEFAULT 0,
    down_count BIGINT NOT NULL DEFAULT 0,
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS research_predictions (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('UP','DOWN','NO_SIGNAL')),
    p_up DOUBLE PRECISION,
    p_down DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    sway_direction TEXT,
    sway_delta DOUBLE PRECISION,
    matched_interactions JSONB NOT NULL DEFAULT '[]'::jsonb,
    state_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    next_round_id TEXT,
    actual_next_outcome TEXT,
    resolved_at TIMESTAMPTZ,
    UNIQUE(session_id, round_id, engine)
);

CREATE INDEX IF NOT EXISTS idx_research_state_obs_session_round
  ON research_state_observations(session_id, round_id);
CREATE INDEX IF NOT EXISTS idx_research_predictions_session_round
  ON research_predictions(session_id, round_id);

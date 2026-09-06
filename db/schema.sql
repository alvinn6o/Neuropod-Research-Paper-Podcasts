-- Neuropod multi-user schema.
-- Run on a fresh Postgres database. Requires pgvector and pgcrypto.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- Users
-- ============================================================================
-- The id matches the Cognito `sub` (UUID) when Cognito is wired in.
-- In stub-auth mode the server mints a UUID per email on first sign-in.
CREATE TABLE IF NOT EXISTS users (
  id            UUID PRIMARY KEY,
  email         TEXT UNIQUE NOT NULL,
  feed_slug     TEXT UNIQUE NOT NULL,
  display_name  TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_feed_slug ON users(feed_slug);

-- ============================================================================
-- Per-user topic preferences
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_topics (
  user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  topic      TEXT NOT NULL,
  position   INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id, topic)
);

-- Optional arXiv category filters (e.g., 'cs.AI', 'q-bio.QM', 'hep-th').
-- Combined with topics as: (cat:X OR cat:Y) AND (all:"topic1" OR all:"topic2").
-- Empty = "any category, just match topics."
CREATE TABLE IF NOT EXISTS user_categories (
  user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  category   TEXT NOT NULL,
  position   INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id, category)
);

-- ============================================================================
-- Papers (shared across users - same arXiv paper isn't re-extracted per user)
-- ============================================================================
CREATE TABLE IF NOT EXISTS papers (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  arxiv_id        TEXT UNIQUE NOT NULL,
  title           TEXT NOT NULL,
  authors         TEXT[] NOT NULL DEFAULT '{}',
  abstract        TEXT NOT NULL,
  categories      TEXT[] NOT NULL DEFAULT '{}',
  published_at    TIMESTAMPTZ NOT NULL,
  pdf_url         TEXT,
  citation_count  INT NOT NULL DEFAULT 0,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_papers_published_at ON papers(published_at DESC);

-- ============================================================================
-- Paper chunks with embeddings (shared)
-- ============================================================================
CREATE TABLE IF NOT EXISTS paper_chunks (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  paper_id     UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  section      TEXT NOT NULL,
  chunk_index  INT NOT NULL,
  content      TEXT NOT NULL,
  token_count  INT NOT NULL,
  token_source TEXT NOT NULL DEFAULT 'unknown',   -- tiktoken:<enc> | estimate:<rule>
  embedding    vector(1536),
  -- Which model produced `embedding`. NOT a cosmetic label: vectors from two
  -- different models are not comparable, so a mixed index silently returns
  -- garbage. Retrieval filters on this, and NULL embedding means "not embedded"
  -- rather than a zero vector.
  embedding_model TEXT,
  embedding_dim   INT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper ON paper_chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON paper_chunks USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- Episodes (per-user - script + optional audio belong to whoever generated them)
-- ============================================================================
CREATE TABLE IF NOT EXISTS episodes (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  paper_id        UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  title           TEXT NOT NULL,
  description     TEXT NOT NULL,
  topic           TEXT NOT NULL,
  score           DOUBLE PRECISION NOT NULL DEFAULT 0,
  script          TEXT NOT NULL,
  qa_status       TEXT NOT NULL DEFAULT 'verified',
  qa_notes        TEXT,
  duration_secs   INT NOT NULL DEFAULT 0,
  llm_provider    TEXT NOT NULL DEFAULT 'demo',
  tts_provider    TEXT NOT NULL DEFAULT 'none',
  audio_key       TEXT,                   -- null until optional TTS generation runs
  audio_mime      TEXT NOT NULL DEFAULT 'audio/wav',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_episodes_user_created ON episodes(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_paper ON episodes(paper_id);

-- ============================================================================
-- Engagement events
-- ============================================================================
CREATE TABLE IF NOT EXISTS feedback_events (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  episode_id    UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  event_type    TEXT NOT NULL,           -- play | pause | skip | complete
  position_secs INT NOT NULL DEFAULT 0,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_user_created
  ON feedback_events(user_id, created_at DESC);

-- ============================================================================
-- Async pipeline jobs (web -> worker)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pipeline_jobs (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  status        TEXT NOT NULL DEFAULT 'queued', -- queued|running|done|error
  window_days   INT NOT NULL DEFAULT 7,
  topics        TEXT[] NOT NULL DEFAULT '{}',
  episode_count INT NOT NULL DEFAULT 3,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at    TIMESTAMPTZ,
  finished_at   TIMESTAMPTZ,
  error         TEXT,
  result_count  INT NOT NULL DEFAULT 0,
  skipped_count INT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_user_created
  ON pipeline_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created
  ON pipeline_jobs(status, created_at DESC);

-- ============================================================================
-- Stub-mode auth sessions (Cognito mode validates JWT directly, no row needed)
-- ============================================================================
CREATE TABLE IF NOT EXISTS auth_sessions (
  token       TEXT PRIMARY KEY,
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON auth_sessions(user_id);

-- ============================================================================
-- Per-user rate limiting (simple counter, day bucket)
-- ============================================================================
CREATE TABLE IF NOT EXISTS rate_limits (
  user_id   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  bucket    TEXT NOT NULL,         -- e.g. 'pipeline_run', 'ask'
  day       DATE NOT NULL,
  count     INT NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, bucket, day)
);

-- ============================================================================
-- LLM call ledger. One row per provider call, including failures.
-- Feeds: cost-per-episode, the monthly budget kill-switch, p50/p95 latency,
-- and the cost side of any model-routing decision.
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_calls (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID REFERENCES users(id) ON DELETE SET NULL,
  purpose             TEXT NOT NULL,        -- script | ask | judge | embed
  provider            TEXT NOT NULL,        -- anthropic | openai | bedrock | demo
  model               TEXT NOT NULL,
  prompt_tokens       INT NOT NULL DEFAULT 0,
  completion_tokens   INT NOT NULL DEFAULT 0,
  cached_read_tokens  INT NOT NULL DEFAULT 0,
  cached_write_tokens INT NOT NULL DEFAULT 0,
  cost_usd            DOUBLE PRECISION,     -- NULL = model has no price entry
  latency_ms          INT NOT NULL DEFAULT 0,
  ok                  BOOLEAN NOT NULL DEFAULT TRUE,
  status              INT,
  error               TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_user_created ON llm_calls(user_id, created_at DESC);

-- ============================================================================
-- Retrieval traces. Which chunks actually grounded each script, and why.
-- The orchestrator used to compute this, pass it to the writer, and drop it —
-- so there was no way to attribute a claim to a chunk, replay a retrieval
-- offline, or build training data for a reranker. This is that record.
-- ============================================================================
CREATE TABLE IF NOT EXISTS retrieval_traces (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  episode_id        UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  chunk_id          UUID NOT NULL,
  query_text        TEXT NOT NULL,
  retriever_version TEXT NOT NULL,
  rank_position     INT NOT NULL,          -- 0-based, after scoring
  dense_score       DOUBLE PRECISION,
  sparse_score      DOUBLE PRECISION,
  section_bonus     DOUBLE PRECISION,
  final_score       DOUBLE PRECISION NOT NULL,
  used_in_prompt    BOOLEAN NOT NULL DEFAULT FALSE,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_traces_episode ON retrieval_traces(episode_id, rank_position);
CREATE INDEX IF NOT EXISTS idx_traces_chunk ON retrieval_traces(chunk_id);

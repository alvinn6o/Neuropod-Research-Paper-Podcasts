CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS user_topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    category TEXT NOT NULL,
    keywords TEXT[],
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL,
    abstract TEXT,
    categories TEXT[],
    published_at TIMESTAMPTZ,
    pdf_url TEXT,
    citation_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
    section TEXT,
    chunk_index INT,
    content TEXT NOT NULL,
    token_count INT,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS paper_chunks_embedding_hnsw
    ON paper_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID REFERENCES papers(id),
    user_id UUID,
    title TEXT NOT NULL,
    description TEXT,
    script TEXT,
    audio_url TEXT,
    duration_secs INT,
    tts_provider TEXT,
    qa_status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    episode_id UUID REFERENCES episodes(id),
    event_type TEXT NOT NULL,
    position_secs INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_affinities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    category TEXT NOT NULL,
    affinity FLOAT DEFAULT 0.0,
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, category)
);

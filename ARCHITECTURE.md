# Architecture

## System Overview

PaperPod is a **batch-generative pipeline** with a **real-time serving layer**. The pipeline runs daily on a schedule, producing podcast episodes from discovered papers. The serving layer handles API requests, audio delivery, RSS feeds, and real-time RAG queries.

This separation is intentional: episode generation is expensive (LLM + TTS API calls) and latency-insensitive, so it runs as a batch job. Serving is latency-sensitive and cheap, so it runs as a standard web API.

```
                    ┌─────────────┐
                    │  Scheduler  │
                    │ (APScheduler)│
                    └──────┬──────┘
                           │ daily trigger
                           ▼
┌─────────────────────────────────────────────────┐
│              BATCH PIPELINE                      │
│                                                  │
│  discover → extract → chunk → embed → script     │
│  → QA → TTS → post-process → store              │
│                                                  │
│  Runs once/day, ~5-15 min per episode            │
└─────────────────────┬───────────────────────────┘
                      │ writes episodes to DB + storage
                      ▼
┌─────────────────────────────────────────────────┐
│              SERVING LAYER                       │
│                                                  │
│  FastAPI:                                        │
│    /episodes   — browse + filter                 │
│    /feed       — RSS XML                         │
│    /ask        — real-time RAG                   │
│    /feedback   — engagement logging              │
│    /topics     — preference management           │
│                                                  │
│  Serves pre-generated content (fast)             │
│  RAG queries hit pgvector in real-time           │
└─────────────────────────────────────────────────┘
```

---

## Pipeline Deep Dive

### Stage 1: Discovery

**Goal:** Find the most relevant and interesting papers for each user, daily.

**Inputs:** User topic preferences, arXiv categories, date range.

**Process:**
1. Query arXiv API for papers published in the last 1–3 days across user's subscribed categories
2. Query Semantic Scholar for citation metadata on candidates (citation count, influential citation count, publication date)
3. Score each paper:

```
paper_score = w1 * recency_score + w2 * trending_score + w3 * user_affinity_score

where:
  recency_score   = exponential decay from publish date (half-life: 3 days)
  trending_score  = citation_velocity / max(days_since_published, 1)
  user_affinity   = cosine_sim(paper_abstract_embedding, user_interest_embedding)
```

4. Select top-K papers (default K=3 per user per day)

**Why this scoring approach:**
- Recency ensures freshness (this is a *daily* podcast, not a literature review)
- Citation velocity catches breakout papers that the community is engaging with
- User affinity personalizes without needing extensive history — even a new user's topic keywords produce a workable interest embedding

**Edge cases:**
- New user with no history → fall back to trending papers in selected categories
- Slow news day (few papers in user's niche) → broaden to adjacent categories with reduced affinity weight
- Duplicate papers across sources → deduplicate on arXiv ID

---

### Stage 2: Extraction + Chunking

**Goal:** Turn a PDF into structured, retrievable text chunks.

**Process:**
1. Download PDF from arXiv (`https://arxiv.org/pdf/{arxiv_id}.pdf`)
2. Extract text with PyMuPDF — page by page, preserving reading order
3. Detect section boundaries using regex patterns on common academic headers:
   - Primary: `Abstract`, `Introduction`, `Related Work`, `Methods`/`Methodology`, `Experiments`/`Results`, `Discussion`, `Conclusion`
   - Numbered sections: `1. Introduction`, `2. Background`, etc.
4. Chunk by section. If a section exceeds 1500 tokens, split at paragraph boundaries with 200-token overlap
5. Tag each chunk: `{paper_id, section_label, chunk_index, token_count, text}`

**Why section-aware chunking:**
Naive fixed-size chunking splits mid-paragraph and loses structural context. When the scriptwriter asks "what were the results?", section-aware retrieval returns the actual Results section — not a fragment that starts mid-sentence in Methods and ends mid-sentence in Results.

**Known limitations:**
- Multi-column PDFs: PyMuPDF handles most, but some complex layouts produce garbled text. Fallback: treat as single-column extraction, accept noise.
- Equations: rendered as best-effort plaintext. Heavy math papers will have lower script quality — this is acceptable for v1.
- Figures/tables: skipped entirely in v1. Captions are extracted if detected.
- Non-English papers: out of scope for v1.

---

### Stage 3: Embedding + Storage

**Goal:** Make paper chunks semantically searchable.

**Model:** OpenAI `text-embedding-3-small` (1536 dimensions)
- Chosen over local models for retrieval quality on academic text
- Cost is minimal: ~$0.002 per paper (a few hundred chunks across all sections)
- Alternative: `text-embedding-3-large` for higher quality at 2x cost — benchmark later

**Storage:** pgvector extension in Supabase PostgreSQL
- Chunks stored in `paper_chunks` table with an `embedding` vector column
- HNSW index for approximate nearest neighbor search (`lists=100` for expected corpus size)
- Co-located with all other app data — no separate vector DB service to manage

**Why pgvector over ChromaDB:**
The Job Tracker project already uses ChromaDB. Using pgvector here demonstrates range and is operationally simpler — one database for everything. For a portfolio project, showing you can work with multiple vector store backends is a plus.

---

### Stage 4: Script Generation (RAG)

**Goal:** Produce a natural, conversational podcast script grounded in the paper's content.

**Retrieval:**
- Query: paper's title + abstract (as a semantic summary of the full paper)
- Retrieve: top 8–12 chunks from that paper's embeddings, prioritizing abstract + results + conclusion sections
- Deduplication: remove near-duplicate chunks (cosine sim > 0.95)

**Generation prompt structure:**

```
System: You are a podcast scriptwriter who turns academic research papers
into engaging, accessible audio episodes. Write scripts that sound natural
when read aloud — conversational, clear, with varied sentence length. Avoid
jargon without explanation. Target: 800-1200 words (5-8 min spoken).

Context: [retrieved chunks with section labels]

Paper metadata: [title, authors, date, arXiv category]

Write a podcast script with these segments:
1. HOOK (2-3 sentences): Why should a busy ML practitioner care about this?
   Lead with the real-world implication, not the paper title.
2. CONTEXT (3-4 sentences): What problem are they solving? What's been
   tried before?
3. KEY FINDINGS (main body): What did they discover? Use analogies.
   Explain one or two results in depth rather than listing everything.
4. HOW (2-3 sentences): Simplified methods — the intuition, not the math.
5. SO WHAT (2-3 sentences): Implications. Limitations the authors acknowledge.
   Open questions.
6. WRAP (1-2 sentences): What to watch for next. Suggest what kind of
   follow-up work this enables.

Do NOT use segment labels in the script — it should flow as continuous speech.
Do NOT start with "Welcome to..." or any podcast intro cliche.
```

**Why Claude Sonnet for generation:**
Script quality is the make-or-break for this product. The script needs to sound like a knowledgeable friend explaining a paper, not a summary bot. Sonnet handles long-form conversational generation well and follows structural prompts reliably. Cost per episode: ~$0.01–0.03 depending on chunk count.

---

### Stage 5: Script QA

**Goal:** Catch hallucinations before they reach audio.

**Process:**
1. Second LLM call with a verification prompt:

```
Given the source material and the generated script, identify any claims
in the script that are NOT supported by the source material. For each
unsupported claim, state what the script says and what the source
actually says (or that the source doesn't address it).

If all claims are supported, respond with: VERIFIED
```

2. If unsupported claims found:
   - Auto-revise: re-generate the specific segment with tighter grounding instructions
   - If second attempt still fails: flag for manual review, still generate episode but add disclaimer

**Why a separate QA pass:**
LLMs are prone to "reasonable-sounding but fabricated" details when summarizing technical content — plausible method descriptions, invented metrics, wrong attribution. A verification pass catches the most egregious cases. It's not perfect, but it's a meaningful safety layer.

---

### Stage 6: Audio Synthesis

**Goal:** Convert script text to natural-sounding speech with podcast-quality production.

**TTS providers (pluggable):**

| Provider | Pros | Cons | Cost |
|----------|------|------|------|
| ElevenLabs | Expressive, multi-voice, SSML support | Higher cost, rate limits | ~$0.15/episode |
| OpenAI TTS | Simple API, consistent quality | Less expressive, fewer voices | ~$0.03/episode |

**Post-processing pipeline (pydub + pyloudnorm):**
1. Load TTS output audio
2. Prepend intro music: 3s bed with 1s fade-in, duck to -20dB under speech
3. Append outro music: 3s bed with 2s fade-out
4. Normalize loudness to -16 LUFS (podcast standard per Apple/Spotify specs)
5. Export as MP3, 128kbps, 44.1kHz
6. Write ID3 tags: title, artist ("PaperPod"), album (topic category), description (abstract snippet), duration

**Audio storage:**
- Upload MP3 to Supabase Storage bucket (`episodes/`)
- Store public URL in `episodes` table
- Signed URLs optional for private feeds (stretch goal)

---

## Data Model

```sql
-- Users + auth handled by Supabase Auth

CREATE TABLE user_topics (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES auth.users(id),
    category    TEXT NOT NULL,          -- e.g., "cs.LG", "cs.CL"
    keywords    TEXT[],                 -- custom keywords
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE papers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arxiv_id        TEXT UNIQUE NOT NULL,
    title           TEXT NOT NULL,
    authors         TEXT[] NOT NULL,
    abstract        TEXT,
    categories      TEXT[],
    published_at    TIMESTAMPTZ,
    pdf_url         TEXT,
    citation_count  INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE paper_chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id    UUID REFERENCES papers(id) ON DELETE CASCADE,
    section     TEXT,                   -- "abstract", "methods", "results", etc.
    chunk_index INT,
    content     TEXT NOT NULL,
    token_count INT,
    embedding   VECTOR(1536),          -- pgvector
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON paper_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE episodes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id        UUID REFERENCES papers(id),
    user_id         UUID REFERENCES auth.users(id),  -- NULL for public/demo episodes
    title           TEXT NOT NULL,
    description     TEXT,
    script          TEXT,               -- full generated script
    audio_url       TEXT,               -- Supabase Storage URL
    duration_secs   INT,
    tts_provider    TEXT,               -- "elevenlabs" or "openai"
    qa_status       TEXT DEFAULT 'pending',  -- "pending", "verified", "flagged"
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE feedback_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES auth.users(id),
    episode_id  UUID REFERENCES episodes(id),
    event_type  TEXT NOT NULL,          -- "play", "pause", "skip", "complete"
    position_secs   INT,               -- playback position when event fired
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE user_affinities (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES auth.users(id),
    category    TEXT NOT NULL,
    affinity    FLOAT DEFAULT 0.0,      -- computed nightly
    updated_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, category)
);
```

---

## Cost Estimates (per episode)

| Component | Provider | Estimated Cost |
|-----------|----------|---------------|
| Embeddings | OpenAI text-embedding-3-small | $0.002 |
| Script generation | Claude Sonnet | $0.02 |
| Script QA | Claude Sonnet | $0.01 |
| TTS (primary) | ElevenLabs | $0.15 |
| TTS (budget) | OpenAI TTS | $0.03 |
| Storage | Supabase (5MB MP3) | ~$0 (within free tier) |
| **Total (ElevenLabs)** | | **~$0.18/episode** |
| **Total (OpenAI TTS)** | | **~$0.06/episode** |

At 3 episodes/day for 1 user: **$0.18–$0.54/day** (ElevenLabs) or **$0.06–$0.18/day** (OpenAI TTS).

---

## Security + Privacy

- **Auth:** Supabase Auth with RLS (Row Level Security) — users only see their own episodes, topics, and feedback
- **RSS feeds:** UUID-based slugs, not sequential IDs. Feeds are technically public (RSS readers can't auth), but URLs are unguessable
- **API keys:** stored in environment variables, never committed. `.env.example` shows required keys without values
- **Paper content:** sourced from public arXiv (open access). No paywalled content ingestion in v1
- **Audio storage:** Supabase Storage with public bucket for episode audio (required for RSS enclosures)

---

## Scaling Notes (Beyond Portfolio Scope)

For a production system serving many users, the architecture would evolve:
- **Pipeline:** move from APScheduler to a task queue (Celery + Redis) for parallel episode generation
- **Caching:** shared episodes for popular papers — don't regenerate the same paper for every user
- **CDN:** CloudFront or similar for audio file delivery
- **Embeddings:** batch embedding API calls, consider caching common paper embeddings
- **Personalization:** upgrade from weighted formula to a lightweight collaborative filtering model
- **Multi-language:** add translation pipeline stage between script generation and TTS

These are noted for interview discussion but are explicitly out of scope for the portfolio build.

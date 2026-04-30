# Build Plan

## Phase 1 — Paper Ingestion Pipeline (Week 1–2)

The foundation. No UI, just a CLI that discovers papers, extracts content, and stores embeddings.

### 1.1 Paper Discovery
- [ ] arXiv API client: query by category (cs.LG, cs.CL, cs.AI, etc.) and date range
- [ ] Semantic Scholar API client: fetch papers by keyword, get citation counts and velocity
- [ ] Candidate ranker: score papers by recency + trending signal (citation velocity / time since published)
- [ ] CLI: `python discover.py --topics "reinforcement learning, LLM" --days 3 --top-k 5`
- [ ] Output: list of `PaperCandidate` objects (title, abstract, authors, arxiv_id, score, pdf_url)

### 1.2 PDF Extraction + Chunking
- [ ] PDF downloader: fetch from arXiv PDF endpoint
- [ ] PyMuPDF extractor: raw text extraction with page boundaries
- [ ] Section-aware chunker:
  - Detect section headers (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
  - Fall back to fixed-size chunking with overlap if headers not detected
  - Each chunk tagged with: paper_id, section_label, chunk_index, text
- [ ] Handle edge cases: multi-column layouts, inline equations (extract as plaintext), figures (skip gracefully)

### 1.3 Embedding + Storage
- [ ] Supabase setup: PostgreSQL with pgvector extension enabled
- [ ] Schema: papers table, chunks table (with embedding column), episodes table (empty for now)
- [ ] Embed each chunk using OpenAI `text-embedding-3-small`
- [ ] Store paper metadata + embedded chunks in Supabase
- [ ] Batch processing with rate limiting and progress bar

### 1.4 Verification
- [ ] Ingest 10 papers across 2–3 categories
- [ ] Verify: sections detected correctly, embeddings stored, similarity search returns relevant chunks
- [ ] CLI: `python search.py "attention mechanism efficiency" --paper-id <id>` → returns top chunks

**Deliverable:** CLI that discovers papers, extracts + chunks PDFs, and stores searchable embeddings.

---

## Phase 2 — Script Generation + Audio (Week 3–4)

Turn retrieved paper content into listenable podcast episodes.

### 2.1 RAG Script Generation
- [ ] Retriever: given a paper, pull its most relevant chunks (abstract + key results + methods summary)
- [ ] Script template design — prompt engineering for conversational tone:
  - **Hook** (30s): why should the listener care? real-world relevance
  - **Context** (1min): what problem does this solve? prior work in plain language
  - **Key findings** (2–3min): main results, explained simply with analogies
  - **How they did it** (1–2min): methods simplified, skip heavy math
  - **So what?** (1min): implications, limitations, open questions
  - **Connection** (30s): tie to listener's interests / related recent work
- [ ] LLM call (Claude Sonnet): system prompt with script template + retrieved chunks as context
- [ ] Target episode length: 5–8 minutes of spoken content (~800–1200 words)

### 2.2 Script QA
- [ ] Hallucination check: second LLM pass comparing generated script against source chunks
  - Flag any claims not grounded in retrieved content
  - Auto-revise or flag for review
- [ ] Readability check: ensure script sounds natural when read aloud (no dense jargon blocks, proper transitions)

### 2.3 TTS Synthesis
- [ ] TTS provider abstraction (base class with `synthesize(script) -> audio_bytes`)
- [ ] ElevenLabs implementation: voice selection, pacing control, SSML tags for emphasis
- [ ] OpenAI TTS implementation: simpler API, lower cost baseline
- [ ] Config flag to switch providers

### 2.4 Audio Post-Processing
- [ ] pydub pipeline:
  - Prepend intro music bed (2–3s fade in, duck under speech)
  - Append outro with fade out
  - Normalize loudness to -16 LUFS (podcast standard) using pyloudnorm
- [ ] Export as MP3 (128kbps) with ID3 tags (title, author, description, artwork)
- [ ] Upload to Supabase Storage (or S3), store URL in episodes table

### 2.5 Verification
- [ ] Generate 3 episodes from ingested papers
- [ ] Listen and evaluate: does it sound natural? are facts accurate? is pacing good?
- [ ] CLI: `python generate.py --paper-id <id> --tts elevenlabs` → produces MP3

**Deliverable:** CLI that takes a paper and produces a polished podcast episode MP3.

---

## Phase 3 — API + RSS Feed (Week 5–6)

Serve episodes and enable podcast app subscriptions.

### 3.1 Core API
- [ ] FastAPI app with CORS, auth middleware (Supabase Auth JWT)
- [ ] `GET /episodes` — list episodes, filter by topic/date, paginate
- [ ] `GET /episodes/{id}` — episode details + audio URL + source paper link
- [ ] `GET /episodes/{id}/paper` — source paper metadata + abstract
- [ ] `POST /topics` — set user topic preferences
- [ ] `GET /topics` — get current preferences

### 3.2 RSS Feed
- [ ] `GET /feed/{user_id}` — generate RSS XML with feedgen library
- [ ] Include: enclosure (MP3 URL), title, description, pub date, duration
- [ ] Validate against Apple Podcasts and Spotify RSS specs
- [ ] Unauthenticated endpoint (RSS readers can't send JWTs) — use user-specific UUID slug

### 3.3 Deep Dive RAG Endpoint
- [ ] `POST /episodes/{id}/ask` — natural language question about the source paper
  - Retrieve relevant chunks from that paper's embeddings
  - LLM generates grounded answer with chunk citations
  - Return answer + cited sections + page references
- [ ] Rate limit to prevent abuse

### 3.4 Scheduled Pipeline
- [ ] APScheduler job: run discovery + generation pipeline daily at configured time (e.g., 5 AM UTC)
- [ ] Per-user topic preferences feed into discovery ranker
- [ ] Generate N episodes per user per day (configurable, default 3)
- [ ] Error handling: retry failed papers, skip if PDF extraction fails, alert on TTS API errors

### 3.5 Verification
- [ ] Test all endpoints via Swagger UI
- [ ] Subscribe to RSS feed in a podcast app — verify episodes appear
- [ ] Trigger pipeline manually, confirm new episodes show up in feed within minutes

**Deliverable:** Working API with RSS feed — subscribe in your podcast app and hear AI-generated episodes.

---

## Phase 4 — Frontend + Feedback (Week 7–8)

Web dashboard for browsing, listening, and preference management.

### 4.1 Setup
- [ ] Next.js 14 (App Router) + TypeScript + Tailwind
- [ ] PWA configuration: manifest.json, service worker for offline cached episodes
- [ ] Supabase Auth integration (magic link or OAuth)
- [ ] API client module

### 4.2 Daily Feed
- [ ] Home page: today's episodes as cards (title, topic tag, duration, source paper)
- [ ] Persistent audio player bar at bottom (play/pause, scrub, speed control)
- [ ] Episode detail page: description, source paper link, audio player

### 4.3 Topic Preferences
- [ ] Topic selector: browse arXiv categories + custom keyword input
- [ ] Save preferences → API → influences next pipeline run
- [ ] "Explore" page: trending papers outside your usual topics (discovery mode)

### 4.4 Deep Dive Chat
- [ ] Per-episode "Ask about this paper" panel
- [ ] Chat interface: question → RAG response with cited sections
- [ ] Show which paper sections were used as context

### 4.5 Implicit Feedback
- [ ] Track events client-side: play, pause, skip, complete, time-listened
- [ ] `POST /feedback` — log events with timestamps
- [ ] Lightweight feedback hook: `useFeedback()` wraps audio player events

### 4.6 RSS Subscription UX
- [ ] "Subscribe" page: show personal RSS URL
- [ ] One-tap buttons for Apple Podcasts, Spotify, Overcast (deep links with RSS URL)
- [ ] Copy-to-clipboard fallback

**Deliverable:** Full web dashboard — browse, listen, personalize, ask questions.

---

## Phase 5 — Personalization + Polish (Week 9–10)

Close the feedback loop and deploy.

### 5.1 User Affinity Model
- [ ] Nightly job: aggregate feedback events per user per topic
- [ ] Compute affinity scores: topic_affinity = f(plays, completions, skips, recency)
- [ ] Simple weighted formula v1 (no ML model needed yet):
  - `affinity = completions * 1.0 + plays * 0.5 - skips * 0.3`
  - Decay older signals with exponential time weighting
- [ ] Feed affinity scores into discovery ranker as user_score component
- [ ] Log ranked candidates + final selections for future evaluation

### 5.2 Deployment
- [ ] Deploy FastAPI to Railway (with APScheduler running in-process)
- [ ] Deploy Next.js to Vercel
- [ ] Supabase hosted instance for DB + storage + auth
- [ ] Environment variable management across services
- [ ] Demo mode: pre-generated episodes across ML/AI/NLP, no auth required

### 5.3 Docker
- [ ] Dockerfile for backend (multi-stage: slim Python image)
- [ ] docker-compose.yml: backend + local PostgreSQL with pgvector
- [ ] Document local setup in README

### 5.4 README + Portfolio
- [ ] Record demo video: subscribe → wake up → listen to episode → ask deep dive question
- [ ] Architecture diagram (polished version)
- [ ] Screenshots of dashboard, player, topic selector
- [ ] "AI/ML Concepts" section explaining each pipeline stage
- [ ] Cost analysis: per-episode cost breakdown (embedding + LLM + TTS)

### 5.5 Stretch Goals
- [ ] Multi-voice episodes: host + guest style using two TTS voices for a dialogue format
- [ ] Episode series: cluster related papers into multi-episode arcs
- [ ] Transcript display synced with audio playback
- [ ] Email digest: morning email with episode links alongside RSS
- [ ] Collaborative feeds: shared topic subscriptions for research groups
- [ ] Evaluation harness: compare script quality across LLMs (Claude vs GPT-4 vs Gemini)

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Paper sources | arXiv + Semantic Scholar | arXiv for breadth, S2 for citation signals |
| PDF extraction | PyMuPDF | Fast, handles academic layouts, pure Python |
| Chunking strategy | Section-aware | Preserves paper structure, better retrieval than naive splitting |
| Embedding model | OpenAI text-embedding-3-small | Best quality/cost ratio for retrieval |
| Vector store | pgvector (Supabase) | Single database for everything, no extra infra |
| Script LLM | Claude Sonnet | Best conversational generation quality |
| TTS | ElevenLabs (primary), OpenAI TTS (fallback) | Quality vs cost configurable |
| Audio format | MP3 128kbps, -16 LUFS | Podcast industry standard |
| Delivery | RSS feed + web player | Works with existing podcast apps, no custom mobile app needed |
| Frontend | Next.js PWA | Offline caching, installable, push notifications without native app |
| Personalization | Weighted affinity formula | Interpretable, no ML model needed v1, easy to upgrade later |
| Scheduling | APScheduler in-process | Simple, avoids Celery/Redis overhead for a portfolio project |
| Auth | Supabase Auth | Integrated with DB, magic link = no password management |
| Storage | Supabase Storage | Audio files alongside DB, single vendor |
| Deploy | Railway + Vercel + Supabase | Consistent with existing project stack |

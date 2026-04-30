# Neuropod AI

Research papers, distilled into audio.

Neuropod ranks new arXiv papers against your topics, generates a 90-second narrated brief grounded in the paper, and serves it through a clean dashboard and RSS feed. It runs end-to-end with zero API keys (demo mode), and progressively upgrades each stage as you add provider keys.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Configure (optional — runs in demo mode without any keys)
cp .env.example .env

# 3. Boot
# Terminal A — API
python -m uvicorn api.main:app --reload --port 8000
# Terminal B — Frontend
cd frontend && npm run dev
```

Open <http://localhost:3000>. The status badge in the top-right shows whether you're running `demo` or `live`.

Or with Docker: `docker-compose up`.

---

## What you need to set up

Neuropod is fully functional with **no setup**. To upgrade each stage:

| Capability | Provider | Env var | Where to get it |
|---|---|---|---|
| **Script generation** (recommended) | Anthropic Claude | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> |
| Script generation (alt) | OpenAI | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |
| **TTS — expressive** (recommended) | ElevenLabs | `ELEVENLABS_API_KEY` | <https://elevenlabs.io/app/settings/api-keys> |
| TTS — fast & cheap | OpenAI | `OPENAI_API_KEY` | (same key as above) |
| Live arXiv discovery | arXiv API | `NEUROPOD_LIVE_DISCOVERY=true` | (no key — public API) |
| Production storage | Supabase | `SUPABASE_URL`, `SUPABASE_KEY` | <https://supabase.com/dashboard> |

Routing is automatic: if a key is present, that provider is used. The TTS provider preference can be overridden with `NEUROPOD_TTS_PROVIDER=elevenlabs|openai|demo`.

### Minimum recommended setup

For a polished daily podcast, get one of each:

```bash
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=...
NEUROPOD_LIVE_DISCOVERY=true
```

That covers script writing, expressive narration, and live paper discovery. Approx cost per episode: ~$0.02 LLM + ~$0.15 TTS.

---

## How it works

```
arXiv / Semantic Scholar  →  ranker (recency + citations + affinity)
        ↓
PDF extraction → section-aware chunking → embeddings
        ↓
Retriever (top-k) → ScriptWriter (Claude/GPT) → QA check
        ↓
TTS (ElevenLabs / OpenAI) → audio cache → RSS feed + dashboard
```

Each stage has a real implementation (used when keys are set) and a deterministic demo fallback (used otherwise). The fallback is good enough to demo the full UX without spending a cent.

---

## Layout

```
api/                    FastAPI app
  config.py             Settings (reads .env, with PAPERPOD_* legacy aliases)
  routes/
    episodes.py         List, get, stream audio (with disk cache)
    feed.py             RSS XML
    ask.py              Grounded Q&A on an episode
    topics.py           User topic preferences
    feedback.py         Play/pause/skip events
    status.py           Provider health & demo/live indicator
  storage.py            Local JSON store (DemoStore)

pipeline/               The discover → script → audio pipeline
  discover/             arXiv client (live + demo), ranker, citation enrichment
  ingest/               PDF extraction, chunker
  generate/             Embedder, retriever, ScriptWriter (Claude/GPT/demo), QA
  synthesize/           TTSProvider (ElevenLabs/OpenAI/demo), audio metadata
  orchestrator.py       Single entry point: build_demo_payload()

frontend/src/
  app/                  Next.js App Router pages: feed, explore, episode, topics, subscribe
  components/
    AudioPlayer         Custom player: scrub, skip, speed, keyboard shortcuts
    EpisodeCard         Compact card with QA badge + inline play
    ExploreView         Search + topic filter + sort
    DeepDiveChat        Grounded Q&A with citations
    TopicSelector       Add/remove tracked topics
    Toast / StatusBadge System UI
```

---

## Keyboard shortcuts (player)

| Key | Action |
|---|---|
| <kbd>Space</kbd> | Play / pause |
| <kbd>J</kbd> / <kbd>←</kbd> | Skip back 15s |
| <kbd>L</kbd> / <kbd>→</kbd> | Skip forward 30s |

Click the speed indicator (`1×`) on the player to cycle through 1× → 1.25× → 1.5× → 1.75× → 2× → 0.85×.

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/status` | Provider configuration + demo/live state |
| GET | `/episodes` | List episodes (filter by `?topic=`) |
| GET | `/episodes/{id}` | Single episode with paper + script |
| GET | `/episodes/{id}/audio` | Stream audio (cached on disk by script hash) |
| POST | `/episodes/{id}/ask` | Ask a grounded question (returns answer + citations) |
| GET | `/feed/{user_slug}` | RSS XML |
| GET/POST | `/topics` | Read/write tracked topics |
| POST | `/feedback` | Log play/pause/skip/complete events |

---

## Refresh demo data

```bash
python scripts/seed_demo.py
```

This re-runs the pipeline against the seeded catalog and replaces `data/demo_store.json`. Once you set a provider key and rerun, the `tts_provider` field on each episode reflects the real provider used.

---

## Development notes

- The audio route writes synthesized bytes to `data/audio_cache/<sha1>.bin` keyed by `(provider, script)`, so re-streams don't burn TTS credits.
- The frontend is fully server-rendered for the initial pages; only the player, search, and chat are client components.
- Environment variables accept both `NEUROPOD_*` and legacy `PAPERPOD_*` names so existing `.env` files keep working.
- Type-check the frontend with `cd frontend && npx tsc --noEmit`.
- Smoke-test the API with `python -c "from fastapi.testclient import TestClient; from api.main import app; print(TestClient(app).get('/episodes').json())"`.

---

## Status

| Stage | Demo | Real |
|---|---|---|
| Discovery | ✅ seeded catalog | ✅ arXiv API (`NEUROPOD_LIVE_DISCOVERY=true`) |
| Citation signal | ✅ baked-in | ⏳ Semantic Scholar API |
| PDF extraction | ✅ pre-extracted sections | ⏳ PyMuPDF + arXiv PDF fetch |
| Embeddings | ✅ hash-based | ⏳ OpenAI `text-embedding-3-small` |
| Script | ✅ template | ✅ Anthropic / OpenAI |
| QA | ✅ token overlap | ⏳ second-LLM verification |
| TTS | ✅ tone | ✅ ElevenLabs / OpenAI |
| Storage | ✅ local JSON | ⏳ Supabase / Postgres |
| Auth | — | ⏳ Supabase Auth |

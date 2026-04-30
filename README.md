# Neuropod

Turns recent arXiv papers into short narrated podcast episodes. You set a few research topics, the app pulls fresh papers in your areas, ranks them, generates a 6-9 minute audio brief grounded in the paper, and hands you an RSS feed so you can listen in any podcast app.

It also has a web UI for browsing episodes, asking grounded follow-up questions about a paper, and tweaking your topics.

The app runs without any API keys (demo mode with seeded papers and a synthesized tone for audio). Add provider keys to upgrade each stage.

## Tech stack

- **Backend**: Python, FastAPI, APScheduler
- **Pipeline**: arXiv API, PyMuPDF (PDF extraction), OpenAI / Anthropic (script generation), OpenAI embeddings, ElevenLabs / OpenAI (text-to-speech)
- **Frontend**: Next.js 16 (App Router), TypeScript, plain CSS
- **Storage**: local JSON store (Postgres + pgvector schema included for production)
- **Container**: Docker / docker-compose

## Running it

You need Python 3.11+ and Node 20+.

```
git clone https://github.com/alvinn6o/Neuropod-Research-Paper-Podcasts.git
cd Neuropod-Research-Paper-Podcasts

cp .env.example .env
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

Start the backend in one terminal:

```
python -m uvicorn api.main:app --reload --port 8000
```

Start the frontend in another:

```
cd frontend && npm run dev
```

Open http://localhost:3000.

Or run everything with Docker:

```
docker-compose up
```

## API keys (optional)

The app works without any keys. To get real papers and real audio, edit `.env` and add whichever keys you have.

```
ANTHROPIC_API_KEY=         # script generation (preferred)
OPENAI_API_KEY=            # script generation (fallback) + embeddings + TTS fallback
ELEVENLABS_API_KEY=        # better-sounding narration
NEUROPOD_LIVE_DISCOVERY=true   # fetch live papers from arXiv instead of the demo catalog
```

The status badge in the top right of the UI flips from "demo" to "live" when real providers are wired. If a provider call fails (bad key, rate limit, etc.) the badge shows a warning and you can hover to see the error.

There's no signup, no Anthropic/OpenAI/ElevenLabs login flow inside the app — you paste keys directly into `.env`. Costs roughly land around $0.02 per script (LLM) plus $0.03 to $0.15 per episode for TTS depending on which provider you use.

## How it works

1. Pulls candidate papers from arXiv filtered to your topics and a window (default last 7 days).
2. Scores them by recency, citation velocity, and how well they match your topics. Past listening behavior also nudges the ranking.
3. Downloads the PDF, splits it into sections (Abstract, Methods, Results, etc.), chunks each section, and embeds the chunks.
4. Retrieves the most relevant chunks for the paper's topic, sends them to the LLM with a structured prompt, gets back a 800-1200 word script.
5. Runs a quick QA check, sends the script to TTS, caches the audio on disk.
6. Episodes show up in the feed and the RSS endpoint at `/feed/<slug>`.

You can ask follow-up questions on any episode page — the same retriever finds relevant chunks and shows them as citations under the answer.

## Layout

```
api/                FastAPI app, routes, storage, audio cache
pipeline/
  discover/         arXiv client, citation enrichment, ranker, affinity
  ingest/           PDF extraction, section-aware chunking
  generate/         Embedder, retriever, script writer, QA
  synthesize/       TTS providers, audio post-processing
frontend/
  src/app/          Next.js pages
  src/components/   UI components (player, chat, topic editor, etc.)
db/schema.sql       Postgres + pgvector schema for production
scripts/            CLI helpers (seed, run pipeline)
```

## Keyboard shortcuts

- `Space` — play / pause
- `J` or `←` — back 15 seconds
- `L` or `→` — forward 30 seconds
- `⌘K` or `/` — open command palette
- `?` — show shortcuts overlay

Click the speed indicator on the player (`1×`) to cycle through 0.85× → 1× → 1.25× → 1.5× → 1.75× → 2×.

## License

MIT.

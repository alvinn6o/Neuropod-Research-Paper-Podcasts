# Neuropod

Neuropod turns recent arXiv papers into citation-grounded, audio-ready research scripts. Users set research topics and optional arXiv categories; the system discovers papers, extracts sections from PDFs, chunks and embeds the text, retrieves relevant context, and generates a 6-9 minute narrated script. Audio is a separate optional step: after a script exists, the app can synthesize TTS audio and attach it to the episode/feed.

The project is designed as an AI engineering portfolio system, not a permanently hosted SaaS. It runs locally without API keys in demo mode, and the repo includes Docker plus an AWS reference architecture/IaC scaffold for interview discussion.

## Tech Stack

- **Backend**: Python, FastAPI, APScheduler-ready job model
- **Pipeline**: arXiv API, PyMuPDF section extraction, section-aware chunking, OpenAI embeddings or deterministic hash fallback, Anthropic/OpenAI/Bedrock-compatible script generation
- **Retrieval**: in-process dense/sparse retriever today; Postgres + pgvector HNSW schema included for production retrieval
- **Audio**: optional ElevenLabs/OpenAI TTS after script generation; demo fallback emits a short WAV tone
- **Frontend**: Next.js 16 App Router, TypeScript, plain CSS
- **Storage**: SQLite for zero-cost local runs; Postgres + pgvector schema for production
- **Container**: Docker / docker-compose

## Running It

Two ways: Docker (recommended — single command, everything wired) or local Python + Node.

### Docker (recommended)

You need Docker Desktop running.

**First time setup:**

```bash
git clone https://github.com/alvinn6o/Neuropod-Research-Paper-Podcasts.git
cd Neuropod-Research-Paper-Podcasts
cp .env.example .env
```

**Start it:**

```bash
docker-compose up --build -d
```

Open http://localhost:3000.

**Daily commands:**

```bash
# Start (no rebuild — fast)
docker-compose up -d

# Restart after pulling code changes (rebuilds API/worker images)
docker-compose down && docker-compose up --build -d

# Watch logs live
docker-compose logs -f api

# Stop everything (keeps your DB + episodes)
docker-compose down

# Nuke everything including users + episodes
docker-compose down -v
```

**Verify it's up:**

```bash
curl http://localhost:8000/healthz
```

Should print `{"status":"ok"}`.

### Local (no Docker)

You need Python 3.11+ and Node 20+ on your machine.

**First time setup:**

```bash
git clone https://github.com/alvinn6o/Neuropod-Research-Paper-Podcasts.git
cd Neuropod-Research-Paper-Podcasts

cp .env.example .env
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

**Run it (two terminals):**

```bash
# Terminal 1 — API
python -m uvicorn api.main:app --reload --port 8000
```

```bash
# Terminal 2 — frontend
cd frontend && npm run dev
```

Open http://localhost:3000. Postgres is optional locally — the app falls back to a SQLite file at `data/neuropod.sqlite3` if `NEUROPOD_DATABASE_URL` is empty.

### Useful one-liners

```bash
# Run the test suite (40 tests, ~1s, no API calls)
make test

# Type-check the frontend
make typecheck

# Remove any duplicate episodes (one-shot cleanup)
make dedupe

# Run the optional LLM-as-judge eval (costs API credits, requires OPENAI_API_KEY)
make eval
```

## API Keys

The app works without keys. Demo mode uses seeded papers, deterministic embeddings, fallback script generation, and optional demo audio.

Add provider keys to `.env` when you want real model/audio calls:

```bash
ANTHROPIC_API_KEY=              # script generation, preferred direct provider
OPENAI_API_KEY=                 # embeddings, script fallback, TTS fallback
ELEVENLABS_API_KEY=             # optional higher-quality narration
NEUROPOD_LIVE_DISCOVERY=true    # fetch live arXiv papers instead of demo catalog
```

Audio is intentionally not part of the default pipeline run:

```bash
NEUROPOD_GENERATE_AUDIO_ON_PIPELINE=false
```

Use the UI's episode page, `POST /episodes/{id}/audio`, or `python scripts/run_pipeline.py --with-audio` when you want TTS generated.

This portfolio build uses operator-level provider keys from environment variables. It does not include an in-app user BYOK key vault.

## Reference AWS Architecture

This repo is not currently deployed to AWS. The AWS path is documented and scaffolded so the architecture can be discussed and redeployed later without keeping idle resources running.

```mermaid
flowchart TB
  user(["User browser"])
  web[("Next.js frontend")]
  apigw["API Gateway or ALB"]
  api["FastAPI API<br/>(Lambda container or ECS service)"]
  cognito[("Cognito<br/>optional user auth")]
  worker["Pipeline worker<br/>(ECS Fargate task)"]
  ecr[("ECR<br/>API + worker images")]
  eb["EventBridge<br/>optional scheduled runs"]
  pg[("Postgres + pgvector<br/>papers, chunks, jobs, episodes")]
  s3["S3<br/>optional audio/PDF storage"]
  ssm["Parameter Store<br/>provider config and secrets"]
  iam["IAM roles<br/>least-privilege task access"]
  bedrock["Bedrock Claude<br/>optional script provider"]
  direct["Anthropic/OpenAI APIs<br/>direct provider path"]
  kb["Bedrock Knowledge Bases<br/>evaluated alternative RAG path"]
  tts["ElevenLabs/OpenAI TTS<br/>optional audio"]
  arxiv["arXiv API"]

  user --> web
  web --> apigw
  apigw --> api
  api <--> cognito
  api <--> pg
  api --> ssm
  api -->|enqueue script/audio jobs| worker
  eb -->|scheduled refresh| worker
  worker <-- ecr
  worker --> iam
  worker <--> pg
  worker --> arxiv
  worker --> bedrock
  worker --> direct
  worker --> tts
  worker --> s3
  kb -. managed RAG option, not default .-> bedrock
```

Why pgvector instead of keeping Bedrock Knowledge Bases live: Knowledge Bases depend on OpenSearch Serverless, which has an idle cost floor. For a portfolio system, Postgres + pgvector keeps local and production paths cheaper while still demonstrating vector retrieval design. Bedrock Knowledge Bases remain a reasonable enterprise alternative if the managed ingestion/retrieval workflow is worth the cost.

See [infra/aws/README.md](infra/aws/README.md) for the AWS service mapping and [infra/aws/terraform](infra/aws/terraform) for a non-applied Terraform scaffold.

## How It Works

1. Pulls candidate papers from arXiv filtered to user topics, arXiv categories, and a discovery window.
2. Scores papers by recency, topic affinity, citation signal, and prior listening feedback.
3. Extracts PDF sections with PyMuPDF, chunks each section, and embeds chunks.
4. Retrieves relevant chunks and sends them to the script writer with a grounded prompt.
5. Persists the generated script, paper metadata, chunks, QA status, and estimated runtime.
6. Optionally synthesizes TTS audio for an existing script and stores the audio on disk or S3.
7. RSS feed items include audio enclosures only for episodes with generated audio.

Follow-up Q&A uses the same indexed paper chunks and returns citation excerpts under each answer.

## Evaluation

The retriever has a checked-in benchmark fixture for the Mamba paper (`arXiv:2312.00752`):

```bash
pytest tests/test_recall.py -q -s
```

Current deterministic fixture results (1536-dim hash embeddings, 63 chunks across 7 sections):

- recall@1: 58.3%
- recall@5: 83.3%
- recall@10: 83.3%
- MRR: 0.692

The full deterministic test suite covers API smoke tests, auth/session behavior, topic/category CRUD, chunking invariants, retriever behavior, QA heuristics, and the script-first/audio-optional episode flow.

```bash
pytest tests/ -q
```

## Layout

```text
api/                FastAPI app, routes, DB helpers, audio storage
pipeline/
  discover/         arXiv client, ranker, affinity, citation-signal interface
  ingest/           PDF extraction, section-aware chunking
  generate/         Embedders, retriever, script writer, QA
  synthesize/       Optional TTS providers and audio helpers
frontend/
  src/app/          Next.js pages
  src/components/   UI components: player, chat, topic editor, command palette
db/schema.sql       Postgres + pgvector schema
eval/               Recall fixture generation and optional LLM-as-judge eval
infra/aws/          AWS reference architecture and Terraform scaffold
scripts/            CLI helpers
```

## Interview Positioning

The accurate one-liner:

> Neuropod is a script-first RAG system for tracking research papers: it discovers arXiv papers, extracts and chunks PDFs, benchmarks retrieval on a real paper, generates grounded narrated scripts, and can optionally synthesize podcast audio. I have not kept it deployed on AWS, but I designed the AWS path around ECS/Fargate, ECR, IAM roles, Parameter Store, S3, Bedrock Claude, and an evaluated Bedrock Knowledge Bases alternative.

## Keyboard Shortcuts

- `Space` - play / pause
- `J` or `Left Arrow` - back 15 seconds
- `L` or `Right Arrow` - forward 30 seconds
- `Cmd+K` or `/` - open command palette
- `?` - show shortcuts overlay

Click the speed indicator on the player (`1x`) to cycle through 0.85x -> 1x -> 1.25x -> 1.5x -> 1.75x -> 2x.

## License

MIT

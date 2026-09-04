# Neuropod

Neuropod turns recent arXiv papers into citation-grounded, audio-ready research scripts. Users set research topics and optional arXiv categories; the system discovers papers, extracts sections from PDFs, chunks and embeds the text, retrieves relevant context, and generates a 6-9 minute narrated script. Audio is a separate optional step: after a script exists, the app can synthesize TTS audio and attach it to the episode/feed.

The project is designed as an AI engineering portfolio system, not a permanently hosted SaaS. It runs locally without API keys in demo mode, and the repo includes Docker plus an AWS reference architecture/IaC scaffold for interview discussion.

## Tech Stack

- **Backend**: Python, FastAPI, APScheduler-ready job model
- **Pipeline**: arXiv API, PyMuPDF section extraction, section-aware chunking, tiktoken token accounting, OpenAI embeddings or deterministic hash fallback, Anthropic/OpenAI/Bedrock-compatible script generation
- **Retrieval**: in-process dense/sparse retriever today; Postgres + pgvector HNSW schema included for production retrieval
- **Audio**: optional ElevenLabs/OpenAI TTS after script generation; demo fallback emits a short WAV tone
- **Frontend**: Next.js 16 App Router, TypeScript, plain CSS
- **Storage**: SQLite for zero-cost local runs; Postgres + pgvector schema for production
- **Telemetry**: per-call `llm_calls` ledger (exact tokens + cost) and `retrieval_traces` (which chunks grounded each script)
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
# Run the test suite (~1s, no API calls)
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

Retrieval is measured on a **frozen 42-paper corpus** with **472 queries**, and
every number carries a confidence interval. Changes are compared with a *paired*
bootstrap on the same queries, not by eyeballing whether two intervals overlap.

```bash
python -m eval.harness           # the ablation table below
pytest tests/test_eval_harness.py -q   # the regression gate, ~3s
```

### Ablation

nDCG@10, **1,935 ICT queries over 168 papers**, hash embeddings, retrieval
scoped to one paper (matching production).

| Config | nDCG@10 | 95% CI | vs. baseline (paired) | |
|---|---|---|---|---|
| `current` (shipping) | 0.281 | [0.268, 0.294] | — | baseline |
| `dense+prior` | 0.284 | [0.271, 0.297] | +0.003 [−0.006, +0.012] p=0.464 | not significant |
| `dense` | 0.318 | [0.303, 0.332] | +0.037 [+0.026, +0.049] p<0.001 | significant |
| `rrf+prior` | 0.345 | [0.332, 0.359] | +0.065 [+0.055, +0.076] p<0.001 | significant |
| `rrf` | 0.356 | [0.343, 0.370] | +0.075 [+0.063, +0.087] p<0.001 | significant |
| **`bm25`** | **0.359** | [0.346, 0.373] | **+0.078 [+0.064, +0.092] p<0.001** | **significant** |

### A leak, found and fixed

The first version of this table was wrong, and how it was wrong is the most
instructive part of the project.

ICT redacts the query sentence from its gold chunk. The chunker caps chunks at
110 words — so after redacting *only* the gold, **88.9% of non-gold chunks sat
exactly at the cap and 0% of gold chunks did**. The gold chunk was identifiable
without reading the query.

| Ranker | nDCG@10 |
|---|---|
| by chunk length alone, query ignored | **0.369** |
| BM25 on the same fold | 0.224 |
| random ordering | 0.075 |

A query-independent rule beat BM25. A gradient-boosted reranker with a length
feature reached **0.667** — a 3× headline that was almost entirely the artifact.

The tell was the fitted coefficient: `chunk_tokens` came out at **−2.07**,
several times any relevance feature. A model insisting shorter chunks are more
relevant is not describing retrieval.

`queries.redact_pool` now removes a sentence from **every** candidate. Two tests
in `tests/test_eval_harness.py` fail if either property regresses. Every number
below is post-fix.

### Findings — including a negative one

**1. The hand-tuned section prior makes retrieval worse.**
`Retriever.section_bonus` adds a hand-set constant (`abstract: 0.18`,
`results: 0.16`, …) directly onto a cosine score. Tested with paired bootstraps
rather than inferred from the table:

| Comparison | Δ nDCG@10 | 95% CI | p | |
|---|---|---|---|---|
| `dense` → `dense+prior` | **−0.034** | [−0.042, −0.025] | <0.001 | significantly worse |
| `rrf` → `rrf+prior` | **−0.011** | [−0.016, −0.005] | <0.001 | significantly worse |

At the earlier corpus size (n=472) the second row was −0.009 with p=0.078 —
*not* significant. Quadrupling the corpus resolved a real effect that the
smaller sample could not. That is the argument for corpus size, made concretely.

**2. Proper BM25 beats the shipping sparse path by 29% relative.** The shipping
"sparse fallback" is raw term-frequency cosine — no IDF, no length
normalization, no stopwords.

**3. RRF fusion does *not* beat BM25 alone here.** Δ = −0.003, CI
[−0.012, +0.006], p=0.480, even at n=1935. Hybrid retrieval is usually the right
default and it did not win on this query set.

### A trained reranker

Grouped by **paper**, never by query — queries from one paper share a chunk
pool, so a query-level split lets a model memorize chunks it is then scored on.
5-fold GroupKFold for model selection, plus 34 papers held out and scored once.

Cross-validation (134 papers, 5 folds):

| Model | mean nDCG@10 | std | per-fold |
|---|---|---|---|
| **LambdaMART** | **0.502** | 0.010 | 0.497 0.510 0.517 0.495 0.489 |
| GBDT (sklearn) | 0.498 | 0.028 | 0.503 0.515 0.524 0.445 0.506 |
| Logistic regression | 0.376 | 0.009 | 0.381 0.388 0.379 0.362 0.372 |
| BM25 | 0.367 | 0.012 | 0.374 0.377 0.377 0.350 0.355 |
| `dense+prior` (shipping) | 0.296 | 0.014 | 0.296 0.321 0.283 0.293 0.284 |

Held out (34 papers, 382 queries, scored once):

| Model | nDCG@10 | 95% CI | vs. BM25 (paired) | |
|---|---|---|---|---|
| **LambdaMART** | **0.484** | [0.449, 0.520] | **+0.160 [+0.134, +0.185] p<0.001** | **significant** |
| GBDT | 0.423 | [0.388, 0.460] | +0.099 [+0.079, +0.122] p<0.001 | significant |
| BM25 | 0.324 | [0.293, 0.355] | — | baseline |
| Logistic regression | 0.320 | [0.290, 0.352] | −0.003 [−0.018, +0.011] p=0.598 | not significant |
| `dense+prior` (shipping) | 0.237 | [0.210, 0.263] | −0.086 [−0.117, −0.057] p<0.001 | significant |

**The ranking objective is worth +0.061 on held-out papers.** LambdaMART
and the sklearn GBDT are both gradient-boosted trees on identical features; the
only difference is that LambdaMART optimizes NDCG over query groups while the
GBDT classifies each candidate independently and hopes the induced order is
good. Knowing that candidates *compete within a query* is the single largest
modelling gain here.

**A linear model is not enough.** Logistic regression ties BM25
(−0.003, p=0.598). The gain is not "any learned model beats a heuristic" — it
needs the capacity for feature interactions.

**What nDCG@10 = 0.48 actually means here.** With exactly one relevant chunk
per query, nDCG@10 reduces to `1/log2(rank+1)` — so the headline number is a
restatement of where the gold chunk lands. Reported as ranks instead:

| k | LambdaMART hit@k | BM25 hit@k |
|---|---|---|
| 1 | 24.1% | 8.4% |
| 3 | 48.4% | 33.0% |
| 5 | 61.8% | 46.1% |
| 10 | 75.9% | 60.7% |
| **14** | **84.0%** | 69.1% |
| 20 | 89.8% | 81.7% |

Median gold rank is **4** (BM25: 6), out of a mean pool of 45 chunks.

**hit@14 is the number that matters for this product**, not nDCG@10: the
scriptwriter puts the top 14 chunks into the prompt, so 84.0% is the rate at
which the relevant passage actually reaches the model — up from 69.1%. nDCG@10
is the right metric for *comparing rankers*; hit@14 is the right one for
deciding whether the pipeline works.

**What the model learned about sections** — the interpretable model's fitted
coefficients against the hand-set values they replace:

| Section | hand-set | learned | |
|---|---|---|---|
| `body` | 0.000 | **+0.166** | unlisted, most useful of all |
| `abstract` | +0.180 | +0.130 | agrees |
| `results` | +0.160 | −0.004 | 2nd-highest prior, learned ≈ 0 |
| `discussion` | +0.100 | **−0.082** | sign flipped |
| `methods` | +0.080 | −0.022 | sign flipped |
| `introduction` | +0.060 | −0.000 | sign flipped |
| `experiments` | 0.000 | −0.080 | unlisted, actively bad |

Only `abstract` broadly survives. `results` — the second-highest hand-set weight
— is learned as ≈0, and `body`, which the prior does not list at all, gets the
largest positive weight. That is the concrete version of "these weights were
never fit to anything."

**Negative sampling mattered more than model choice.** Training on the top 20
BM25 hard negatives scored 0.264 on dev; the full candidate pool scored 0.443.
Subsampling trains on a distribution the model is never served — at inference it
ranks all candidates including easy negatives it never saw. Classic train/serve
skew, and it cost more than the gap between logistic regression and GBDT.

```bash
python -m eval.train_reranker                  # needs libomp for LambdaMART
python -m eval.train_reranker --no-lambdamart  # sklearn models only
```

LightGBM links a system OpenMP runtime — `brew install libomp` on macOS,
bundled in Linux wheels. `train_reranker` degrades to the sklearn models with an
actionable message rather than crashing.

### What these numbers do not show

**The queries are ICT, not natural questions.** Each query is a sentence
extracted from a chunk, with that sentence redacted from the chunk before
retrieval (Lee et al. 2019). That makes the label set free, deterministic and
large — the properties a CI gate needs — but the queries share vocabulary with
their gold chunk far more than a real question would, which **systematically
favours lexical matching**. BM25's win is therefore an upper bound on its real
advantage, and finding 3 should be read as "RRF did not win *on ICT*", not "RRF
does not help".

Finding 1 is not exposed to that bias: the prior is applied on top of both
lexical and dense scoring, and hurts both.

`eval/queries.py` also implements doc2query-style natural-question generation;
it runs when `OPENAI_API_KEY` is set and caches by content hash. Those results
are not in this table because they have not been run.

**Embeddings are the hash fallback,** not OpenAI. See the note below.

**The reranker's CV→holdout gap is 0.502 → 0.484.** Real, but far smaller
than the margin over BM25 (+0.160), and much better than the 0.443 → 0.298 gap
at the previous corpus size. Fold-to-fold std is 0.010.

**The reranker is not wired into the serving path yet.** These are offline
numbers on a frozen corpus.

### The regression gate

`eval/baselines.json` pins the measured values on the 168-paper corpus;
`tests/test_eval_harness.py` fails the build if nDCG@10 drops more than 0.005
below them. The tolerance is
0.005 rather than the CI half-width (~0.028) because the point estimate is
fully deterministic — verified, not assumed — so run-to-run variance is zero
and the gate only needs to absorb rounding.

The gate was checked by breaking retrieval on purpose: disabling IDF in BM25
costs 0.013 nDCG@10, which a 0.02 tolerance let through and 0.005 catches.

### Legacy single-paper fixture

The original Mamba fixture (`arXiv:2312.00752`) is kept for continuity:

| Metric | Value | 95% CI (Wilson) |
|---|---|---|
| hit@1 | 58.3% (7/12) | 32.0% – 80.7% |
| hit@5 | 83.3% (10/12) | 55.2% – 95.3% |
| MRR | 0.692 | — |

Two caveats, both of which motivated the corpus above:

1. **It measures the `HashEmbedder` path, not the shipping path.** The fixture
   was built with the deterministic SHA256 bag-of-words fallback
   (`embedder_backend` in `tests/fixtures/mamba_meta.json`). Regenerate with
   `OPENAI_API_KEY=... python -m eval.precompute_fixtures` to measure the path
   that ships. The 42-paper corpus has the same limitation today.
2. **n = 12 on 1 paper cannot detect a change.** The interval on hit@1 spans
   32%–81%; a 10-point improvement is indistinguishable from noise. At n=472 the
   interval is ~5 points wide, which is what makes a gate possible.

Note the metric name: `test_recall.py` calls it "recall" but computes hit@k
(*any* gold chunk in the top k). Both are implemented separately in
`eval/metrics.py`.

### Corpus

168 papers, stratified across cs.LG / cs.CL / cs.CV / stat.ML and 2020–2025,
pinned by arXiv id *and version* in `eval/corpus/papers.txt`, with each PDF's
sha256 in `manifest.json`. PDFs are not committed (200MB) — they are re-fetchable
and hash-verified. Derived artifacts are committed so CI never hits the network.

Extraction quality is recorded per paper, and it is not perfect — deliberately:

- **3.6%** (6/168) fail extraction entirely and fall back to abstract-only
- **9.5%** (16/168) hit the `body` fallback, where no section header was recognised

Those failure modes are *in* the corpus rather than filtered out of it, so the
benchmark measures the pipeline that exists.

```bash
python -m eval.corpus_build select   # re-pin the corpus (deliberate, rare)
python -m eval.corpus_build build    # fetch + extract + chunk
python -m eval.queries --mode ict    # regenerate queries
```

## Cost Controls

This is a demo, and generation costs real money, so spend is bounded in four
independent places rather than one:

| Control | Default | Env var |
|---|---|---|
| Episodes per click | 5 | — (`MAX_EPISODES_PER_RUN`) |
| Per-user runs/day | 20 | `NEUROPOD_DAILY_PIPELINE_LIMIT` |
| **Global runs/day (all users)** | 60 | `NEUROPOD_GLOBAL_DAILY_RUN_LIMIT` |
| **Monthly USD ceiling** | $5.00 | `NEUROPOD_MONTHLY_BUDGET_USD` |
| **Daily USD ceiling** | $1.00 | `NEUROPOD_DAILY_BUDGET_USD` |

The global and USD caps exist because the per-user caps are not a spend bound on
their own: `POST /auth/stub/login` mints a persistent identity for any email
with no verification, so a per-user quota is free to reset by making a new
account. At Sonnet prices one identity running its cap is roughly $5/day.

Spend is measured, not estimated. Every provider call writes a row to
`llm_calls` with exact token counts from the response's `usage` block and a cost
derived from the pricing table in `pipeline/usage.py`. Current spend against the
caps is visible at `GET /status` under `budget`, and a per-model rollup for the
month is under `spend` when authenticated.

**Exceeding a USD cap degrades, it does not error.** Script generation falls
back to the zero-cost template and `/ask` falls back to deterministic metadata
answers, so the site still works — it just stops spending. Affected episodes are
recorded with `llm_provider = "demo-budget"` so the degradation is visible in
the data rather than silent. The global run cap is the one control that
rejects (429), because a run that cannot call a model would just fill a feed
with template scripts.

These are defence in depth, not the last line. **Set a spend cap at the provider
too** (Anthropic workspace limits, OpenAI project budgets) — that is the only
ceiling a bug in this repo cannot bypass.

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

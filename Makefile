.PHONY: help install dev worker test test-pg typecheck build clean reset eval eval-judge corpus reranker recommend frontier tune annotate dedupe deps-audit

help:
	@echo "Targets:"
	@echo "  install     install backend + frontend deps"
	@echo "  dev         run API on :8000 (uvicorn --reload)"
	@echo "  worker      run pipeline worker (needs USER_ID=...)"
	@echo "  test        run pytest suite (deterministic, no API calls)"
	@echo "  test-pg     run the same suite against Postgres+pgvector (needs docker)"
	@echo "  deps-audit  fail if any pinned dependency is <7 days old"
	@echo "  typecheck   run frontend tsc --noEmit"
	@echo "  build       build both Docker images"
	@echo "  reset       wipe local SQLite + audio cache"
	@echo "  eval        retrieval ablation over the frozen corpus (free, deterministic)"
	@echo "  eval-judge  LLM-as-judge eval (costs API credits, requires OPENAI_API_KEY)"
	@echo "  corpus      rebuild the frozen eval corpus (fetches PDFs from arXiv)"
	@echo "  reranker    train + evaluate the learned reranker on held-out papers"
	@echo "  recommend   Task A: paper-recommendation baselines vs labels"
	@echo "  frontier    cross-encoder quality/latency frontier (downloads a model)"
	@echo "  tune        nested CV hyperparameter sweep (~6 min)"
	@echo "  annotate    human spot-check of the LLM relevance labels"
	@echo "  dedupe      remove duplicate episodes per (user_id, paper_id)"

install:
	pip install -r requirements-dev.txt
	cd frontend && npm ci --ignore-scripts

dev:
	uvicorn api.main:app --reload --port 8000

worker:
	@if [ -z "$(USER_ID)" ]; then echo "USER_ID=... required"; exit 1; fi
	python -m pipeline.worker --user-id $(USER_ID)

test:
	pytest tests/ -v

# The SQLite shim maps vector(1536) to TEXT, so it accepts embeddings that real
# pgvector rejects. This runs the same suite against the real thing.
test-pg:
	docker run -d --rm --name neuropod-test-pg \
	  -e POSTGRES_DB=neuropod -e POSTGRES_USER=neuropod -e POSTGRES_PASSWORD=neuropod \
	  -p 5433:5432 pgvector/pgvector:pg16
	@echo "waiting for postgres..."
	@until docker exec neuropod-test-pg pg_isready -U neuropod >/dev/null 2>&1; do sleep 1; done
	-NEUROPOD_DATABASE_URL=postgres://neuropod:neuropod@localhost:5433/neuropod pytest tests/ -v
	docker stop neuropod-test-pg

# Enforces the 7-day minimum release age on every pinned version.
deps-audit:
	python scripts/audit_deps.py

typecheck:
	cd frontend && npx tsc --noEmit

build:
	docker build -t neuropod-api .
	docker build -f Dockerfile.worker -t neuropod-worker .

reset:
	rm -f data/neuropod.sqlite3 data/neuropod.sqlite3-shm data/neuropod.sqlite3-wal
	rm -f data/audio_cache/*.bin data/audio_cache/*.meta

eval:
	python -m eval.harness

eval-judge:
	python -m eval.ragas_eval

corpus:
	python -m eval.corpus_build build
	python -m eval.queries --mode ict

reranker:
	python -m eval.train_reranker

# Task A: which papers should become episodes (vs Task B: which chunks reach
# the prompt). Different unit, different labels, different baseline.
recommend:
	python -m eval.recommend

# Quality/latency frontier: BM25 vs LambdaMART vs a pretrained cross-encoder.
# Downloads ~90MB from HuggingFace on first run.
frontier:
	python -m eval.cross_encoder --papers $(or $(PAPERS),30)

# Nested CV: inner folds select hyperparameters, outer folds report. The
# non-nested number is printed alongside so the selection bias is visible.
tune:
	python -m eval.tune --configs $(or $(CONFIGS),12)

# Human spot-check of the LLM labels. Reports Cohen's kappa afterwards; below
# ~0.6 the label definition is the problem, not the model.
annotate:
	@if [ -z "$(TOPIC)" ]; then echo "TOPIC=llm|vision|rl|graph|theory required"; exit 1; fi
	python -m eval.annotate review --topic $(TOPIC) --n $(or $(N),20) --chars $(or $(CHARS),1100) $(if $(REDO),--redo,)
	python -m eval.annotate agreement

dedupe:
	python scripts/dedupe_episodes.py

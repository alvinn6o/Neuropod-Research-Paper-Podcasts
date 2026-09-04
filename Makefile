.PHONY: help install dev worker test test-pg typecheck build clean reset eval dedupe deps-audit

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
	@echo "  eval        run LLM-as-judge eval (costs API credits, requires OPENAI_API_KEY)"
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
	python -m eval.ragas_eval

dedupe:
	python scripts/dedupe_episodes.py

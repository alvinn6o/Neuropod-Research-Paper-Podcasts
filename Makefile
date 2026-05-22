.PHONY: help install dev worker test typecheck build clean reset eval dedupe

help:
	@echo "Targets:"
	@echo "  install     install backend + frontend deps"
	@echo "  dev         run API on :8000 (uvicorn --reload)"
	@echo "  worker      run pipeline worker (needs USER_ID=...)"
	@echo "  test        run pytest suite (deterministic, no API calls)"
	@echo "  typecheck   run frontend tsc --noEmit"
	@echo "  build       build both Docker images"
	@echo "  reset       wipe local SQLite + audio cache"
	@echo "  eval        run LLM-as-judge eval (costs API credits, requires OPENAI_API_KEY)"
	@echo "  dedupe      remove duplicate episodes per (user_id, paper_id)"

install:
	pip install -r requirements.txt
	cd frontend && npm install

dev:
	uvicorn api.main:app --reload --port 8000

worker:
	@if [ -z "$(USER_ID)" ]; then echo "USER_ID=... required"; exit 1; fi
	python -m pipeline.worker --user-id $(USER_ID)

test:
	pytest tests/ -v

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

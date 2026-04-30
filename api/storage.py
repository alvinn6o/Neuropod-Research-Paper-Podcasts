from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.orchestrator import build_demo_payload


class DemoStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def ensure_seeded(self, topics: list[str], episode_count: int, window_days: int = 7) -> None:
        with self._lock:
            payload = self._load_unlocked()
            if payload["episodes"]:
                return

            seeded = build_demo_payload(topics=topics, num_episodes=episode_count, window_days=window_days)
            payload["topics"] = topics
            payload["papers"] = seeded["papers"]
            payload["chunks"] = seeded["chunks"]
            payload["episodes"] = seeded["episodes"]
            payload["meta"] = {
                "seeded_at": datetime.now(timezone.utc).isoformat(),
                "last_pipeline_run": seeded["generated_at"],
            }
            self._save_unlocked(payload)

    def reset(self, topics: list[str], episode_count: int, window_days: int = 7) -> dict[str, Any]:
        with self._lock:
            seeded = build_demo_payload(topics=topics, num_episodes=episode_count, window_days=window_days)
            payload = self._blank_payload()
            payload["topics"] = topics
            payload["papers"] = seeded["papers"]
            payload["chunks"] = seeded["chunks"]
            payload["episodes"] = seeded["episodes"]
            payload["meta"] = {
                "seeded_at": datetime.now(timezone.utc).isoformat(),
                "last_pipeline_run": seeded["generated_at"],
            }
            self._save_unlocked(payload)
            return deepcopy(payload)

    def run_pipeline(self, topics: list[str], episode_count: int, window_days: int = 7) -> dict[str, Any]:
        with self._lock:
            payload = self._load_unlocked()
            generated = build_demo_payload(
                topics=topics,
                num_episodes=episode_count,
                window_days=window_days,
                feedback_events=payload.get("feedback", []),
                prior_episodes=payload.get("episodes", []),
            )

            paper_by_arxiv = {paper["arxiv_id"]: paper for paper in payload["papers"]}
            chunk_by_id = {chunk["id"]: chunk for chunk in payload["chunks"]}
            episode_by_id = {episode["id"]: episode for episode in payload["episodes"]}

            for paper in generated["papers"]:
                paper_by_arxiv[paper["arxiv_id"]] = paper

            for chunk in generated["chunks"]:
                chunk_by_id[chunk["id"]] = chunk

            for episode in generated["episodes"]:
                episode_by_id[episode["id"]] = episode

            payload["topics"] = topics
            payload["papers"] = sorted(
                paper_by_arxiv.values(),
                key=lambda item: item["published_at"],
                reverse=True,
            )
            payload["chunks"] = list(chunk_by_id.values())
            payload["episodes"] = sorted(
                episode_by_id.values(),
                key=lambda item: item["created_at"],
                reverse=True,
            )
            payload["meta"]["last_pipeline_run"] = generated["generated_at"]
            self._save_unlocked(payload)
            return deepcopy(generated)

    def get_topics(self) -> list[str]:
        with self._lock:
            return list(self._load_unlocked()["topics"])

    def set_topics(self, topics: list[str]) -> list[str]:
        cleaned = [topic.strip() for topic in topics if topic.strip()]
        with self._lock:
            payload = self._load_unlocked()
            payload["topics"] = cleaned
            payload["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_unlocked(payload)
            return list(cleaned)

    def list_episodes(self, topic: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._load_unlocked()
            episodes = payload["episodes"]
            papers = {paper["id"]: paper for paper in payload["papers"]}

        items: list[dict[str, Any]] = []
        for episode in episodes:
            paper = papers.get(episode["paper_id"])
            if not paper:
                continue
            if topic and topic.lower() not in episode["topic"].lower() and not any(
                topic.lower() in category.lower() for category in paper["categories"]
            ):
                continue
            items.append(self._attach_paper(episode, paper))

        return items[:limit]

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._load_unlocked()
            episode = next((item for item in payload["episodes"] if item["id"] == episode_id), None)
            if not episode:
                return None
            paper = next((item for item in payload["papers"] if item["id"] == episode["paper_id"]), None)
            if not paper:
                return None
            return self._attach_paper(episode, paper)

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._load_unlocked()
            return next((item for item in payload["papers"] if item["id"] == paper_id), None)

    def get_chunks_for_paper(self, paper_id: str) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._load_unlocked()
            return [chunk for chunk in payload["chunks"] if chunk["paper_id"] == paper_id]

    def add_feedback(self, event: dict[str, Any]) -> int:
        with self._lock:
            payload = self._load_unlocked()
            payload["feedback"].append(
                {
                    "id": event["id"],
                    "episode_id": event["episode_id"],
                    "event_type": event["event_type"],
                    "position_secs": event["position_secs"],
                    "created_at": event["created_at"],
                }
            )
            self._save_unlocked(payload)
            return len(payload["feedback"])

    def get_meta(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._load_unlocked()["meta"])

    def _attach_paper(self, episode: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
        item = deepcopy(episode)
        item["paper"] = deepcopy(paper)
        return item

    def _blank_payload(self) -> dict[str, Any]:
        return {
            "topics": [],
            "papers": [],
            "chunks": [],
            "episodes": [],
            "feedback": [],
            "meta": {},
        }

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._blank_payload()

        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

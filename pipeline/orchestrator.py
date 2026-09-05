from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .discover.arxiv_client import ArxivClient
from .discover.ranker import rank_candidates
from .discover.semantic_scholar import SemanticScholarClient
from .generate.embedder import EmbeddingError, get_embedder
from .generate.facets import build_facet_queries, facet_names
from .generate.qa_check import QAChecker
from .generate.retriever import PROMPT_CHUNK_LIMIT, RETRIEVER_VERSION, Retriever
from .generate.scriptwriter import ScriptWriter
from .ingest.chunker import SectionAwareChunker
from .ingest.pdf_extractor import PDFExtractor
from .models import EpisodeDraft
from .synthesize.audio_processor import AudioProcessor

logger = logging.getLogger("neuropod.orchestrator")


def build_demo_payload(
    topics: list[str],
    num_episodes: int,
    window_days: int = 7,
    feedback_events: Optional[list[dict]] = None,
    prior_episodes: Optional[list[dict]] = None,
    categories: Optional[list[str]] = None,
) -> dict:
    """Run discovery, extraction, retrieval, script generation, and QA.

    Audio synthesis is deliberately handled as a separate optional step so the
    core research artifact remains a grounded script even when no TTS provider
    is configured.
    """
    from .discover.affinity import compute_affinity

    discovery = ArxivClient()
    metadata = SemanticScholarClient()
    extractor = PDFExtractor()
    chunker = SectionAwareChunker()
    embedder = get_embedder()
    retriever = Retriever(embedder=embedder)
    writer = ScriptWriter()
    checker = QAChecker()
    audio = AudioProcessor()

    affinity_scores = compute_affinity(feedback_events or [], prior_episodes or [])

    candidates = discovery.search(
        topics=topics,
        days=window_days,
        max_results=max(6, num_episodes * 2),
        categories=categories or [],
    )
    candidates = metadata.enrich(candidates)
    selected = rank_candidates(
        candidates,
        topics,
        top_k=num_episodes,
        window_days=window_days,
        affinity_scores=affinity_scores,
    )

    papers: list[dict] = []
    chunks: list[dict] = []
    episodes: list[dict] = []

    generated_at = datetime.now(timezone.utc).isoformat()

    skipped: list[dict] = []

    for candidate in selected:
        paper_id = str(uuid4())
        paper_record = {
            "id": paper_id,
            "arxiv_id": candidate.arxiv_id,
            "title": candidate.title,
            "authors": candidate.authors,
            "abstract": candidate.abstract,
            "categories": candidate.categories,
            "published_at": candidate.published_at,
            "pdf_url": candidate.pdf_url,
            "citation_count": candidate.citation_count,
            "score": round(candidate.score, 4),
        }

        section_map = extractor.extract_sections(candidate)
        chunk_models = chunker.chunk_sections(paper_id, section_map)
        try:
            chunk_models = embedder.embed_chunks(chunk_models)
        except EmbeddingError as exc:
            # Skip the paper rather than persist a half-embedded chunk set.
            # A partially embedded paper is worse than a missing one: retrieval
            # over a mixed index returns confident nonsense.
            logger.warning("skipping %s: %s", candidate.arxiv_id, exc)
            skipped.append({"arxiv_id": candidate.arxiv_id, "reason": str(exc)})
            continue

        # Appended only after embedding succeeds, so a skipped paper leaves no
        # partial row behind.
        papers.append(paper_record)
        chunk_dicts = [chunk.to_dict() for chunk in chunk_models]
        chunks.extend(chunk_dicts)

        # One query per thing the prompt asks the model to cover, fused with
        # RRF. Replaces `title + abstract`, which asked "find chunks like the
        # summary of the whole paper" — diffuse by construction, and partly
        # retrieving its own source since the abstract is in the index.
        #
        # Measured on 168 papers (eval/coverage.py): section coverage@14 goes
        # 0.901 -> 0.946, CI [+0.021, +0.069], p<0.001. The gain is concentrated
        # where it matters most — the results section reaches the prompt 94.2%
        # of the time, up from 74.2%. A script cannot report quantitative
        # results that were never retrieved.
        facet_queries = build_facet_queries(candidate.title, candidate.abstract)
        scored = retriever.retrieve_multi(chunk_dicts, facet_queries, limit=PROMPT_CHUNK_LIMIT)
        retrieved = [row["chunk"] for row in scored]
        script, llm_label = writer.write(candidate, retrieved, topics)
        qa_status, qa_notes = checker.verify(script, chunk_dicts)

        retrieval_trace = [
            {
                "chunk_id": row["chunk"]["id"],
                "query_text": " | ".join(facet_names())[:2000],
                "retriever_version": RETRIEVER_VERSION,
                "rank": row["rank"],
                "dense_score": row["dense_score"],
                "sparse_score": row["sparse_score"],
                "section_bonus": row["section_bonus"],
                "final_score": row["final_score"],
                "used_in_prompt": row["rank"] < PROMPT_CHUNK_LIMIT,
            }
            for row in scored
        ]
        topic = _derive_topic(candidate.categories, topics)
        episode = EpisodeDraft(
            id=str(uuid4()),
            paper_id=paper_id,
            title=candidate.title,
            description=audio.build_description(script),
            topic=topic,
            score=round(candidate.score, 4),
            script=script,
            qa_status=qa_status,
            qa_notes=qa_notes,
            duration_secs=audio.estimate_duration_secs(script),
            tts_provider="none",
            created_at=generated_at,
        )
        episode_dict = episode.to_dict()
        episode_dict["llm_provider"] = llm_label
        # Carried on the payload so the runner can persist it once the episode
        # has a real database id.
        episode_dict["retrieval_trace"] = retrieval_trace
        episodes.append(episode_dict)

    return {
        "generated_at": generated_at,
        "papers": papers,
        "chunks": chunks,
        "episodes": episodes,
        "skipped": skipped,
    }


def _derive_topic(categories: list[str], topics: list[str]) -> str:
    if topics:
        return topics[0]
    if categories:
        return categories[0]
    return "general AI"

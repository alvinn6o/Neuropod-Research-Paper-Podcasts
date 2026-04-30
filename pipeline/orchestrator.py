from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .discover.arxiv_client import ArxivClient
from .discover.ranker import rank_candidates
from .discover.semantic_scholar import SemanticScholarClient
from .generate.embedder import get_embedder
from .generate.qa_check import QAChecker
from .generate.retriever import Retriever
from .generate.scriptwriter import ScriptWriter
from .ingest.chunker import SectionAwareChunker
from .ingest.pdf_extractor import PDFExtractor
from .models import EpisodeDraft
from .synthesize.audio_processor import AudioProcessor
from .synthesize.tts import TTSProvider


def build_demo_payload(
    topics: list[str],
    num_episodes: int,
    window_days: int = 7,
    feedback_events: list[dict] | None = None,
    prior_episodes: list[dict] | None = None,
) -> dict:
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
    tts = TTSProvider()

    affinity_scores = compute_affinity(feedback_events or [], prior_episodes or [])

    candidates = discovery.search(topics=topics, days=window_days, max_results=max(6, num_episodes * 2))
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
        papers.append(paper_record)

        section_map = extractor.extract_sections(candidate)
        chunk_models = chunker.chunk_sections(paper_id, section_map)
        chunk_models = embedder.embed_chunks(chunk_models)
        chunk_dicts = [chunk.to_dict() for chunk in chunk_models]
        chunks.extend(chunk_dicts)

        query = f"{candidate.title} {candidate.abstract}"
        retrieved = retriever.retrieve(chunk_dicts, query, limit=10)
        script, llm_label = writer.write(candidate, retrieved, topics)
        qa_status, qa_notes = checker.verify(script, chunk_dicts)
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
            tts_provider=tts.provider_name,
            created_at=generated_at,
        )
        episodes.append(episode.to_dict())

    return {
        "generated_at": generated_at,
        "papers": papers,
        "chunks": chunks,
        "episodes": episodes,
    }


def _derive_topic(categories: list[str], topics: list[str]) -> str:
    if topics:
        return topics[0]
    if categories:
        return categories[0]
    return "general AI"

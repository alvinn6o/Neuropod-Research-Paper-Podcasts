from __future__ import annotations

from uuid import uuid4

from ..models import PaperChunk
from .tokenizer import backend as token_backend
from .tokenizer import count_tokens


class SectionAwareChunker:
    def __init__(self, max_words: int = 110, overlap_words: int = 24) -> None:
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk_sections(self, paper_id: str, sections: dict[str, str]) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        for section, text in sections.items():
            words = text.split()
            if len(words) <= self.max_words:
                chunks.append(self._build_chunk(paper_id, section, 0, " ".join(words)))
                continue

            start = 0
            index = 0
            while start < len(words):
                end = min(start + self.max_words, len(words))
                window = words[start:end]
                chunks.append(self._build_chunk(paper_id, section, index, " ".join(window)))
                if end >= len(words):
                    break
                start = max(end - self.overlap_words, start + 1)
                index += 1

        return chunks

    def _build_chunk(self, paper_id: str, section: str, index: int, text: str) -> PaperChunk:
        content = text.strip()
        return PaperChunk(
            id=str(uuid4()),
            paper_id=paper_id,
            section=section,
            chunk_index=index,
            content=content,
            # Real token count, not len(text.split()). The window below is still
            # measured in words: switching the window to tokens changes chunk
            # boundaries, and therefore retrieval, and there is no evaluation
            # harness yet to measure whether that helps. That is a Phase 2 change
            # to be made against a benchmark, not a drive-by here.
            token_count=count_tokens(content),
            token_source=token_backend(),
        )

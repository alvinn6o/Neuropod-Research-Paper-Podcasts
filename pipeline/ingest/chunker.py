from __future__ import annotations

from uuid import uuid4

from ..models import PaperChunk


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
        return PaperChunk(
            id=str(uuid4()),
            paper_id=paper_id,
            section=section,
            chunk_index=index,
            content=text.strip(),
            token_count=len(text.split()),
        )

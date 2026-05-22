from __future__ import annotations

import logging
import re
import urllib.request
from typing import Optional

from ..models import PaperCandidate

logger = logging.getLogger("neuropod.pdf")

# Header patterns. Most ML papers use numbered sections like "3. Methodology" or
# unnumbered "Methods". Order matters — earlier matches win when multiple regexes
# could match (e.g. "Experimental Setup" matches both 'experiments' and 'methods').
_NUM = r"(\d+(?:\.\d+)*\.?\s*)?"  # optional leading "3.", "3.1", "3.1.2."
SECTION_HEADERS = [
    ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE)),
    ("introduction", re.compile(rf"^\s*{_NUM}introduction\s*$", re.IGNORECASE)),
    ("background", re.compile(rf"^\s*{_NUM}(background|related work|prior work|preliminaries)\s*$", re.IGNORECASE)),
    ("methods", re.compile(
        rf"^\s*{_NUM}(method|methods|methodology|approach|approaches|model|"
        r"architecture|framework|algorithm|design|proposed (method|approach|model)|"
        r"implementation(\s+details)?)\s*$", re.IGNORECASE)),
    ("experiments", re.compile(
        rf"^\s*{_NUM}(experiments?|experimental (setup|methodology)|"
        r"setup|evaluation(\s+methodology)?|empirical evaluation|"
        r"protocol|ablation(s|\s+study|\s+studies)?)\s*$", re.IGNORECASE)),
    ("results", re.compile(
        rf"^\s*{_NUM}(results?|findings?|main results?|empirical results?|"
        r"observations?|performance)\s*$", re.IGNORECASE)),
    ("discussion", re.compile(rf"^\s*{_NUM}(discussion|analysis|takeaways?)\s*$", re.IGNORECASE)),
    ("limitations", re.compile(rf"^\s*{_NUM}(limitations?|threats to validity)\s*$", re.IGNORECASE)),
    ("conclusion", re.compile(
        rf"^\s*{_NUM}(conclusion|conclusions?|summary|"
        r"conclusion(\s+and\s+future\s+work)?|"
        r"concluding remarks)\s*$", re.IGNORECASE)),
]
STOP_HEADERS = [
    re.compile(r"^\s*references\s*$", re.IGNORECASE),
    re.compile(r"^\s*acknowledgements?\s*$", re.IGNORECASE),
    re.compile(r"^\s*bibliography\s*$", re.IGNORECASE),
    re.compile(r"^\s*appendix\s*[A-Z]?\s*$", re.IGNORECASE),
]


class PDFExtractor:
    """Live PDF section extractor using PyMuPDF, with seed-catalog fallback."""

    def __init__(self, max_pdf_bytes: int = 12_000_000, fetch_timeout: int = 25) -> None:
        self.max_pdf_bytes = max_pdf_bytes
        self.fetch_timeout = fetch_timeout

    def extract_sections(self, candidate: PaperCandidate) -> dict[str, str]:
        # Seed-catalog fast path: the demo PaperCandidate ships pre-extracted sections.
        if candidate.sections and any(len(v) > 200 for v in candidate.sections.values()):
            return candidate.sections

        if candidate.pdf_url:
            try:
                pdf_bytes = self._fetch(candidate.pdf_url)
            except Exception as exc:
                logger.warning(
                    "pdf fetch failed for %s (%s); falling back to abstract-only chunks",
                    candidate.arxiv_id, exc,
                )
                pdf_bytes = None

            if pdf_bytes:
                try:
                    sections = self._extract_from_pdf(pdf_bytes)
                except Exception as exc:
                    logger.warning(
                        "pdf parse failed for %s (%s); falling back to abstract-only chunks",
                        candidate.arxiv_id, exc,
                    )
                    sections = {}
                if sections and sum(len(v) for v in sections.values()) > 600:
                    return _normalize_sections(sections, candidate.abstract)
                logger.warning(
                    "pdf parse for %s yielded thin content (%d sections, %d chars); "
                    "falling back to abstract-only",
                    candidate.arxiv_id,
                    len(sections or {}),
                    sum(len(v) for v in (sections or {}).values()),
                )

        # Distinguishable fallback so downstream consumers can tell whether the
        # chunks they're looking at are full-PDF or abstract-only.
        return candidate.sections or {"abstract": candidate.abstract}

    def _fetch(self, url: str) -> Optional[bytes]:
        request = urllib.request.Request(url, headers={"User-Agent": "neuropod-research-bot/0.2"})
        with urllib.request.urlopen(request, timeout=self.fetch_timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_pdf_bytes:
                logger.info("pdf too large (%s bytes) for %s", content_length, url)
                return None
            data = response.read(self.max_pdf_bytes + 1)
        if len(data) > self.max_pdf_bytes:
            logger.info("pdf exceeds cap (%d bytes) for %s", len(data), url)
            return None
        return data

    def _extract_from_pdf(self, pdf_bytes: bytes) -> dict[str, str]:
        try:
            import fitz
        except ImportError:
            logger.warning("pymupdf not installed; cannot extract pdf")
            return {}

        sections: dict[str, list[str]] = {}
        current_label: str | None = None

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text("text") or ""
                for raw_line in text.split("\n"):
                    line = raw_line.strip()
                    if not line:
                        continue

                    if any(p.match(line) for p in STOP_HEADERS):
                        current_label = None
                        continue

                    matched_label = self._match_header(line)
                    if matched_label:
                        current_label = matched_label
                        sections.setdefault(current_label, [])
                        continue

                    if current_label is None:
                        continue
                    sections[current_label].append(line)

        cleaned: dict[str, str] = {}
        for label, lines in sections.items():
            text = " ".join(lines)
            text = re.sub(r"-\s+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 120:
                cleaned[label] = text[:8000]
        return cleaned

    def _match_header(self, line: str) -> str | None:
        if len(line) > 60:
            return None
        for label, pattern in SECTION_HEADERS:
            if pattern.match(line):
                return label
        return None


def _normalize_sections(sections: dict[str, str], candidate_abstract: str) -> dict[str, str]:
    """Stitch the abstract back in + guard against the 'everything-tagged-methods' failure mode.

    Section header detection is heuristic — many papers use unconventional
    headings ('Approach', '3.2 Proposed Framework') that our regex misses.
    When that happens the parser tends to dump most of the body into whichever
    section header DID match (often 'methods'), which falsely amplifies the
    section_bonus reranking. If we ended up with a single section that
    contains the bulk of the paper, re-tag it as a generic 'body' so retrieval
    doesn't over-trust the 'methods' label.
    """
    cleaned = {k: v for k, v in sections.items() if v}

    # Always ensure the abstract is present (PDF parse may have missed it,
    # but the arXiv API gave us a clean one).
    if "abstract" not in cleaned and candidate_abstract:
        cleaned["abstract"] = candidate_abstract.strip()

    # If only ONE substantive section made it through, the parser likely
    # misclassified everything. Re-tag the big bucket as 'body' to neutralize
    # section_bonus over-weighting.
    substantive = [k for k, v in cleaned.items() if len(v) > 1500 and k != "abstract"]
    if len(substantive) == 1 and len(cleaned) <= 2:
        big_key = substantive[0]
        body_text = cleaned.pop(big_key)
        cleaned["body"] = body_text

    return cleaned

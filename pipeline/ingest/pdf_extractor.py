from __future__ import annotations

import logging
import re
import urllib.request
from typing import Optional

from ..models import PaperCandidate

logger = logging.getLogger("neuropod.pdf")

SECTION_HEADERS = [
    ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE)),
    ("introduction", re.compile(r"^\s*(1\.?\s*)?introduction\s*$", re.IGNORECASE)),
    ("background", re.compile(r"^\s*(2\.?\s*)?(background|related work|prior work)\s*$", re.IGNORECASE)),
    ("methods", re.compile(r"^\s*(\d+\.?\s*)?(method|methods|approach|methodology|model)\s*$", re.IGNORECASE)),
    ("experiments", re.compile(r"^\s*(\d+\.?\s*)?(experiments?|setup|evaluation)\s*$", re.IGNORECASE)),
    ("results", re.compile(r"^\s*(\d+\.?\s*)?(results?|findings?)\s*$", re.IGNORECASE)),
    ("discussion", re.compile(r"^\s*(\d+\.?\s*)?(discussion|analysis)\s*$", re.IGNORECASE)),
    ("limitations", re.compile(r"^\s*(\d+\.?\s*)?limitations?\s*$", re.IGNORECASE)),
    ("conclusion", re.compile(r"^\s*(\d+\.?\s*)?(conclusion|conclusions?|summary)\s*$", re.IGNORECASE)),
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
        if candidate.sections and any(len(v) > 200 for v in candidate.sections.values()):
            return candidate.sections

        if candidate.pdf_url:
            try:
                pdf_bytes = self._fetch(candidate.pdf_url)
                if pdf_bytes:
                    sections = self._extract_from_pdf(pdf_bytes)
                    if sections and sum(len(v) for v in sections.values()) > 600:
                        if "abstract" not in sections and candidate.abstract:
                            sections["abstract"] = candidate.abstract
                        return sections
            except Exception as exc:
                logger.warning("pdf extraction failed for %s: %s", candidate.arxiv_id, exc)

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

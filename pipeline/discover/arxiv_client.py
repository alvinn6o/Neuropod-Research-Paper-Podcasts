from __future__ import annotations

import copy
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

from ..demo_catalog import get_demo_catalog
from ..models import PaperCandidate


_ATOM = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivClient:
    """arXiv client. Live-queries the API when NEUROPOD_LIVE_DISCOVERY=true; demo otherwise."""

    def __init__(self) -> None:
        self.live = os.getenv("NEUROPOD_LIVE_DISCOVERY", "false").lower() in {"1", "true", "yes"}

    def search(self, topics: list[str], days: int = 7, max_results: int = 10) -> list[PaperCandidate]:
        if self.live and topics:
            try:
                live = self._live_search(topics, max_results, days)
                if live:
                    return live
            except Exception:
                pass
        return self._demo_search(topics, max_results)

    def _live_search(self, topics: list[str], max_results: int, days: int) -> list[PaperCandidate]:
        topic_clause = " OR ".join(f'all:"{t}"' for t in topics[:5])
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(days, 1))
        date_clause = f"submittedDate:[{start.strftime('%Y%m%d%H%M')} TO {end.strftime('%Y%m%d%H%M')}]"
        full_query = f"({topic_clause}) AND {date_clause}"

        params = urllib.parse.urlencode({
            "search_query": full_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        })
        url = f"http://export.arxiv.org/api/query?{params}"
        with urllib.request.urlopen(url, timeout=30) as response:
            root = ET.fromstring(response.read())

        candidates: list[PaperCandidate] = []
        for entry in root.findall("atom:entry", _ATOM):
            arxiv_url = entry.findtext("atom:id", default="", namespaces=_ATOM)
            arxiv_id = arxiv_url.rsplit("/", 1)[-1]
            title = (entry.findtext("atom:title", default="", namespaces=_ATOM) or "").strip()
            abstract = (entry.findtext("atom:summary", default="", namespaces=_ATOM) or "").strip()
            published = entry.findtext("atom:published", default="", namespaces=_ATOM) or ""
            authors = [
                (a.findtext("atom:name", default="", namespaces=_ATOM) or "").strip()
                for a in entry.findall("atom:author", _ATOM)
            ]
            categories = [
                (c.attrib.get("term") or "").strip()
                for c in entry.findall("atom:category", _ATOM)
            ]
            pdf_url = ""
            for link in entry.findall("atom:link", _ATOM):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break

            candidates.append(PaperCandidate(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published_at=published or datetime.now(timezone.utc).isoformat(),
                pdf_url=pdf_url or arxiv_url,
                citation_count=0,
                citation_velocity=0.0,
                sections={"abstract": abstract},
            ))
        return candidates

    def _demo_search(self, topics: list[str], max_results: int) -> list[PaperCandidate]:
        topic_terms = self._topic_terms(topics)
        catalog = [copy.deepcopy(item) for item in get_demo_catalog()]
        if not topic_terms:
            return catalog[:max_results]

        matches: list[PaperCandidate] = []
        fallbacks: list[PaperCandidate] = []
        for candidate in catalog:
            haystack = " ".join(
                [candidate.title, candidate.abstract, *candidate.categories, *candidate.sections.values()]
            ).lower()
            if any(term in haystack for term in topic_terms):
                matches.append(candidate)
            else:
                fallbacks.append(candidate)

        ordered = matches + fallbacks
        return ordered[:max_results]

    def _topic_terms(self, topics: list[str]) -> set[str]:
        terms: set[str] = set()
        for topic in topics:
            for token in re.split(r"[^a-zA-Z0-9]+", topic.lower()):
                if len(token) >= 3:
                    terms.add(token)
            collapsed = topic.lower().strip()
            if collapsed:
                terms.add(collapsed)
        return terms

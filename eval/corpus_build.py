"""Build the frozen evaluation corpus.

Two phases, deliberately separated:

  select — query arXiv once and write a pinned list of ids to papers.txt.
  build  — fetch exactly those ids, verify the PDF hash, extract and chunk.

The split is the whole point. A benchmark that re-queries arXiv at run time
measures a different corpus every run, so a metric change cannot be attributed
to a code change — which is the flaw in the current `eval/ragas_eval.py`
(it claims a "curated held-out set" in a comment and then calls a live search).
Here the corpus is a committed artifact: papers.txt pins the ids, and
manifest.json pins each PDF's sha256, so a silently revised paper shows up as a
hash mismatch rather than as a mysterious metric drift.

Usage:
    python -m eval.corpus_build select --per-category 12
    python -m eval.corpus_build build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.ingest.chunker import SectionAwareChunker
from pipeline.ingest.pdf_extractor import PDFExtractor
from pipeline.models import PaperCandidate

CORPUS_DIR = ROOT / "eval" / "corpus"
PAPERS_TXT = CORPUS_DIR / "papers.txt"
MANIFEST = CORPUS_DIR / "manifest.json"
CHUNKS = CORPUS_DIR / "chunks.jsonl"
PDF_CACHE = CORPUS_DIR / "_pdf_cache"

ATOM = "{http://www.w3.org/2005/Atom}"

# Stratified across subfields so the corpus is not four flavours of one topic.
# Two-column layouts, heavy math and table-dense results sections are all
# represented, which is where the section-header state machine actually
# struggles — the corpus should contain the failure modes, not avoid them.
CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "stat.ML"]

# arXiv asks for one request every 3 seconds. Respected rather than hammered.
POLITE_DELAY_S = 3.0


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "neuropod-eval/0.1 (research benchmark)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def select(per_category: int, start_year: int, end_year: int) -> None:
    """Query arXiv and pin the resulting ids. Additive by default.

    Existing pins are preserved and new ones appended, so growing the corpus is
    a superset rather than a resample. A resampled corpus would make every
    previously published number incomparable for no reason.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    seen: list[str] = []
    if PAPERS_TXT.exists():
        seen = [
            line.strip() for line in PAPERS_TXT.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"  keeping {len(seen)} already-pinned papers")
    for category in CATEGORIES:
        for year in range(start_year, end_year + 1):
            query = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=cat:{category}+AND+submittedDate:[{year}01010000+TO+{year}12312359]"
                f"&start=0&max_results={per_category}"
                f"&sortBy=submittedDate&sortOrder=descending"
            )
            print(f"  querying {category} {year}...", flush=True)
            try:
                raw = _get(query)
            except Exception as exc:
                print(f"    failed: {exc}")
                continue
            root = ET.fromstring(raw)
            for entry in root.findall(f"{ATOM}entry"):
                url = (entry.findtext(f"{ATOM}id") or "").strip()
                m = re.search(r"abs/([\w.\-/]+?)(?:v(\d+))?$", url)
                if not m:
                    continue
                arxiv_id = m.group(1)
                version = m.group(2) or "1"
                pin = f"{arxiv_id}v{version}"
                if pin not in seen:
                    seen.append(pin)
            time.sleep(POLITE_DELAY_S)

    PAPERS_TXT.write_text(
        "# Frozen evaluation corpus. Pinned arXiv ids WITH version.\n"
        "# Additive: `select` appends, never resamples, so the corpus stays a\n"
        "# superset and old numbers remain interpretable.\n"
        + "\n".join(seen)
        + "\n"
    )
    print(f"pinned {len(seen)} papers total -> {PAPERS_TXT}")


def _pinned_ids() -> list[str]:
    if not PAPERS_TXT.exists():
        raise SystemExit("run `python -m eval.corpus_build select` first")
    return [
        line.strip()
        for line in PAPERS_TXT.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _primary_category(entry) -> str:
    node = entry.find("{http://arxiv.org/schemas/atom}primary_category")
    return node.attrib.get("term", "") if node is not None else ""


def _fetch_metadata(pin: str) -> dict | None:
    base = pin.split("v")[0] if "v" in pin.rsplit("/", 1)[-1] else pin
    url = f"http://export.arxiv.org/api/query?id_list={base}&max_results=1"
    try:
        root = ET.fromstring(_get(url))
    except Exception as exc:
        print(f"    metadata failed: {exc}")
        return None
    entry = root.find(f"{ATOM}entry")
    if entry is None:
        return None
    return {
        "arxiv_id": base,
        "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
        "abstract": " ".join((entry.findtext(f"{ATOM}summary") or "").split()),
        "authors": [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in entry.findall(f"{ATOM}author")
        ][:12],
        # arXiv puts <category term="cs.LG"/> in the ATOM namespace, not its own
        # schema namespace — the latter only carries <primary_category>.
        "categories": [
            c.attrib.get("term", "") for c in entry.findall(f"{ATOM}category")
        ],
        # NOTE: `elem or {}` is a trap here — an Element with no children is
        # falsy in ElementTree, so the fallback fires even when the element
        # exists. Must compare against None explicitly.
        "primary_category": _primary_category(entry),
        "published_at": (entry.findtext(f"{ATOM}published") or "").strip(),
    }


def build(limit: int | None = None) -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    pins = _pinned_ids()[:limit]

    extractor = PDFExtractor()
    chunker = SectionAwareChunker()
    manifest: list[dict] = []
    all_chunks: list[dict] = []

    for i, pin in enumerate(pins, start=1):
        print(f"[{i}/{len(pins)}] {pin}", flush=True)
        cached = PDF_CACHE / f"{pin.replace('/', '_')}.pdf"
        if cached.exists():
            pdf_bytes = cached.read_bytes()
        else:
            try:
                pdf_bytes = _get(f"https://arxiv.org/pdf/{pin}", timeout=90)
            except Exception as exc:
                print(f"    pdf fetch failed: {exc}")
                continue
            if not pdf_bytes.startswith(b"%PDF"):
                print("    not a pdf (paywall/withdrawn?), skipping")
                continue
            cached.write_bytes(pdf_bytes)
            time.sleep(POLITE_DELAY_S)

        meta = _fetch_metadata(pin)
        if not meta:
            print("    no metadata, skipping")
            continue
        time.sleep(1.0)

        candidate = PaperCandidate(
            arxiv_id=meta["arxiv_id"], title=meta["title"], abstract=meta["abstract"],
            authors=meta["authors"], categories=meta["categories"],
            published_at=meta["published_at"], pdf_url=f"https://arxiv.org/pdf/{pin}",
            sections={},
        )
        sections = extractor.extract_sections(candidate, pdf_bytes=pdf_bytes) \
            if _accepts_bytes(extractor) else extractor.extract_sections(candidate)
        chunks = chunker.chunk_sections(meta["arxiv_id"], sections)

        # Recorded so extraction drift is visible: if a future PyMuPDF or a
        # changed arXiv layout degrades parsing, these numbers move and the
        # corpus can be rebuilt knowingly rather than silently.
        quality = {
            "section_count": len(sections),
            "chars_per_section": {k: len(v) for k, v in sections.items()},
            "total_chars": sum(len(v) for v in sections.values()),
            "chunk_count": len(chunks),
            "body_fallback": "body" in sections,
            "abstract_only": set(sections) == {"abstract"},
        }
        manifest.append({
            "pin": pin, **meta,
            "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "pdf_bytes": len(pdf_bytes),
            "parse_quality": quality,
        })
        for c in chunks:
            d = c.to_dict()
            d.pop("embedding", None)  # recomputed deterministically; keeps the repo small
            d["paper_id"] = meta["arxiv_id"]
            all_chunks.append(d)
        print(f"    {len(sections)} sections, {len(chunks)} chunks"
              f"{' [ABSTRACT-ONLY]' if quality['abstract_only'] else ''}")

    MANIFEST.write_text(json.dumps({"papers": manifest}, indent=2))
    with CHUNKS.open("w") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c) + "\n")
    print(f"\n{len(manifest)} papers, {len(all_chunks)} chunks")
    print(f"  -> {MANIFEST}")
    print(f"  -> {CHUNKS}")


def _accepts_bytes(extractor) -> bool:
    import inspect
    return "pdf_bytes" in inspect.signature(extractor.extract_sections).parameters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--per-category", type=int, default=4)
    s.add_argument("--start-year", type=int, default=2021)
    s.add_argument("--end-year", type=int, default=2025)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "select":
        select(args.per_category, args.start_year, args.end_year)
    else:
        build(args.limit)

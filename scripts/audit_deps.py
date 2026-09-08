"""Fail if any pinned dependency is younger than the minimum release age.

Rationale: the 2026 PyPI compromises (LiteLLM, Telnyx, PyTorch Lightning) were
all published through legitimate pipelines with valid metadata, and were caught
within days. A minimum release age is the cheap control that catches that class
of attack without needing to detect the payload. Signature or provenance
validity is not evidence of safety.

Usage:  python scripts/audit_deps.py [--min-age-days 7]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]+\])?\s*==\s*([^\s#]+)")


def pinned(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0]
        m = PIN_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def released_at(name: str, version: str) -> datetime | None:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = json.load(resp)
    stamps = [f["upload_time_iso_8601"] for f in data.get("urls", []) if f.get("upload_time_iso_8601")]
    if not stamps:
        return None
    return datetime.fromisoformat(min(stamps).replace("Z", "+00:00"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-age-days", type=int, default=7)
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    seen: set[str] = set()
    violations: list[str] = []
    checked = 0

    for manifest in ("requirements.txt", "requirements-dev.txt"):
        path = ROOT / manifest
        if not path.exists():
            continue
        for name, version in pinned(path):
            key = f"{name.lower()}=={version}"
            if key in seen:
                continue
            seen.add(key)
            try:
                when = released_at(name, version)
            except Exception as exc:
                violations.append(f"{name}=={version}: lookup failed ({exc})")
                continue
            if when is None:
                violations.append(f"{name}=={version}: no upload timestamp")
                continue
            age = (now - when).days
            checked += 1
            if age < args.min_age_days:
                violations.append(
                    f"{name}=={version}: released {when.date()} ({age}d old, minimum {args.min_age_days}d)"
                )

    if violations:
        print(f"FAIL — {len(violations)} of {checked + len(violations)} pins violate policy:")
        for v in violations:
            print(f"  {v}")
        return 1
    print(f"OK — all {checked} pinned versions are at least {args.min_age_days} days old.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

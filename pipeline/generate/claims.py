"""Deterministic checks on a generated script.

Retrieval quality is measured; generation quality was not measured at all. The
only check in the pipeline was `qa_check.py` — sixteen lines of unigram set
overlap that never blocked publication.

The most damaging failure this system can have is a **fabricated number**. A
script that says a model reached 94% accuracy when the paper says 71% is worse
than a vague one: it is confidently, specifically wrong, and a listener has no
way to tell. So the first check is numeric: every number in the script must
appear in the retrieved context that produced it.

This is deliberately cheap and deterministic — no API calls, no model, and the
same input always gives the same answer, which is what lets it run on every
generation and gate in CI.

**What it is not.** A number missing from context is a *flag*, not a verdict.
Known false positives, all of them real:

  * the model derived a value the paper never states outright ("roughly a third
    faster" from 71% vs 94%);
  * it restated a figure in different units (0.71 vs 71%);
  * it referred to a quantity from its own framing rather than the paper.

Years and bare small integers are filtered out for this reason. What remains is
a grounding *rate* to track and threshold, not a hallucination detector to
trust blindly. A script scoring 0.6 deserves a look; one scoring 0.0 with ten
numeric claims is almost certainly fabricating.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches integers, decimals, thousands separators, and trailing units that
# change a number's meaning (5x is not 5, 24% is not 24).
_NUMBER = re.compile(r"\d[\d,]*\.?\d*\s*(?:%|x\b|×)?", re.IGNORECASE)

# Numbers that carry no factual claim about the paper's results.
_STOP_NUMBERS = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                           "100", "1000"})

# Word-number substitution was tried and REMOVED. Mapping "third" -> "33%"
# turned the ordinal in "the third experiment" into a numeric claim needing
# support, and naive str.replace matched inside other words entirely ("Third."
# became "33%"). The metric is about explicit numeric claims; "twice as fast"
# is not one, and treating it as one only manufactures false positives.

# Years are excluded. In this corpus a four-digit number in the modern range is
# nearly always a citation or a date ("the 2017 Transformer paper"), not a
# result, and counting them as ungrounded claims swamps the real signal.
def _is_year(normalized: str) -> bool:
    if not normalized.isdigit() or len(normalized) != 4:
        return False
    return 1900 <= int(normalized) <= 2100

MARKDOWN_TELLS = ("```", "\n#", "\n*", "\n- ", "\n1.", "**")


def normalize_number(raw: str) -> str:
    """Canonical form so '1,024' and '1024' compare equal, '24 percent' and
    '24%' compare equal, and '5.20' and '5.2' compare equal."""
    text = raw.strip().lower().replace(",", "").replace("×", "x")
    text = re.sub(r"\s+", "", text)
    suffix = ""
    if text.endswith("%"):
        suffix, text = "%", text[:-1]
    elif text.endswith("x"):
        suffix, text = "x", text[:-1]
    try:
        value = float(text)
    except ValueError:
        return raw.strip().lower()
    # Drop trailing zeros so 5.20 == 5.2, but keep integers as integers.
    formatted = f"{value:.10g}"
    return f"{formatted}{suffix}"


def extract_numbers(text: str) -> list[str]:
    """Normalized numeric claims, minus the ones that carry no information."""
    lowered = re.sub(r"\b(\d[\d,.]*)\s*percent\b", r"\1%", text.lower())

    out = []
    for match in _NUMBER.finditer(lowered):
        norm = normalize_number(match.group())
        if norm and norm not in _STOP_NUMBERS and not _is_year(norm):
            out.append(norm)
    return out


@dataclass
class ScriptReport:
    numbers_total: int = 0
    numbers_grounded: int = 0
    ungrounded: list[str] = field(default_factory=list)
    word_count: int = 0
    paragraph_count: int = 0
    markdown_leaks: list[str] = field(default_factory=list)

    @property
    def numeric_precision(self) -> float:
        """Fraction of the script's numeric claims found in retrieved context.

        1.0 when the script makes no numeric claims at all — vacuously grounded,
        and worth knowing separately from 'made claims and supported them',
        which is why `numbers_total` is reported alongside.
        """
        if self.numbers_total == 0:
            return 1.0
        return self.numbers_grounded / self.numbers_total

    @property
    def within_word_budget(self) -> bool:
        """The prompt instructs 800-1200 words. Nothing ever checked it."""
        return 800 <= self.word_count <= 1200

    @property
    def within_paragraph_budget(self) -> bool:
        return 4 <= self.paragraph_count <= 7

    def to_dict(self) -> dict:
        return {
            "numeric_precision": round(self.numeric_precision, 4),
            "numbers_total": self.numbers_total,
            "numbers_grounded": self.numbers_grounded,
            "ungrounded": self.ungrounded[:20],
            "word_count": self.word_count,
            "within_word_budget": self.within_word_budget,
            "paragraph_count": self.paragraph_count,
            "within_paragraph_budget": self.within_paragraph_budget,
            "markdown_leaks": self.markdown_leaks,
        }


def check_script(script: str, chunks: list[dict], *, abstract: str = "") -> ScriptReport:
    """Score a script against the context that produced it.

    `abstract` is included in the grounding context because the scriptwriter
    prompt supplies it alongside the retrieved chunks — a number the model took
    from the abstract is grounded, and counting it as a fabrication would make
    the metric wrong in a way that punishes correct behaviour.
    """
    context = " ".join([abstract] + [c.get("content", "") for c in chunks])
    context_numbers = set(extract_numbers(context))

    script_numbers = extract_numbers(script)
    ungrounded = [n for n in script_numbers if n not in context_numbers]

    paragraphs = [p for p in re.split(r"\n\s*\n", script.strip()) if p.strip()]

    return ScriptReport(
        numbers_total=len(script_numbers),
        numbers_grounded=len(script_numbers) - len(ungrounded),
        ungrounded=ungrounded,
        word_count=len(script.split()),
        paragraph_count=len(paragraphs),
        markdown_leaks=[t.strip() for t in MARKDOWN_TELLS if t in script],
    )

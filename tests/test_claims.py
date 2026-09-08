"""Deterministic generation checks.

Retrieval quality was measured from Phase 1; generation quality was not
measured at all. The only check in the pipeline was sixteen lines of unigram
overlap that never blocked publication.

The failure these tests target is a **fabricated number**: a script that states
a result the paper never reported. It is the most damaging thing this system
can do, because it is confidently and specifically wrong, and a listener cannot
tell.
"""
from __future__ import annotations

import pytest

from pipeline.generate.claims import check_script, extract_numbers, normalize_number
from pipeline.orchestrator import MIN_CLAIMS_TO_JUDGE, _grade_claims

CONTEXT = [{"content": "We observe 71% accuracy and a 5.2x speedup over the baseline on 8 GPUs."}]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1,024", "1024"),
    ("24 %", "24%"),
    ("5.20x", "5.2x"),
    ("5.2×", "5.2x"),
    ("0.710", "0.71"),
])
def test_numbers_normalize_to_a_canonical_form(raw, expected):
    """A script saying '1,024' and a chunk saying '1024' are the same claim.
    Without normalization the metric would report fabrications that are
    formatting differences."""
    assert normalize_number(raw) == expected


def test_percent_written_as_a_word_matches_the_symbol():
    assert "24%" in extract_numbers("it improved by 24 percent")
    assert "24%" in extract_numbers("it improved by 24%")


# ---------------------------------------------------------------------------
# What is deliberately NOT counted
# ---------------------------------------------------------------------------

def test_years_are_not_treated_as_numeric_claims():
    """In this corpus a four-digit modern number is nearly always a citation or
    a date, not a result. Counting them swamps the real signal."""
    assert extract_numbers("the 2017 Transformer paper and 2024 follow-up") == []


def test_bare_small_integers_are_ignored():
    """'one of three approaches' is not a result claim."""
    assert extract_numbers("we compare 3 methods across 2 datasets") == []


def test_ordinals_do_not_become_fractions():
    """Regression test. An earlier version mapped word-numbers to digits with
    str.replace, so 'the third experiment' became a 33% claim needing support —
    and 'Third.' at the start of a paragraph matched too."""
    assert extract_numbers("Third. The third experiment halved the error.") == []


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

def test_a_grounded_script_scores_one():
    report = check_script("It reached 71% accuracy with a 5.2x speedup on 8 GPUs.", CONTEXT)
    assert report.numeric_precision == 1.0
    assert report.ungrounded == []


def test_a_fabricated_number_is_caught():
    """The canary. Every number here is plausible and none appear in context."""
    report = check_script("It hit 94% accuracy, a 12.7x speedup, and 33.5 BLEU.", CONTEXT)
    assert report.numeric_precision == 0.0
    assert set(report.ungrounded) == {"94%", "12.7x", "33.5"}


def test_swapping_one_digit_is_detected():
    """The realistic failure mode: not an invented paragraph, a changed digit."""
    honest = check_script("Accuracy was 71%.", CONTEXT)
    altered = check_script("Accuracy was 17%.", CONTEXT)
    assert honest.numeric_precision == 1.0
    assert altered.numeric_precision == 0.0


def test_the_abstract_counts_as_grounding_context():
    """The prompt supplies the abstract alongside retrieved chunks, so a number
    taken from it IS grounded. Counting it as fabrication would penalise
    correct behaviour."""
    report = check_script(
        "The method reaches 93.7 on the benchmark.", CONTEXT,
        abstract="Our approach reaches 93.7 on the benchmark.",
    )
    assert report.numeric_precision == 1.0


def test_no_numeric_claims_is_reported_separately_from_supported_ones():
    """1.0 for a script that says nothing numeric is vacuous, so the count is
    carried alongside the rate."""
    report = check_script("The paper describes a new approach.", CONTEXT)
    assert report.numeric_precision == 1.0
    assert report.numbers_total == 0


# ---------------------------------------------------------------------------
# Format checks the prompt asks for and nothing verified
# ---------------------------------------------------------------------------

def test_word_and_paragraph_budgets_are_checked():
    """`scriptwriter.SYSTEM_PROMPT` demands 800-1200 words in 4-7 paragraphs.
    Nothing had ever verified either."""
    short = check_script("Too short.", CONTEXT)
    assert not short.within_word_budget

    ok = check_script("\n\n".join([" ".join(["word"] * 250)] * 4), CONTEXT)
    assert ok.within_word_budget and ok.within_paragraph_budget


def test_markdown_leakage_is_detected():
    """The prompt says pure spoken prose — markdown read aloud is gibberish."""
    assert check_script("Intro\n\n## Heading\n\ntext", CONTEXT).markdown_leaks
    assert check_script("Intro.\n\nClean prose.", CONTEXT).markdown_leaks == []


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_fabrication_flags_the_episode():
    report = check_script("It hit 94% accuracy, a 12.7x speedup, and 33.5 BLEU.", CONTEXT)
    status, note = _grade_claims(report, "test")
    assert status == "flagged"
    assert "94%" in note and "not supported" in note


def test_a_single_odd_number_does_not_flag():
    """Below MIN_CLAIMS_TO_JUDGE one unmatched figure drops precision to 0.5
    and the flag would be noise, not signal."""
    report = check_script("Accuracy was 71% and something scored 99.9.", CONTEXT)
    assert report.numbers_total < MIN_CLAIMS_TO_JUDGE
    assert _grade_claims(report, "test")[0] == "verified"


def test_the_note_names_the_offending_numbers():
    """The previous QA note said 'terms may not be fully grounded', which is not
    actionable. This one has to name them."""
    report = check_script("94% accuracy, 12.7x speedup, 33.5 BLEU, 88.2 F1.", CONTEXT)
    _, note = _grade_claims(report, "test")
    for number in ("94%", "12.7x", "33.5"):
        assert number in note

from __future__ import annotations

import logging
import os
import time

from .._http import ProviderError, post_json
from ..models import PaperCandidate
from ..provider_status import record_failure, record_success

logger = logging.getLogger("neuropod.scriptwriter")

SYSTEM_PROMPT = (
    "You are a senior research analyst writing 6-9 minute audio briefs for a daily AI research "
    "podcast. Write conversational, technically accurate narration grounded ONLY in the provided "
    "paper content. Open with the single most surprising or counterintuitive result. Then explain: "
    "(1) the problem the paper addresses and why it matters, (2) the core method or mechanism — "
    "in plain language, but with specifics, no hand-waving, (3) the key quantitative results "
    "and how they were measured, (4) limitations the authors call out or that follow from the "
    "method, (5) what this means for the field — what becomes possible or what's now in question. "
    "Use 800–1200 words. No bullet points, no markdown, no headers — pure spoken prose, broken "
    "into 4-7 paragraphs. Do not invent results that aren't in the source material. Do not pad."
)


class ScriptWriter:
    def __init__(self) -> None:
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.provider = os.getenv("NEUROPOD_LLM_PROVIDER", "auto").lower()

    def write(
        self,
        candidate: PaperCandidate,
        retrieved_chunks: list[dict],
        audience_topics: list[str],
    ) -> tuple[str, str]:
        """Returns (script, provider_label_used)."""
        prompt = self._build_prompt(candidate, retrieved_chunks, audience_topics)

        if self.provider in {"anthropic", "auto"} and self.anthropic_key:
            try:
                return self._call_anthropic(prompt), "anthropic"
            except ProviderError as exc:
                record_failure("script:anthropic", error=exc.detail, status=exc.status)
                logger.warning("anthropic failed, falling back: %s", exc)

        if self.provider in {"openai", "auto"} and self.openai_key:
            try:
                return self._call_openai(prompt), "openai"
            except ProviderError as exc:
                record_failure("script:openai", error=exc.detail, status=exc.status)
                logger.warning("openai failed, falling back: %s", exc)

        return self._fallback(candidate, retrieved_chunks, audience_topics), "demo"

    def _build_prompt(
        self,
        candidate: PaperCandidate,
        retrieved_chunks: list[dict],
        topics: list[str],
    ) -> str:
        topic_line = ", ".join(topics[:5]) if topics else "general AI research"
        chunk_lines = []
        for chunk in retrieved_chunks[:14]:
            chunk_lines.append(f"[{chunk['section']}]\n{chunk['content']}")
        chunks_block = "\n\n".join(chunk_lines)

        return (
            f"PAPER: {candidate.title}\n"
            f"AUTHORS: {', '.join(candidate.authors)}\n"
            f"CATEGORIES: {', '.join(candidate.categories)}\n"
            f"AUDIENCE TOPICS: {topic_line}\n\n"
            f"PAPER ABSTRACT:\n{candidate.abstract}\n\n"
            f"RETRIEVED PAPER SECTIONS:\n{chunks_block}\n\n"
            "Write the 6-9 minute narration now. 800-1200 words. Pure spoken prose."
        )

    def _call_anthropic(self, prompt: str) -> str:
        start = time.time()
        result = post_json(
            provider="anthropic",
            url="https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
            },
            body={
                "model": "claude-sonnet-4-6",
                "max_tokens": 2400,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        record_success("script:anthropic", latency_ms=int((time.time() - start) * 1000))
        return result["content"][0]["text"].strip()

    def _call_openai(self, prompt: str) -> str:
        start = time.time()
        result = post_json(
            provider="openai",
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key}",
            },
            body={
                "model": "gpt-4o-mini",
                "max_tokens": 2400,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=90,
        )
        record_success("script:openai", latency_ms=int((time.time() - start) * 1000))
        return result["choices"][0]["message"]["content"].strip()

    def _fallback(
        self,
        candidate: PaperCandidate,
        retrieved_chunks: list[dict],
        topics: list[str],
    ) -> str:
        sections = {chunk["section"]: chunk["content"] for chunk in retrieved_chunks}
        focus = ", ".join(topics[:3]) if topics else "your tracked research topics"
        hook = f"If you care about {focus}, today's paper is interesting because {self._sentence(candidate.abstract)}"
        context = self._sentence(sections.get("introduction", candidate.sections.get("introduction", candidate.abstract)))
        findings = self._sentence(sections.get("results", candidate.sections.get("results", candidate.abstract)))
        method = self._sentence(sections.get("methods", candidate.sections.get("methods", candidate.abstract)))
        conclusion = self._sentence(sections.get("conclusion", candidate.sections.get("conclusion", candidate.abstract)))

        return " ".join([
            hook,
            f"The paper, titled {candidate.title}, focuses on a practical bottleneck in modern AI systems. {context}",
            f"The main takeaway is that {findings}",
            f"Under the hood, the authors use a relatively simple idea: {method}",
            f"What makes it worth watching next is the broader implication. {conclusion}",
            "For a daily podcast format, that means the work is both technically credible and easy to map to where the field may move next.",
        ])

    def _sentence(self, text: str) -> str:
        sentence = text.strip().split(".")[0].strip()
        return sentence if sentence.endswith(".") else f"{sentence}."

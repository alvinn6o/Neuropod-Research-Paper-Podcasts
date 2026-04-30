from __future__ import annotations


class QAChecker:
    def verify(self, script: str, chunks: list[dict]) -> tuple[str, str]:
        script_terms = {token for token in script.lower().split() if len(token) > 4}
        source_terms = {
            token
            for chunk in chunks
            for token in chunk["content"].lower().split()
            if len(token) > 4
        }
        overlap = len(script_terms.intersection(source_terms)) / max(len(script_terms), 1)
        if overlap >= 0.3:
            return "verified", "Script language overlaps strongly with retrieved source chunks."
        return "flagged", "Script contains terms that may not be fully grounded in the retrieved context."

from __future__ import annotations


class AudioProcessor:
    words_per_minute = 155

    def estimate_duration_secs(self, script: str) -> int:
        words = len(script.split())
        return max(int((words / self.words_per_minute) * 60), 60)

    def build_description(self, script: str) -> str:
        summary = " ".join(script.split()[:40]).strip()
        if not summary.endswith("."):
            summary += "..."
        return summary

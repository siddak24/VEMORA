from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionContext:
    """
    Lightweight representation of the current session
    that can be supplied to the SLM.
    """

    session_id: int | None = None

    recent_transcript: list[str] = field(
        default_factory=list
    )

    session_type: str = "conversation"

    def as_text(self) -> str:

        if not self.recent_transcript:
            return ""

        return "\n".join(
            self.recent_transcript
        )
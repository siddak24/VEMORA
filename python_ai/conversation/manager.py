from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass
class ConversationTurn:
    user: str
    assistant: str


class ConversationManager:
    """
    Short-term conversational context for direct VEMORA
    interactions.

    This is separate from:
        - long-term memory
        - active session transcript

    It is used for follow-up questions and references such as:
        "What about tomorrow?"
        "And where is it?"
        "Who was that?"
    """

    def __init__(
        self,
        max_turns: int = 5,
        follow_up_timeout: float = 8.0,
    ) -> None:

        self.max_turns = max_turns
        self.follow_up_timeout = follow_up_timeout

        self.turns: list[ConversationTurn] = []

        # Timestamp of the most recent direct interaction.
        self.last_interaction_time: float | None = None

        # True while the short conversational window is active.
        self.active = False

    # ==========================================================
    # ADD TURN
    # ==========================================================

    def add_turn(
        self,
        user: str,
        assistant: str,
    ) -> None:

        self.turns.append(
            ConversationTurn(
                user=user.strip(),
                assistant=assistant.strip(),
            )
        )

        if len(self.turns) > self.max_turns:

            self.turns = (
                self.turns[-self.max_turns:]
            )

        self.last_interaction_time = monotonic()
        self.active = True

    # ==========================================================
    # CHECK WHETHER FOLLOW-UP WINDOW IS ACTIVE
    # ==========================================================

    def is_active(self) -> bool:

        if not self.active:
            return False

        if self.last_interaction_time is None:
            return False

        elapsed = (
            monotonic()
            - self.last_interaction_time
        )

        if elapsed > self.follow_up_timeout:

            self.active = False

            return False

        return True

    # ==========================================================
    # TIME REMAINING
    # ==========================================================

    def seconds_remaining(self) -> float:

        if not self.is_active():

            return 0.0

        elapsed = (
            monotonic()
            - self.last_interaction_time
        )

        return max(
            0.0,
            self.follow_up_timeout - elapsed,
        )

    # ==========================================================
    # CONTEXT
    # ==========================================================

    def context(self) -> str:

        if not self.turns:

            return ""

        parts: list[str] = []

        for index, turn in enumerate(
            self.turns,
            start=1,
        ):

            parts.append(
                f"Turn {index}:\n"
                f"User: {turn.user}\n"
                f"VEMORA: {turn.assistant}"
            )

        return "\n\n".join(parts)

    # ==========================================================
    # CLEAR
    # ==========================================================

    def clear(self) -> None:

        self.turns.clear()

        self.last_interaction_time = None
        self.active = False
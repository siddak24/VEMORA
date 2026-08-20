from __future__ import annotations

from memory.database import MemoryDatabase
from session.models import SessionState


class SessionManager:
    """
    Manages VEMORA's live conversation/session state.

    V0.1:
        Session lifecycle + transcript storage.

    Later:
        VAD
        automatic pausing
        rolling summaries
        session finalization
        memory extraction
    """

    def __init__(
        self,
        database: MemoryDatabase,
        user_id: str = "default_user",
    ) -> None:

        self.database = database
        self.user_id = user_id

        self.session_id: int | None = None
        self.state = SessionState.IDLE

        self.sequence = 0

    # ==========================================================
    # START
    # ==========================================================

    def start(
        self,
        session_type: str = "conversation",
    ) -> int:

        if self.state != SessionState.IDLE:

            raise RuntimeError(
                "A session is already active."
            )

        self.session_id = (
            self.database.create_session(
                user_id=self.user_id,
                session_type=session_type,
            )
        )

        self.sequence = 0
        self.state = SessionState.ACTIVE

        print(
            f"[SESSION] Started #{self.session_id}"
        )

        return self.session_id

    # ==========================================================
    # ADD TRANSCRIPT
    # ==========================================================

    def add_transcript(
        self,
        text: str,
    ) -> int:

        if self.state != SessionState.ACTIVE:

            raise RuntimeError(
                "Cannot add transcript while "
                "session is not active."
            )

        if self.session_id is None:

            raise RuntimeError(
                "No session ID."
            )

        text = text.strip()

        if not text:
            return -1

        self.sequence += 1

        chunk_id = (
            self.database.add_transcript_chunk(
                session_id=self.session_id,
                sequence=self.sequence,
                text=text,
            )
        )

        return chunk_id

    # ==========================================================
    # PAUSE
    # ==========================================================

    def pause(self) -> None:

        if self.state != SessionState.ACTIVE:
            return

        if self.session_id is None:
            return

        self.database.update_session_state(
            session_id=self.session_id,
            state=SessionState.PAUSED.value,
        )

        self.state = SessionState.PAUSED

        print(
            "[SESSION] Paused."
        )

    # ==========================================================
    # RESUME
    # ==========================================================

    def resume(self) -> None:

        if self.state != SessionState.PAUSED:
            return

        if self.session_id is None:
            return

        self.database.update_session_state(
            session_id=self.session_id,
            state=SessionState.ACTIVE.value,
        )

        self.state = SessionState.ACTIVE

        print(
            "[SESSION] Resumed."
        )

    # ==========================================================
    # RECENT CONTEXT
    # ==========================================================

    def recent_context(
        self,
        limit: int = 10,
    ) -> str:

        if self.session_id is None:
            return ""

        chunks = (
            self.database
            .get_recent_session_chunks(
                self.session_id,
                limit=limit,
            )
        )

        return "\n".join(
            chunk["text"]
            for chunk in chunks
        )

    # ==========================================================
    # FULL TRANSCRIPT
    # ==========================================================

    def full_transcript(self) -> str:

        if self.session_id is None:
            return ""

        chunks = (
            self.database
            .get_session_chunks(
                self.session_id
            )
        )

        return "\n".join(
            chunk["text"]
            for chunk in chunks
        )

    # ==========================================================
    # END
    # ==========================================================

    def end(
        self,
        summary: str | None = None,
    ) -> None:

        if self.session_id is None:
            return

        self.state = SessionState.FINALIZING

        self.database.close_session(
            session_id=self.session_id,
            summary=summary,
        )

        print(
            f"[SESSION] Ended #{self.session_id}"
        )

        self.session_id = None
        self.sequence = 0
        self.state = SessionState.IDLE
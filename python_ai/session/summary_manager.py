from __future__ import annotations

from memory.database import MemoryDatabase


class SessionSummaryManager:
    """
    Manages section-level and session-level summaries.
    """

    def __init__(
        self,
        database: MemoryDatabase,
    ) -> None:

        self.database = database

    # ==========================================================
    # SECTION SUMMARY
    # ==========================================================

    def save_section_summary(
        self,
        session_id: int,
        start_sequence: int,
        end_sequence: int,
        start_time: str | None,
        end_time: str | None,
        content: str,
    ) -> int:

        return self.database.add_session_summary(
            session_id=session_id,
            summary_type="SECTION",
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            start_time=start_time,
            end_time=end_time,
            content=content,
        )

    # ==========================================================
    # WHOLE SESSION SUMMARY
    # ==========================================================

    def save_session_summary(
        self,
        session_id: int,
        content: str,
    ) -> int:

        return self.database.add_session_summary(
            session_id=session_id,
            summary_type="SESSION",
            start_sequence=None,
            end_sequence=None,
            start_time=None,
            end_time=None,
            content=content,
        )

    # ==========================================================
    # SECTION SUMMARIES
    # ==========================================================

    def get_sections(
        self,
        session_id: int,
    ) -> list[dict]:

        return self.database.get_session_summaries(
            session_id=session_id,
            summary_type="SECTION",
        )

    # ==========================================================
    # SESSION SUMMARY
    # ==========================================================

    def get_session_summary(
        self,
        session_id: int,
    ) -> list[dict]:

        return self.database.get_session_summaries(
            session_id=session_id,
            summary_type="SESSION",
        )
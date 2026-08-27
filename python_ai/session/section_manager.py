from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from memory.database import MemoryDatabase


@dataclass
class SessionSection:
    """
    One overlapping time window within a session.
    """

    number: int
    start_time: datetime
    end_time: datetime

    @property
    def display_start(self) -> str:
        return self.start_time.strftime("%H:%M")

    @property
    def display_end(self) -> str:
        return self.end_time.strftime("%H:%M")

    @property
    def label(self) -> str:
        return (
            f"Section {self.number}: "
            f"{self.display_start}–{self.display_end}"
        )


class SessionSectionManager:
    """
    Creates and retrieves overlapping session sections.

    Configuration:

        Section duration = 22 minutes
        Overlap          = 2 minutes
        Step             = 20 minutes

    Example:

        Section 1 → 00:00–00:22
        Section 2 → 00:20–00:42
        Section 3 → 00:40–01:02
        Section 4 → 01:00–01:22

    Internally, precise timestamps are used.
    """

    SECTION_MINUTES = 22
    OVERLAP_MINUTES = 2

    def __init__(
        self,
        database: MemoryDatabase,
        section_minutes: int = SECTION_MINUTES,
        overlap_minutes: int = OVERLAP_MINUTES,
    ) -> None:

        if section_minutes <= overlap_minutes:

            raise ValueError(
                "section_minutes must be greater than "
                "overlap_minutes."
            )

        self.database = database

        self.section_minutes = section_minutes
        self.overlap_minutes = overlap_minutes

        self.step_minutes = (
            section_minutes
            - overlap_minutes
        )

        self.sections: list[SessionSection] = []

    # ==========================================================
    # CREATE FIRST SECTION
    # ==========================================================

    def create_first_section(
        self,
        start_time: datetime,
    ) -> SessionSection:

        end_time = (
            start_time
            + timedelta(
                minutes=self.section_minutes
            )
        )

        section = SessionSection(
            number=1,
            start_time=start_time,
            end_time=end_time,
        )

        self.sections = [section]

        return section

    # ==========================================================
    # CREATE NEXT SECTION
    # ==========================================================

    def create_next_section(
        self,
    ) -> SessionSection:

        if not self.sections:

            raise RuntimeError(
                "No section exists yet."
            )

        previous = self.sections[-1]

        start_time = (
            previous.start_time
            + timedelta(
                minutes=self.step_minutes
            )
        )

        end_time = (
            start_time
            + timedelta(
                minutes=self.section_minutes
            )
        )

        section = SessionSection(
            number=previous.number + 1,
            start_time=start_time,
            end_time=end_time,
        )

        self.sections.append(
            section
        )

        return section

    # ==========================================================
    # GET CURRENT SECTION
    # ==========================================================

    @property
    def current_section(
        self,
    ) -> SessionSection | None:

        if not self.sections:

            return None

        return self.sections[-1]

    # ==========================================================
    # GET CHUNKS FOR SECTION
    # ==========================================================

    def get_section_chunks(
        self,
        session_id: int,
        section: SessionSection,
    ) -> list[dict]:
        """
        Retrieve all transcript chunks belonging to
        the section's timestamp window.

        Direct VEMORA commands are excluded because they
        are interaction metadata rather than session content.
        """

        chunks = (
            self.database
            .get_session_chunks_in_time_range(
                session_id=session_id,
                start_time=(
                    section.start_time.isoformat()
                ),
                end_time=(
                    section.end_time.isoformat()
                ),
            )
        )

        return [
            chunk
            for chunk in chunks
            if chunk["chunk_type"]
            not in {
                "DIRECT_COMMAND",
                "USER_QUESTION",
            }
        ]

    # ==========================================================
    # GET SECTION TEXT
    # ==========================================================

    def get_section_text(
        self,
        session_id: int,
        section: SessionSection,
    ) -> str:
        """
        Return the transcript belonging to the section
        as one text block.
        """

        chunks = self.get_section_chunks(
            session_id=session_id,
            section=section,
        )

        return "\n".join(
            chunk["text"]
            for chunk in chunks
        )

    # ==========================================================
    # SECTION CONTAINS DATA?
    # ==========================================================

    def section_has_data(
        self,
        session_id: int,
        section: SessionSection,
    ) -> bool:

        return bool(
            self.get_section_chunks(
                session_id=session_id,
                section=section,
            )
        )

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self) -> None:

        self.sections.clear()
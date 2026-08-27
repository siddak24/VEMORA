from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from memory.database import MemoryDatabase
from session.models import SessionState


class SessionManager:
    """
    Manages the current VEMORA session.

    Responsibilities:

        1. Manage session lifecycle.
        2. Store transcript chunks.
        3. Generate transcript embeddings.
        4. Perform semantic session search.
        5. Track precise transcript timestamps.
        6. Maintain overlapping time-based section windows.

    Session data is temporary/current context.
    It is separate from long-term memory.

    Section strategy:

        SECTION_DURATION_MINUTES = 20
        SECTION_OVERLAP_MINUTES = 2

    Example:

        Section 1 -> 00:00 - 00:20
        Section 2 -> 00:18 - 00:38
        Section 3 -> 00:36 - 00:56
        Section 4 -> 00:54 - 01:14
    """

    # ==========================================================
    # SECTION CONFIGURATION
    # ==========================================================

    SECTION_DURATION_MINUTES = 20
    SECTION_OVERLAP_MINUTES = 2

    def __init__(
        self,
        database: MemoryDatabase,
        user_id: str = "default_user",
        embedding_model=None,
    ) -> None:

        self.database = database
        self.user_id = user_id

        self.session_id: int | None = None
        self.state = SessionState.IDLE
        self.sequence = 0

        # ------------------------------------------------------
        # Precise timestamp tracking for the current session.
        #
        # We keep this in memory for now.
        # We will also persist timestamps in the database
        # in the next database update.
        # ------------------------------------------------------

        self.chunk_timestamps: dict[
            int,
            datetime,
        ] = {}

        # ------------------------------------------------------
        # Section window state.
        # ------------------------------------------------------

        self.section_start_time: datetime | None = None
        self.section_start_sequence: int | None = None

        self.section_number = 0

        # Reuse MemoryManager's embedding model when possible.
        if embedding_model is not None:

            self.embedding_model = embedding_model

        else:

            print(
                "[SESSION] Loading embedding model..."
            )

            self.embedding_model = (
                SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )
            )

            print(
                "[SESSION] Embedding model ready."
            )

    # ==========================================================
    # CURRENT TIME
    # ==========================================================

    @staticmethod
    def _now() -> datetime:
        """
        Return the current local-aware timestamp.

        Using an offset-aware datetime lets us preserve the
        user's actual timezone instead of silently assuming UTC.
        """

        return datetime.now(
            timezone.utc
        ).astimezone()

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

        self.chunk_timestamps.clear()

        self.section_start_time = None
        self.section_start_sequence = None
        self.section_number = 0

        self.state = SessionState.ACTIVE

        print(
            f"[SESSION] Started #{self.session_id}"
        )

        return self.session_id

    # ==========================================================
    # CLASSIFY TRANSCRIPT CHUNK
    # ==========================================================

    @staticmethod
    def classify_chunk(
        text: str,
        chunk_type: str = "PASSIVE",
    ) -> str:
        """
        Classify transcript content before it enters the
        hierarchical memory system.

        Types:
            PASSIVE
            USER_QUESTION
            DIRECT_COMMAND
        """

        if chunk_type == "DIRECT_COMMAND":
            return "DIRECT_COMMAND"

        text = text.strip().lower()

        if not text:
            return "PASSIVE"

        # ------------------------------------------------------
        # Obvious question patterns.
        # ------------------------------------------------------

        question_starters = (
            "who ",
            "what ",
            "when ",
            "where ",
            "why ",
            "how ",
            "can you ",
            "could you ",
            "would you ",
            "do you ",
            "did you ",
            "is ",
            "are ",
            "was ",
            "were ",
            "will you ",
            "tell me ",
            "give me ",
            "summarize ",
            "summarise ",
            "explain ",
        )

        # ------------------------------------------------------
        # Conversational follow-up patterns.
        # ------------------------------------------------------

        follow_up_patterns = (
            "and who ",
            "and what ",
            "and when ",
            "and where ",
            "and why ",
            "and how ",
            "what about ",
            "how about ",
        )

        if any(
            text.startswith(pattern)
            for pattern in question_starters
        ):
            return "USER_QUESTION"

        if any(
            text.startswith(pattern)
            for pattern in follow_up_patterns
        ):
            return "USER_QUESTION"

        return "PASSIVE"

    # ==========================================================
    # ADD TRANSCRIPT
    # ==========================================================

    def add_transcript(
        self,
        text: str,
        chunk_type: str = "PASSIVE",
    ) -> int:

        if self.state != SessionState.ACTIVE:

            raise RuntimeError(
                "Session is not active."
            )

        if self.session_id is None:

            raise RuntimeError(
                "No active session."
            )

        text = text.strip()

        if not text:
            return -1

        chunk_type = self.classify_chunk(
            text=text,
            chunk_type=chunk_type,
        )
                
        self.sequence += 1

        # ------------------------------------------------------
        # Precise timestamp for this transcript chunk.
        # ------------------------------------------------------

        chunk_time = self._now()

        self.chunk_timestamps[
            self.sequence
        ] = chunk_time

        # ------------------------------------------------------
        # Start first section when the first transcript
        # chunk arrives.
        # ------------------------------------------------------

        if self.section_start_time is None:

            self.section_start_time = chunk_time

            self.section_start_sequence = (
                self.sequence
            )

            self.section_number = 1

            print(
                "[SESSION] "
                f"Started section #1 at "
                f"{chunk_time.isoformat()}"
            )

        # ------------------------------------------------------
        # Create semantic embedding.
        # ------------------------------------------------------

        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()

        chunk_id = (
            self.database.add_transcript_chunk(
                session_id=self.session_id,
                sequence=self.sequence,
                text=text,
                embedding=embedding,
                chunk_type=chunk_type,
                recorded_at=chunk_time.isoformat(),
            )
        )

        return chunk_id

    # ==========================================================
    # SECTION STATUS
    # ==========================================================

    def section_is_ready(self) -> bool:
        """
        Return True when the current section has reached
        the configured duration.

        Important:

            This does NOT create the summary.

        It only tells the higher-level summarizer that
        enough transcript has accumulated.
        """

        if (
            self.section_start_time is None
            or self.state != SessionState.ACTIVE
        ):

            return False

        elapsed = (
            self._now()
            - self.section_start_time
        )

        return (
            elapsed.total_seconds()
            >= self.SECTION_DURATION_MINUTES * 60
        )

    # ==========================================================
    # CURRENT SECTION RANGE
    # ==========================================================

    def current_section_range(
        self,
    ) -> tuple[int, int] | None:
        """
        Return the sequence range currently belonging
        to the active section.

        Example:

            (1, 42)
        """

        if (
            self.section_start_sequence is None
            or self.sequence <= 0
        ):

            return None

        return (
            self.section_start_sequence,
            self.sequence,
        )

    # ==========================================================
    # CLOSE CURRENT SECTION
    # ==========================================================

    def close_current_section(
        self,
    ) -> dict | None:
        """
        Close the current section and create metadata
        for the next overlapping section.

        Example:

            Current section:
                1 -> 60

            With 2-minute overlap:
                next section begins around sequence
                corresponding to 18 minutes.

        Returns metadata for the section that was closed.

        The actual summary will be generated elsewhere.
        """

        if (
            self.section_start_time is None
            or self.section_start_sequence is None
            or self.session_id is None
        ):

            return None

        end_time = self._now()

        closed_section = {
            "session_id": self.session_id,
            "section_number": self.section_number,
            "start_sequence": (
                self.section_start_sequence
            ),
            "end_sequence": self.sequence,
            "start_time": (
                self.section_start_time
            ),
            "end_time": end_time,
        }

        # ------------------------------------------------------
        # Find the next section's overlapping start.
        #
        # New section starts:
        #
        #     end_time - overlap duration
        #
        # Then we find the first stored transcript chunk
        # at or after that timestamp.
        # ------------------------------------------------------

        overlap_seconds = (
            self.SECTION_OVERLAP_MINUTES
            * 60
        )

        next_section_start_time = (
            end_time
            - timedelta(
                seconds=overlap_seconds
            )
        )

        candidate_sequence = (
            self.sequence
        )

        for sequence, timestamp in (
            self.chunk_timestamps.items()
        ):

            if timestamp >= next_section_start_time:

                candidate_sequence = sequence
                break

        self.section_start_sequence = (
            candidate_sequence
        )

        self.section_start_time = (
            self.chunk_timestamps.get(
                candidate_sequence,
                next_section_start_time,
            )
        )

        self.section_number += 1

        print()
        print(
            "[SESSION] Closed section "
            f"#{closed_section['section_number']}"
        )

        print(
            "[SESSION] Section range: "
            f"{closed_section['start_sequence']}"
            f"-"
            f"{closed_section['end_sequence']}"
        )

        print(
            "[SESSION] Next section starts at "
            f"sequence {self.section_start_sequence}"
        )

        return closed_section

    # ==========================================================
    # RECENT CHUNKS
    # ==========================================================

    def recent_chunks(
        self,
        limit: int = 10,
    ) -> list[dict]:

        if self.session_id is None:
            return []

        return (
            self.database
            .get_recent_session_chunks(
                session_id=self.session_id,
                limit=limit,
            )
        )

    # ==========================================================
    # RECENT CONTEXT
    # ==========================================================

    def recent_context(
        self,
        limit: int = 10,
    ) -> str:

        chunks = self.recent_chunks(
            limit
        )

        return "\n".join(
            chunk["text"]
            for chunk in chunks
        )

    # ==========================================================
    # SESSION SUMMARY
    # ==========================================================

    def get_session_summary(
        self,
    ) -> dict | None:
        """
        Return the latest whole-session summary.
        """

        if self.session_id is None:
            return None

        return (
            self.database
            .get_latest_session_summary(
                session_id=self.session_id,
            )
        )


    # ==========================================================
    # SECTION SUMMARIES
    # ==========================================================

    def get_section_summaries(
        self,
    ) -> list[dict]:
        """
        Return all section summaries for the current session.
        """

        if self.session_id is None:
            return []

        return (
            self.database
            .get_section_summaries(
                session_id=self.session_id,
            )
        )

    # ==========================================================
    # GET SECTION TRANSCRIPT
    # ==========================================================

    def get_section_chunks(
        self,
        start_sequence: int,
        end_sequence: int,
    ) -> list[dict]:

        if self.session_id is None:

            return []

        chunks = (
            self.database
            .get_session_chunks(
                session_id=self.session_id
            )
        )

        return [
            chunk
            for chunk in chunks
            if (
                start_sequence
                <= chunk["sequence"]
                <= end_sequence
            )
        ]

    # ==========================================================
    # GET SECTION TEXT
    # ==========================================================

    def get_section_text(
        self,
        start_sequence: int,
        end_sequence: int,
    ) -> str:

        chunks = self.get_section_chunks(
            start_sequence=start_sequence,
            end_sequence=end_sequence,
        )

        return "\n".join(
            chunk["text"]
            for chunk in chunks
        )

    # ==========================================================
    # SEMANTIC SEARCH
    # ==========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.25,
        neighbor_radius: int = 1,
    ) -> list[dict]:

        if self.session_id is None:
            return []

        query = query.strip()

        if not query:
            return []

        query_embedding = np.asarray(
            self.embedding_model.encode(
                query,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

        chunks = (
            self.database
            .get_session_chunks_with_embeddings(
                session_id=self.session_id
            )
        )

        scored: list[dict] = []

        for chunk in chunks:

            # Don't use direct commands as evidence.
            if chunk["chunk_type"] == "DIRECT_COMMAND":
                continue

            if chunk["embedding"] is None:
                continue

            chunk_embedding = np.asarray(
                chunk["embedding"],
                dtype=np.float32,
            )

            similarity = float(
                np.dot(
                    query_embedding,
                    chunk_embedding,
                )
            )

            if similarity >= min_similarity:

                scored.append(
                    {
                        "id": chunk["id"],
                        "sequence": chunk["sequence"],
                        "text": chunk["text"],
                        "similarity": similarity,
                        "chunk_type": chunk[
                            "chunk_type"
                        ],
                    }
                )

        scored.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        # ------------------------------------------------------
        # Expand relevant results with neighboring transcript.
        # ------------------------------------------------------

        final_results: list[dict] = []

        seen_ids: set[int] = set()

        for result in scored[:limit]:

            neighbors = (
                self.database
                .get_neighboring_chunks(
                    session_id=self.session_id,
                    sequence=result["sequence"],
                    radius=neighbor_radius,
                )
            )

            for chunk in neighbors:

                chunk_id = chunk["id"]

                if chunk_id in seen_ids:
                    continue

                if (
                    chunk["chunk_type"]
                    == "DIRECT_COMMAND"
                ):

                    continue

                seen_ids.add(
                    chunk_id
                )

                final_results.append(
                    {
                        "id": chunk_id,
                        "sequence": chunk[
                            "sequence"
                        ],
                        "text": chunk[
                            "text"
                        ],
                        # For now we preserve the original
                        # result similarity.
                        #
                        # Later we'll give neighbors their
                        # own scores.
                        "similarity": result[
                            "similarity"
                        ],
                        "chunk_type": chunk[
                            "chunk_type"
                        ],
                    }
                )

        return final_results

    # ==========================================================
    # FULL TRANSCRIPT
    # ==========================================================

    def full_transcript(
        self,
    ) -> str:

        if self.session_id is None:
            return ""

        chunks = (
            self.database
            .get_session_chunks(
                session_id=self.session_id
            )
        )

        return "\n".join(
            chunk["text"]
            for chunk in chunks
        )

    # ==========================================================
    # PAUSE
    # ==========================================================

    def pause(
        self,
    ) -> None:

        if self.session_id is None:
            return

        self.database.update_session_state(
            session_id=self.session_id,
            state=SessionState.PAUSED.value,
        )

        self.state = SessionState.PAUSED

    # ==========================================================
    # RESUME
    # ==========================================================

    def resume(
        self,
    ) -> None:

        if self.session_id is None:
            return

        self.database.update_session_state(
            session_id=self.session_id,
            state=SessionState.ACTIVE.value,
        )

        self.state = SessionState.ACTIVE

    # ==========================================================
    # END
    # ==========================================================

    def end(
        self,
        summary: str | None = None,
    ) -> None:

        if self.session_id is None:
            return

        self.database.close_session(
            session_id=self.session_id,
            summary=summary,
        )

        print(
            f"[SESSION] Ended #{self.session_id}"
        )

        self.session_id = None
        self.sequence = 0

        self.chunk_timestamps.clear()

        self.section_start_time = None
        self.section_start_sequence = None
        self.section_number = 0

        self.state = SessionState.IDLE
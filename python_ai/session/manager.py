from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from memory.database import MemoryDatabase
from session.models import SessionState


class SessionManager:
    """
    Manages the current VEMORA session.

    Session data is temporary/current context.
    It is separate from long-term memory.
    """

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

        self.sequence += 1

        # Create semantic embedding.
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
            )
        )

        return chunk_id

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

        chunks = self.recent_chunks(limit)

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
                        "chunk_type": chunk["chunk_type"],
                    }
                )

        scored.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        # ------------------------------------------------------
        # Expand relevant results with neighboring transcript
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

                # Don't include direct commands as evidence.
                if chunk["chunk_type"] == "DIRECT_COMMAND":
                    continue

                seen_ids.add(chunk_id)

                final_results.append(
                    {
                        "id": chunk_id,
                        "sequence": chunk["sequence"],
                        "text": chunk["text"],
                        "similarity": result["similarity"],
                        "chunk_type": chunk[
                            "chunk_type"
                        ],
                    }
                )

        return final_results

    # ==========================================================
    # FULL TRANSCRIPT
    # ==========================================================

    def full_transcript(self) -> str:

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

    def pause(self) -> None:

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

    def resume(self) -> None:

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
        self.state = SessionState.IDLE
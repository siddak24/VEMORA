from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from memory.database import MemoryDatabase
from memory.retention import calculate_expiry


class MemoryManager:
    """
    High-level semantic memory manager for VEMORA.

    Responsibilities:
        - Generate embeddings
        - Save memories
        - Search memories semantically
        - Update memories
        - Delete memories
        - Remove expired memories

    The database layer handles SQLite.
    This class handles memory logic.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        user_id: str = "default_user",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:

        # --------------------------------------------------------
        # Database path
        # --------------------------------------------------------

        if db_path is None:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[2]
            )

            db_path = (
                project_root
                / "data"
                / "vemora.db"
            )

        self.user_id = user_id

        # --------------------------------------------------------
        # Embedding model
        # --------------------------------------------------------

        print(
            "[VEMORA] Loading embedding model..."
        )

        self.embedding_model = (
            SentenceTransformer(
                embedding_model
            )
        )

        print(
            "[VEMORA] Embedding model ready."
        )

        # --------------------------------------------------------
        # Database
        # --------------------------------------------------------

        self.database = MemoryDatabase(
            db_path=db_path
        )

    # ============================================================
    # EMBEDDING
    # ============================================================

    def _embed(
        self,
        text: str,
    ) -> list[float]:

        vector = self.embedding_model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    # ============================================================
    # SAVE
    # ============================================================

    def save(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        confidence: float = 0.5,
        retention: str = "SHORT_TERM",
    ) -> int:

        content = content.strip()

        if not content:
            raise ValueError(
                "Cannot save empty memory."
            )

        # Generate embedding.
        embedding = self._embed(
            content
        )

        # Calculate expiration.
        expiry = calculate_expiry(
            retention
        )

        expires_at = (
            expiry.isoformat()
            if expiry is not None
            else None
        )

        # Store in database.
        memory_id = (
            self.database.add_memory(
                user_id=self.user_id,
                content=content,
                memory_type=memory_type,
                embedding=embedding,
                importance=importance,
                confidence=confidence,
                retention=retention,
                expires_at=expires_at,
            )
        )

        return memory_id

    # ============================================================
    # SEMANTIC SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.30,
    ) -> list[dict]:

        query = query.strip()

        if not query:
            return []

        # --------------------------------------------------------
        # Remove expired memories first.
        # --------------------------------------------------------

        expired_count = (
            self.database
            .delete_expired_memories(
                user_id=self.user_id
            )
        )

        if expired_count > 0:

            print(
                f"[MEMORY] Removed "
                f"{expired_count} expired memory(s)."
            )

        # --------------------------------------------------------
        # Embed query.
        # --------------------------------------------------------

        query_embedding = np.asarray(
            self._embed(query),
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # Get stored memory embeddings.
        # --------------------------------------------------------

        memories = (
            self.database
            .get_memories_with_embeddings(
                user_id=self.user_id
            )
        )

        scored_results: list[dict] = []

        # --------------------------------------------------------
        # Similarity search.
        # --------------------------------------------------------

        for memory in memories:

            memory_embedding = np.asarray(
                memory["embedding"],
                dtype=np.float32,
            )

            similarity = float(
                np.dot(
                    query_embedding,
                    memory_embedding,
                )
            )

            if similarity >= min_similarity:

                memory["similarity"] = similarity

                # Do not return the actual vector.
                memory.pop(
                    "embedding",
                    None,
                )

                scored_results.append(
                    memory
                )

        # --------------------------------------------------------
        # Highest similarity first.
        # --------------------------------------------------------

        scored_results.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        return scored_results[:limit]

    # ============================================================
    # UPDATE
    # ============================================================

    def update(
        self,
        memory_id: int,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        confidence: float = 0.5,
        retention: str = "SHORT_TERM",
    ) -> bool:

        content = content.strip()

        if not content:
            raise ValueError(
                "Cannot update memory with empty content."
            )

        # New content needs a new embedding.
        embedding = self._embed(
            content
        )

        # Recalculate retention.
        expiry = calculate_expiry(
            retention
        )

        expires_at = (
            expiry.isoformat()
            if expiry is not None
            else None
        )

        return self.database.update_memory(
            memory_id=memory_id,
            user_id=self.user_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            retention=retention,
            expires_at=expires_at,
        )

    # ============================================================
    # DELETE
    # ============================================================

    def delete(
        self,
        memory_id: int,
    ) -> bool:

        return self.database.delete_memory(
            memory_id=memory_id,
            user_id=self.user_id,
        )

    # ============================================================
    # GET ALL
    # ============================================================

    def all(self) -> list[dict]:

        # Clean expired memories before displaying them.
        self.database.delete_expired_memories(
            user_id=self.user_id
        )

        return (
            self.database
            .get_all_memories(
                user_id=self.user_id
            )
        )

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self) -> None:

        self.database.close()
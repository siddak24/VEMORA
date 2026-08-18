from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from memory.database import MemoryDatabase


class MemoryManager:
    """
    VEMORA semantic memory manager.

    Responsibilities:

        1. Generate embeddings for memories.
        2. Store embeddings.
        3. Generate embeddings for queries.
        4. Perform similarity search.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        user_id: str = "default_user",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:

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

        self.database = MemoryDatabase(
            db_path=db_path
        )

        self._ensure_embedding_column()

    # ============================================================
    # DATABASE MIGRATION
    # ============================================================

    def _ensure_embedding_column(self) -> None:

        cursor = self.database.connection.execute(
            "PRAGMA table_info(memories)"
        )

        columns = [
            row["name"]
            for row in cursor.fetchall()
        ]

        if "embedding" not in columns:

            print(
                "[VEMORA] Adding embedding column "
                "to existing memory database..."
            )

            self.database.connection.execute(
                """
                ALTER TABLE memories
                ADD COLUMN embedding TEXT
                """
            )

            self.database.connection.commit()

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
    ) -> int:

        embedding = self._embed(
            content
        )

        memory_id = (
            self.database.add_memory(
                user_id=self.user_id,
                content=content,
                memory_type=memory_type,
                embedding=embedding,
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

        query_embedding = np.asarray(
            self._embed(query),
            dtype=np.float32,
        )

        memories = (
            self.database
            .get_memories_with_embeddings(
                user_id=self.user_id
            )
        )

        scored_results: list[dict] = []

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

                # Don't expose the large vector to callers.
                memory.pop(
                    "embedding",
                    None,
                )

                scored_results.append(
                    memory
                )

        scored_results.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        return scored_results[:limit]

    # ============================================================
    # ALL MEMORIES
    # ============================================================

    def all(self) -> list[dict]:

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
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class MemoryDatabase:
    """
    SQLite storage for VEMORA memories.

    V0.2:
        Stores both memory text and its embedding.

    Later:
        This can be replaced with PostgreSQL + pgvector.
    """

    def __init__(
        self,
        db_path: str | Path,
    ) -> None:

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.db_path
        )

        self.connection.row_factory = sqlite3.Row

        self._create_tables()

    def _create_tables(self) -> None:

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id TEXT NOT NULL,

                content TEXT NOT NULL,

                memory_type TEXT NOT NULL DEFAULT 'general',

                embedding TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.connection.commit()

    def add_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str,
        embedding: list[float],
    ) -> int:

        embedding_json = json.dumps(
            embedding
        )

        cursor = self.connection.execute(
            """
            INSERT INTO memories (
                user_id,
                content,
                memory_type,
                embedding
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                content,
                memory_type,
                embedding_json,
            ),
        )

        self.connection.commit()

        return int(cursor.lastrowid)

    def get_memories_with_embeddings(
        self,
        user_id: str,
    ) -> list[dict]:

        cursor = self.connection.execute(
            """
            SELECT
                id,
                content,
                memory_type,
                embedding,
                created_at,
                updated_at
            FROM memories
            WHERE user_id = ?
              AND embedding IS NOT NULL
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )

        results = []

        for row in cursor.fetchall():

            item = dict(row)

            item["embedding"] = json.loads(
                item["embedding"]
            )

            results.append(item)

        return results

    def get_all_memories(
        self,
        user_id: str,
    ) -> list[dict]:

        cursor = self.connection.execute(
            """
            SELECT
                id,
                content,
                memory_type,
                created_at,
                updated_at
            FROM memories
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        )

        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    def close(self) -> None:

        self.connection.close()
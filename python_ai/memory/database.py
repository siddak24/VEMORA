from __future__ import annotations

import sqlite3
from pathlib import Path


class MemoryDatabase:
    """
    SQLite storage for VEMORA memories.

    V0.1:
        One local user.

    Later:
        user_id will isolate memories for different users.
    """

    def __init__(
        self,
        db_path: str | Path = "data/vemora.db",
    ) -> None:

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.db_path
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        self._create_tables()

    def _create_tables(self) -> None:

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id TEXT NOT NULL,

                content TEXT NOT NULL,

                memory_type TEXT NOT NULL DEFAULT 'general',

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
        memory_type: str = "general",
    ) -> int:

        cursor = self.connection.execute(
            """
            INSERT INTO memories (
                user_id,
                content,
                memory_type
            )
            VALUES (?, ?, ?)
            """,
            (
                user_id,
                content,
                memory_type,
            ),
        )

        self.connection.commit()

        return int(cursor.lastrowid)

    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        # Simple keyword search for V0.1.
        #
        # Later we will replace this with embeddings/vector search.

        words = [
            word.strip()
            for word in query.lower().split()
            if len(word.strip()) >= 3
        ]

        if not words:
            return []

        conditions = []
        parameters: list[str | int] = [
            user_id
        ]

        for word in words:
            conditions.append(
                "LOWER(content) LIKE ?"
            )
            parameters.append(
                f"%{word}%"
            )

        where_clause = " OR ".join(
            conditions
        )

        parameters.append(limit)

        cursor = self.connection.execute(
            f"""
            SELECT
                id,
                content,
                memory_type,
                created_at,
                updated_at
            FROM memories
            WHERE user_id = ?
              AND ({where_clause})
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            parameters,
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

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
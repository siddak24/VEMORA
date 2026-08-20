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
        self._create_session_tables()

    def create_session(
        self,
        user_id: str,
        session_type: str = "conversation",
    ) -> int:

        cursor = self.connection.execute(
            """
            INSERT INTO sessions (
                user_id,
                session_type,
                state
            )
            VALUES (?, ?, ?)
            """,
            (
                user_id,
                session_type,
                "ACTIVE",
            ),
        )

        self.connection.commit()

        return int(cursor.lastrowid)
    
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

    def update_session_state(
        self,
        session_id: int,
        state: str,
    ) -> None:

        self.connection.execute(
            """
            UPDATE sessions
            SET state = ?
            WHERE id = ?
            """,
            (
                state,
                session_id,
            ),
        )

        self.connection.commit()

    def get_recent_session_chunks(
        self,
        session_id: int,
        limit: int = 10,
    ) -> list[dict]:

        cursor = self.connection.execute(
            """
            SELECT
                id,
                session_id,
                sequence,
                text,
                created_at
            FROM transcript_chunks
            WHERE session_id = ?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (
                session_id,
                limit,
            ),
        )

        rows = cursor.fetchall()

        rows.reverse()

        return [
            dict(row)
            for row in rows
        ]
        
    def get_session_chunks(
        self,
        session_id: int,
    ) -> list[dict]:

        cursor = self.connection.execute(
            """
            SELECT
                id,
                session_id,
                sequence,
                text,
                created_at
            FROM transcript_chunks
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        )

        return [
            dict(row)
            for row in cursor.fetchall()
        ]
    
    def add_transcript_chunk(
        self,
        session_id: int,
        sequence: int,
        text: str,
    ) -> int:

        cursor = self.connection.execute(
            """
            INSERT INTO transcript_chunks (
                session_id,
                sequence,
                text
            )
            VALUES (?, ?, ?)
            """,
            (
                session_id,
                sequence,
                text,
            ),
        )

        self.connection.commit()

        return int(cursor.lastrowid)

    def close_session(
        self,
        session_id: int,
        summary: str | None = None,
    ) -> None:

        self.connection.execute(
            """
            UPDATE sessions
            SET
                state = ?,
                ended_at = CURRENT_TIMESTAMP,
                summary = ?
            WHERE id = ?
            """,
            (
                "FINALIZING",
                summary,
                session_id,
            ),
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
    def _create_session_tables(self) -> None:

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id TEXT NOT NULL,

                session_type TEXT NOT NULL DEFAULT 'conversation',

                state TEXT NOT NULL DEFAULT 'IDLE',

                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                ended_at TIMESTAMP,

                summary TEXT
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transcript_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id INTEGER NOT NULL,

                sequence INTEGER NOT NULL,

                text TEXT NOT NULL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
            )
            """
        )

        self.connection.commit()

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
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class MemoryDatabase:
    """
    SQLite storage for VEMORA.

    Handles:
        - long-term memories
        - sessions
        - session transcript chunks

    Current prototype:
        SQLite + JSON embeddings

    Future:
        PostgreSQL + pgvector
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

        # --------------------------------------------------------
        # Create tables
        # --------------------------------------------------------

        self._create_tables()
        self._create_session_tables()

        # --------------------------------------------------------
        # Migrate existing databases
        # --------------------------------------------------------

        self._ensure_memory_columns()
        self._ensure_transcript_columns()

    # ============================================================
    # LONG-TERM MEMORY TABLE
    # ============================================================

    def _create_tables(self) -> None:

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id TEXT NOT NULL,

                content TEXT NOT NULL,

                memory_type TEXT NOT NULL
                    DEFAULT 'general',

                embedding TEXT,

                importance REAL NOT NULL
                    DEFAULT 0.5,

                confidence REAL NOT NULL
                    DEFAULT 0.5,

                retention TEXT NOT NULL
                    DEFAULT 'SHORT_TERM',

                expires_at TIMESTAMP,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                updated_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                last_accessed TIMESTAMP
            )
            """
        )

        self.connection.commit()

    # ============================================================
    # MEMORY MIGRATION
    # ============================================================

    def _ensure_memory_columns(self) -> None:

        cursor = self.connection.execute(
            "PRAGMA table_info(memories)"
        )

        columns = {
            row["name"]
            for row in cursor.fetchall()
        }

        migrations = {
            "importance": """
                ALTER TABLE memories
                ADD COLUMN importance REAL DEFAULT 0.5
            """,

            "confidence": """
                ALTER TABLE memories
                ADD COLUMN confidence REAL DEFAULT 0.5
            """,

            "retention": """
                ALTER TABLE memories
                ADD COLUMN retention TEXT
                DEFAULT 'SHORT_TERM'
            """,

            "expires_at": """
                ALTER TABLE memories
                ADD COLUMN expires_at TIMESTAMP
            """,

            "last_accessed": """
                ALTER TABLE memories
                ADD COLUMN last_accessed TIMESTAMP
            """,
        }

        for column, sql in migrations.items():

            if column not in columns:

                print(
                    f"[VEMORA] Adding memory column: "
                    f"{column}"
                )

                self.connection.execute(sql)

        self.connection.commit()

    # ============================================================
    # ADD MEMORY
    # ============================================================

    def add_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str,
        embedding: list[float],
        importance: float,
        confidence: float,
        retention: str,
        expires_at: str | None,
    ) -> int:

        cursor = self.connection.execute(
            """
            INSERT INTO memories (
                user_id,
                content,
                memory_type,
                embedding,
                importance,
                confidence,
                retention,
                expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                content,
                memory_type,
                json.dumps(embedding),
                importance,
                confidence,
                retention,
                expires_at,
            ),
        )

        self.connection.commit()

        return int(cursor.lastrowid)

    # ============================================================
    # UPDATE MEMORY
    # ============================================================

    def update_memory(
        self,
        memory_id: int,
        user_id: str,
        content: str,
        embedding: list[float],
        memory_type: str,
        importance: float,
        confidence: float,
        retention: str,
        expires_at: str | None,
    ) -> bool:

        cursor = self.connection.execute(
            """
            UPDATE memories
            SET
                content = ?,
                embedding = ?,
                memory_type = ?,
                importance = ?,
                confidence = ?,
                retention = ?,
                expires_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND user_id = ?
            """,
            (
                content,
                json.dumps(embedding),
                memory_type,
                importance,
                confidence,
                retention,
                expires_at,
                memory_id,
                user_id,
            ),
        )

        self.connection.commit()

        return cursor.rowcount > 0

    # ============================================================
    # DELETE MEMORY
    # ============================================================

    def delete_memory(
        self,
        memory_id: int,
        user_id: str,
    ) -> bool:

        cursor = self.connection.execute(
            """
            DELETE FROM memories
            WHERE id = ?
              AND user_id = ?
            """,
            (
                memory_id,
                user_id,
            ),
        )

        self.connection.commit()

        return cursor.rowcount > 0

    # ============================================================
    # DELETE EXPIRED MEMORIES
    # ============================================================

    def delete_expired_memories(
        self,
        user_id: str,
    ) -> int:

        cursor = self.connection.execute(
            """
            DELETE FROM memories
            WHERE user_id = ?
              AND expires_at IS NOT NULL
              AND datetime(expires_at)
                  <= datetime('now')
            """,
            (user_id,),
        )

        self.connection.commit()

        return cursor.rowcount

    # ============================================================
    # GET MEMORIES WITH EMBEDDINGS
    # ============================================================

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
                importance,
                confidence,
                retention,
                expires_at,
                created_at,
                updated_at,
                last_accessed
            FROM memories
            WHERE user_id = ?
              AND embedding IS NOT NULL
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )

        results: list[dict] = []

        for row in cursor.fetchall():

            item = dict(row)

            item["embedding"] = json.loads(
                item["embedding"]
            )

            results.append(item)

        return results

    # ============================================================
    # GET ALL MEMORIES
    # ============================================================

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
                importance,
                confidence,
                retention,
                expires_at,
                created_at,
                updated_at,
                last_accessed
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

    # ============================================================
    # SESSION TABLES
    # ============================================================

    def _create_session_tables(self) -> None:

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id TEXT NOT NULL,

                session_type TEXT NOT NULL
                    DEFAULT 'conversation',

                state TEXT NOT NULL
                    DEFAULT 'IDLE',

                started_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

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

                embedding TEXT,

                chunk_type TEXT NOT NULL
                    DEFAULT 'PASSIVE',

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
            )
            """
        )

        self.connection.commit()

    # ============================================================
    # TRANSCRIPT MIGRATION
    # ============================================================

    def _ensure_transcript_columns(self) -> None:

        cursor = self.connection.execute(
            "PRAGMA table_info(transcript_chunks)"
        )

        columns = {
            row["name"]
            for row in cursor.fetchall()
        }

        migrations = {
            "embedding": """
                ALTER TABLE transcript_chunks
                ADD COLUMN embedding TEXT
            """,

            "chunk_type": """
                ALTER TABLE transcript_chunks
                ADD COLUMN chunk_type TEXT
                DEFAULT 'PASSIVE'
            """,
        }

        for column, sql in migrations.items():

            if column not in columns:

                print(
                    f"[VEMORA] Adding transcript column: "
                    f"{column}"
                )

                self.connection.execute(sql)

        self.connection.commit()

    # ============================================================
    # CREATE SESSION
    # ============================================================

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

    # ============================================================
    # UPDATE SESSION STATE
    # ============================================================

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

    # ============================================================
    # ADD TRANSCRIPT CHUNK
    # ============================================================

    def add_transcript_chunk(
        self,
        session_id: int,
        sequence: int,
        text: str,
        embedding: list[float] | None = None,
        chunk_type: str = "PASSIVE",
    ) -> int:

        embedding_json = (
            json.dumps(embedding)
            if embedding is not None
            else None
        )

        cursor = self.connection.execute(
            """
            INSERT INTO transcript_chunks (
                session_id,
                sequence,
                text,
                embedding,
                chunk_type
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                sequence,
                text,
                embedding_json,
                chunk_type,
            ),
        )

        self.connection.commit()

        return int(cursor.lastrowid)

    # ============================================================
    # GET RECENT SESSION CHUNKS
    # ============================================================

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
                embedding,
                chunk_type,
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

        results = []

        for row in rows:

            item = dict(row)

            if item["embedding"] is not None:

                item["embedding"] = json.loads(
                    item["embedding"]
                )

            results.append(item)

        return results

    # ============================================================
    # GET ALL SESSION CHUNKS
    # ============================================================

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
                embedding,
                chunk_type,
                created_at
            FROM transcript_chunks
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        )

        results = []

        for row in cursor.fetchall():

            item = dict(row)

            if item["embedding"] is not None:

                item["embedding"] = json.loads(
                    item["embedding"]
                )

            results.append(item)

        return results

    # ============================================================
    # GET SESSION CHUNKS WITH EMBEDDINGS
    # ============================================================

    def get_session_chunks_with_embeddings(
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
                embedding,
                chunk_type,
                created_at
            FROM transcript_chunks
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        )

        results = []

        for row in cursor.fetchall():

            item = dict(row)

            if item["embedding"] is not None:

                item["embedding"] = json.loads(
                    item["embedding"]
                )

            results.append(item)

        return results

    # ============================================================
    # GET NEIGHBORING CHUNKS
    # ============================================================

    def get_neighboring_chunks(
        self,
        session_id: int,
        sequence: int,
        radius: int = 1,
    ) -> list[dict]:

        cursor = self.connection.execute(
            """
            SELECT
                id,
                session_id,
                sequence,
                text,
                embedding,
                chunk_type,
                created_at
            FROM transcript_chunks
            WHERE session_id = ?
              AND sequence BETWEEN ? AND ?
            ORDER BY sequence ASC
            """,
            (
                session_id,
                max(1, sequence - radius),
                sequence + radius,
            ),
        )

        results = []

        for row in cursor.fetchall():

            item = dict(row)

            if item["embedding"] is not None:

                item["embedding"] = json.loads(
                    item["embedding"]
                )

            results.append(item)

        return results

    # ============================================================
    # CLOSE SESSION
    # ============================================================

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

    # ============================================================
    # CLOSE DATABASE
    # ============================================================

    def close(self) -> None:

        self.connection.close()
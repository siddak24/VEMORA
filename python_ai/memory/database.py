from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class MemoryDatabase:
    """
    SQLite storage for VEMORA.

    Handles:
        - long-term memories
        - tasks
        - sessions
        - transcript chunks
        - hierarchical session summaries

    Current prototype:
        SQLite + JSON embeddings

    Future:
        PostgreSQL + pgvector
    """

    def __init__(
        self,
        db_path: str | Path,
    ) -> None:

        self._lock = threading.RLock()

        self.db_path = Path(
            db_path
        )

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        # ------------------------------------------------------
        # Create tables
        # ------------------------------------------------------

        self._create_tables()
        self._create_session_tables()

        # ------------------------------------------------------
        # Migrations
        # ------------------------------------------------------

        self._ensure_memory_columns()
        self._ensure_transcript_columns()
        self._ensure_session_summary_table()

    # ==========================================================
    # COMMON HELPERS
    # ==========================================================

    def _commit(self) -> None:

        with self._lock:

            self.connection.commit()

    # ==========================================================
    # CORE TABLES
    # ==========================================================

    def _create_tables(
        self,
    ) -> None:

        # ------------------------------------------------------
        # TASKS
        # ------------------------------------------------------

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id TEXT NOT NULL,

                title TEXT NOT NULL,

                description TEXT,

                due_at TIMESTAMP,

                status TEXT NOT NULL
                    DEFAULT 'PENDING',

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                updated_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                completed_at TIMESTAMP,

                expires_at TIMESTAMP
            )
            """
        )

        # ------------------------------------------------------
        # LONG-TERM MEMORY
        # ------------------------------------------------------

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

        self._commit()

    # ==========================================================
    # SESSION TABLES
    # ==========================================================

    def _create_session_tables(
        self,
    ) -> None:

        # ------------------------------------------------------
        # SESSIONS
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # TRANSCRIPT CHUNKS
        # ------------------------------------------------------

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

                recorded_at TEXT,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
            )
            """
        )

        # ------------------------------------------------------
        # HIERARCHICAL SESSION SUMMARIES
        # ------------------------------------------------------

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id INTEGER NOT NULL,

                summary_type TEXT NOT NULL,

                start_sequence INTEGER,

                end_sequence INTEGER,

                start_time TEXT,

                end_time TEXT,

                content TEXT NOT NULL,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
            )
            """
        )

        self._commit()

    # ==========================================================
    # SESSION SUMMARY MIGRATION
    # ==========================================================

    def _ensure_session_summary_table(
        self,
    ) -> None:
        """
        Ensure the session_summaries table exists in older
        VEMORA databases.
        """

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id INTEGER NOT NULL,

                summary_type TEXT NOT NULL,

                start_sequence INTEGER,

                end_sequence INTEGER,

                start_time TEXT,

                end_time TEXT,

                content TEXT NOT NULL,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
            )
            """
        )

        self._commit()

    # ==========================================================
    # SESSION SUMMARY INSERT
    # ==========================================================

    def add_session_summary(
        self,
        session_id: int,
        summary_type: str,
        start_sequence: int | None,
        end_sequence: int | None,
        start_time: str | None,
        end_time: str | None,
        content: str,
    ) -> int:

        with self._lock:

            cursor = self.connection.execute(
                """
                INSERT INTO session_summaries (
                    session_id,
                    summary_type,
                    start_sequence,
                    end_sequence,
                    start_time,
                    end_time,
                    content
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    summary_type,
                    start_sequence,
                    end_sequence,
                    start_time,
                    end_time,
                    content,
                ),
            )

            self.connection.commit()

            return int(
                cursor.lastrowid
            )

    # ==========================================================
    # GET SESSION SUMMARIES
    # ==========================================================

    def get_session_summaries(
        self,
        session_id: int,
        summary_type: str | None = None,
    ) -> list[dict]:

        with self._lock:

            if summary_type is None:

                cursor = self.connection.execute(
                    """
                    SELECT *
                    FROM session_summaries
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (
                        session_id,
                    ),
                )

            else:

                cursor = self.connection.execute(
                    """
                    SELECT *
                    FROM session_summaries
                    WHERE session_id = ?
                      AND summary_type = ?
                    ORDER BY id ASC
                    """,
                    (
                        session_id,
                        summary_type,
                    ),
                )

            return [
                dict(row)
                for row in cursor.fetchall()
            ]

    # ==========================================================
    # GET LATEST SESSION SUMMARY
    # ==========================================================

    def get_latest_session_summary(
        self,
        session_id: int,
    ) -> dict | None:

        with self._lock:

            cursor = self.connection.execute(
                """
                SELECT *
                FROM session_summaries
                WHERE session_id = ?
                AND summary_type = 'SESSION'
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    session_id,
                ),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            return dict(row)


    # ==========================================================
    # GET SECTION SUMMARIES
    # ==========================================================

    def get_section_summaries(
        self,
        session_id: int,
    ) -> list[dict]:

        with self._lock:

            cursor = self.connection.execute(
                """
                SELECT *
                FROM session_summaries
                WHERE session_id = ?
                AND summary_type = 'SECTION'
                ORDER BY
                    CASE
                        WHEN start_time IS NULL THEN 1
                        ELSE 0
                    END,
                    start_time ASC,
                    id ASC
                """,
                (
                    session_id,
                ),
            )

            return [
                dict(row)
                for row in cursor.fetchall()
            ]

    # ==========================================================
    # TASKS
    # ==========================================================

    def get_tasks(
        self,
        user_id: str,
        status: str | None = None,
    ) -> list[dict]:

        with self._lock:

            if status is None:

                cursor = self.connection.execute(
                    """
                    SELECT *
                    FROM tasks
                    WHERE user_id = ?
                    ORDER BY
                        CASE
                            WHEN due_at IS NULL THEN 1
                            ELSE 0
                        END,
                        due_at ASC
                    """,
                    (
                        user_id,
                    ),
                )

            else:

                cursor = self.connection.execute(
                    """
                    SELECT *
                    FROM tasks
                    WHERE user_id = ?
                      AND status = ?
                    ORDER BY due_at ASC
                    """,
                    (
                        user_id,
                        status,
                    ),
                )

            return [
                dict(row)
                for row in cursor.fetchall()
            ]

    # ==========================================================
    # CREATE TASK
    # ==========================================================

    def create_task(
        self,
        user_id: str,
        title: str,
        description: str | None,
        due_at: str | None,
        expires_at: str | None,
    ) -> int:

        with self._lock:

            cursor = self.connection.execute(
                """
                INSERT INTO tasks (
                    user_id,
                    title,
                    description,
                    due_at,
                    expires_at,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    title,
                    description,
                    due_at,
                    expires_at,
                    "PENDING",
                ),
            )

            self.connection.commit()

            return int(
                cursor.lastrowid
            )

    # ==========================================================
    # COMPLETE TASK
    # ==========================================================

    def complete_task(
        self,
        task_id: int,
        user_id: str,
    ) -> bool:

        with self._lock:

            cursor = self.connection.execute(
                """
                UPDATE tasks
                SET
                    status = 'COMPLETED',
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND user_id = ?
                """,
                (
                    task_id,
                    user_id,
                ),
            )

            self.connection.commit()

            return cursor.rowcount > 0

    # ==========================================================
    # DELETE TASK
    # ==========================================================

    def delete_task(
        self,
        task_id: int,
        user_id: str,
    ) -> bool:

        with self._lock:

            cursor = self.connection.execute(
                """
                DELETE FROM tasks
                WHERE id = ?
                  AND user_id = ?
                """,
                (
                    task_id,
                    user_id,
                ),
            )

            self.connection.commit()

            return cursor.rowcount > 0

    # ==========================================================
    # DELETE EXPIRED TASKS
    # ==========================================================

    def delete_expired_tasks(
        self,
        user_id: str,
    ) -> int:

        with self._lock:

            cursor = self.connection.execute(
                """
                DELETE FROM tasks
                WHERE user_id = ?
                  AND expires_at IS NOT NULL
                  AND datetime(expires_at)
                      <= datetime('now')
                """,
                (
                    user_id,
                ),
            )

            self.connection.commit()

            return cursor.rowcount

    # ==========================================================
    # MEMORY MIGRATION
    # ==========================================================

    def _ensure_memory_columns(
        self,
    ) -> None:

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
                ADD COLUMN importance REAL
                    DEFAULT 0.5
            """,

            "confidence": """
                ALTER TABLE memories
                ADD COLUMN confidence REAL
                    DEFAULT 0.5
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
                    "[VEMORA] Adding memory column: "
                    f"{column}"
                )

                self.connection.execute(
                    sql
                )

        self._commit()

    # ==========================================================
    # ADD MEMORY
    # ==========================================================

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

        with self._lock:

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

            return int(
                cursor.lastrowid
            )

    # ==========================================================
    # UPDATE MEMORY
    # ==========================================================

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

        with self._lock:

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

    # ==========================================================
    # DELETE MEMORY
    # ==========================================================

    def delete_memory(
        self,
        memory_id: int,
        user_id: str,
    ) -> bool:

        with self._lock:

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

    # ==========================================================
    # DELETE EXPIRED MEMORIES
    # ==========================================================

    def delete_expired_memories(
        self,
        user_id: str,
    ) -> int:

        with self._lock:

            cursor = self.connection.execute(
                """
                DELETE FROM memories
                WHERE user_id = ?
                  AND expires_at IS NOT NULL
                  AND datetime(expires_at)
                      <= datetime('now')
                """,
                (
                    user_id,
                ),
            )

            self.connection.commit()

            return cursor.rowcount

    # ==========================================================
    # GET MEMORIES WITH EMBEDDINGS
    # ==========================================================

    def get_memories_with_embeddings(
        self,
        user_id: str,
    ) -> list[dict]:

        with self._lock:

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
                (
                    user_id,
                ),
            )

            results: list[dict] = []

            for row in cursor.fetchall():

                item = dict(row)

                item["embedding"] = json.loads(
                    item["embedding"]
                )

                results.append(item)

            return results

    # ==========================================================
    # GET ALL MEMORIES
    # ==========================================================

    def get_all_memories(
        self,
        user_id: str,
    ) -> list[dict]:

        with self._lock:

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
                ORDER BY created_at DESC
                """,
                (
                    user_id,
                ),
            )

            return [
                dict(row)
                for row in cursor.fetchall()
            ]

    # ==========================================================
    # TRANSCRIPT MIGRATION
    # ==========================================================

    def _ensure_transcript_columns(
        self,
    ) -> None:

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

            "recorded_at": """
                ALTER TABLE transcript_chunks
                ADD COLUMN recorded_at TEXT
            """,
        }

        for column, sql in migrations.items():

            if column not in columns:

                print(
                    "[VEMORA] Adding transcript column: "
                    f"{column}"
                )

                self.connection.execute(
                    sql
                )

        # ------------------------------------------------------
        # Backfill old transcript timestamps.
        # ------------------------------------------------------

        if "recorded_at" not in columns:

            self.connection.execute(
                """
                UPDATE transcript_chunks
                SET recorded_at = created_at
                WHERE recorded_at IS NULL
                """
            )

        self._commit()

    # ==========================================================
    # CREATE SESSION
    # ==========================================================

    def create_session(
        self,
        user_id: str,
        session_type: str = "conversation",
    ) -> int:

        with self._lock:

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

            return int(
                cursor.lastrowid
            )

    # ==========================================================
    # UPDATE SESSION STATE
    # ==========================================================

    def update_session_state(
        self,
        session_id: int,
        state: str,
    ) -> None:

        with self._lock:

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

    # ==========================================================
    # ADD TRANSCRIPT CHUNK
    # ==========================================================

    def add_transcript_chunk(
        self,
        session_id: int,
        sequence: int,
        text: str,
        embedding: list[float] | None = None,
        chunk_type: str = "PASSIVE",
        recorded_at: str | None = None,
    ) -> int:

        embedding_json = (
            json.dumps(embedding)
            if embedding is not None
            else None
        )

        with self._lock:

            cursor = self.connection.execute(
                """
                INSERT INTO transcript_chunks (
                    session_id,
                    sequence,
                    text,
                    embedding,
                    chunk_type,
                    recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    sequence,
                    text,
                    embedding_json,
                    chunk_type,
                    recorded_at,
                ),
            )

            self.connection.commit()

            return int(
                cursor.lastrowid
            )

    # ==========================================================
    # GET RECENT SESSION CHUNKS
    # ==========================================================

    def get_recent_session_chunks(
        self,
        session_id: int,
        limit: int = 10,
    ) -> list[dict]:

        with self._lock:

            cursor = self.connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    sequence,
                    text,
                    embedding,
                    chunk_type,
                    recorded_at,
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

            results: list[dict] = []

            for row in rows:

                item = dict(row)

                if item["embedding"] is not None:

                    item["embedding"] = json.loads(
                        item["embedding"]
                    )

                results.append(item)

            return results

    # ==========================================================
    # GET TRANSCRIPT CHUNKS IN TIME RANGE
    # ==========================================================

    def get_session_chunks_in_time_range(
        self,
        session_id: int,
        start_time: str,
        end_time: str,
    ) -> list[dict]:
        """
        Return transcript chunks whose recorded_at timestamp
        falls inside the requested time window.

        Boundaries are inclusive.
        """

        with self._lock:

            cursor = self.connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    sequence,
                    text,
                    embedding,
                    chunk_type,
                    recorded_at,
                    created_at
                FROM transcript_chunks
                WHERE session_id = ?
                  AND recorded_at IS NOT NULL
                  AND recorded_at >= ?
                  AND recorded_at <= ?
                ORDER BY sequence ASC
                """,
                (
                    session_id,
                    start_time,
                    end_time,
                ),
            )

            results: list[dict] = []

            for row in cursor.fetchall():

                item = dict(row)

                if item["embedding"] is not None:

                    item["embedding"] = json.loads(
                        item["embedding"]
                    )

                results.append(item)

            return results

    # ==========================================================
    # GET ALL SESSION CHUNKS
    # ==========================================================

    def get_session_chunks(
        self,
        session_id: int,
    ) -> list[dict]:

        with self._lock:

            cursor = self.connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    sequence,
                    text,
                    embedding,
                    chunk_type,
                    recorded_at,
                    created_at
                FROM transcript_chunks
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (
                    session_id,
                ),
            )

            results: list[dict] = []

            for row in cursor.fetchall():

                item = dict(row)

                if item["embedding"] is not None:

                    item["embedding"] = json.loads(
                        item["embedding"]
                    )

                results.append(item)

            return results

    # ==========================================================
    # GET SESSION CHUNKS WITH EMBEDDINGS
    # ==========================================================

    def get_session_chunks_with_embeddings(
        self,
        session_id: int,
    ) -> list[dict]:

        with self._lock:

            cursor = self.connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    sequence,
                    text,
                    embedding,
                    chunk_type,
                    recorded_at,
                    created_at
                FROM transcript_chunks
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (
                    session_id,
                ),
            )

            results: list[dict] = []

            for row in cursor.fetchall():

                item = dict(row)

                if item["embedding"] is not None:

                    item["embedding"] = json.loads(
                        item["embedding"]
                    )

                results.append(item)

            return results

    # ==========================================================
    # GET NEIGHBORING CHUNKS
    # ==========================================================

    def get_neighboring_chunks(
        self,
        session_id: int,
        sequence: int,
        radius: int = 1,
    ) -> list[dict]:

        with self._lock:

            cursor = self.connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    sequence,
                    text,
                    embedding,
                    chunk_type,
                    recorded_at,
                    created_at
                FROM transcript_chunks
                WHERE session_id = ?
                  AND sequence BETWEEN ? AND ?
                ORDER BY sequence ASC
                """,
                (
                    session_id,
                    max(
                        1,
                        sequence - radius,
                    ),
                    sequence + radius,
                ),
            )

            results: list[dict] = []

            for row in cursor.fetchall():

                item = dict(row)

                if item["embedding"] is not None:

                    item["embedding"] = json.loads(
                        item["embedding"]
                    )

                results.append(item)

            return results

    # ==========================================================
    # CLOSE SESSION
    # ==========================================================

    def close_session(
        self,
        session_id: int,
        summary: str | None = None,
    ) -> None:

        with self._lock:

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

    # ==========================================================
    # CLOSE DATABASE
    # ==========================================================

    def close(self) -> None:

        with self._lock:

            self.connection.close()
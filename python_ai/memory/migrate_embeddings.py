from __future__ import annotations

from pathlib import Path

from memory.manager import MemoryManager


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    manager = MemoryManager(
        db_path=(
            project_root
            / "data"
            / "vemora.db"
        )
    )

    cursor = manager.database.connection.execute(
        """
        SELECT id, content
        FROM memories
        WHERE embedding IS NULL
        """
    )

    rows = cursor.fetchall()

    print(
        f"[VEMORA] Memories requiring embeddings: "
        f"{len(rows)}"
    )

    for row in rows:

        memory_id = row["id"]
        content = row["content"]

        embedding = manager._embed(
            content
        )

        import json

        manager.database.connection.execute(
            """
            UPDATE memories
            SET embedding = ?
            WHERE id = ?
            """,
            (
                json.dumps(
                    embedding
                ),
                memory_id,
            ),
        )

    manager.database.connection.commit()

    print(
        "[VEMORA] Embedding migration complete."
    )

    manager.close()


if __name__ == "__main__":
    main()
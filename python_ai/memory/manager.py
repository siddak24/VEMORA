from __future__ import annotations

from pathlib import Path

from memory.database import MemoryDatabase


class MemoryManager:

    def __init__(
        self,
        db_path: str | Path | None = None,
        user_id: str = "default_user",
    ) -> None:

        if db_path is None:
            project_root = Path(__file__).resolve().parents[2]
            db_path = project_root / "data" / "vemora.db"

        self.user_id = user_id

        self.database = MemoryDatabase(
            db_path=db_path
        )

    def save(
        self,
        content: str,
        memory_type: str = "general",
    ) -> int:

        return self.database.add_memory(
            user_id=self.user_id,
            content=content,
            memory_type=memory_type,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        return self.database.search_memories(
            user_id=self.user_id,
            query=query,
            limit=limit,
        )

    def all(self) -> list[dict]:

        return self.database.get_all_memories(
            user_id=self.user_id
        )

    def close(self) -> None:

        self.database.close()
from __future__ import annotations
from datetime import datetime, timedelta
from memory.database import MemoryDatabase


class TaskManager:
    """
    Lightweight task manager for VEMORA.

    Tasks are intentionally simple:
        - title
        - description
        - due time
        - status
        - expiration
    """

    def __init__(
        self,
        database: MemoryDatabase,
        user_id: str = "default_user",
    ) -> None:

        self.database = database
        self.user_id = user_id

    # ==========================================================
    # CREATE
    # ==========================================================

    def create(
        self,
        title: str,
        description: str | None = None,
        due_at: str | None = None,
        expires_at: str | None = None,
    ) -> int:

        title = title.strip()

        if not title:
            raise ValueError(
                "Task title cannot be empty."
            )

        # If due_at is already represented as an ISO
        # datetime and expires_at wasn't provided,
        # automatically retain the task for one extra hour.
        if due_at and not expires_at:

            try:
                due_datetime = datetime.fromisoformat(
                    due_at
                )

                expires_at = (
                    due_datetime
                    + timedelta(hours=1)
                ).isoformat()

            except ValueError:
                # Natural-language dates such as "Friday"
                # are left unchanged for now.
                pass

        return self.database.create_task(
            user_id=self.user_id,
            title=title,
            description=(
                description.strip()
                if description
                else None
            ),
            due_at=due_at,
            expires_at=expires_at,
        )
    def find_duplicate(
        self,
        title: str,
    ) -> dict | None:

        normalized_title = (
            title.strip().lower()
        )

        if not normalized_title:
            return None

        tasks = self.database.get_tasks(
            user_id=self.user_id,
            status="PENDING",
        )

        for task in tasks:

            existing_title = (
                task["title"]
                .strip()
                .lower()
            )

            if existing_title == normalized_title:

                return task

        return None

    # ==========================================================
    # SEARCH
    # ==========================================================

    def search(
        
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        self.database.delete_expired_tasks(
            user_id=self.user_id
        )
        query = query.strip()
        # ----------------------------------------------------------
        # "all tasks" / "my tasks" / "what tasks do I have"
        # means return active tasks instead of keyword searching.
        # ----------------------------------------------------------

        normalized = query.lower()

        all_task_queries = {
            "",
            "all",
            "all tasks",
            "my tasks",
            "tasks",
            "current tasks",
            "pending tasks",
            "what tasks do i have",
            "what are my tasks",
            "show my tasks",
            "list my tasks",
        }

        if normalized in all_task_queries:

            tasks = self.database.get_tasks(
                user_id=self.user_id,
                status="PENDING",
            )

            return tasks[:limit]

        # ----------------------------------------------------------
        # Normal keyword search
        # ----------------------------------------------------------

        tasks = self.database.get_tasks(
            user_id=self.user_id,
        )

        query_words = {
            word.lower()
            for word in query.split()
            if len(word) >= 2
        }

        scored: list[dict] = []

        for task in tasks:

            searchable_text = " ".join(
                filter(
                    None,
                    [
                        task.get("title"),
                        task.get("description"),
                    ],
                )
            ).lower()

            text_words = set(
                searchable_text.split()
            )

            score = len(
                query_words & text_words
            )

            if score > 0:

                task["score"] = score

                scored.append(task)

        scored.sort(
            key=lambda item: (
                -item["score"],
                item.get("due_at") or "",
            )
        )

        return scored[:limit]
    # ==========================================================
    # ALL
    # ==========================================================

    def all(
        self,
        status: str | None = None,
    ) -> list[dict]:

        self.database.delete_expired_tasks(
            user_id=self.user_id
        )

        return self.database.get_tasks(
            user_id=self.user_id,
            status=status,
        )

    # ==========================================================
    # COMPLETE
    # ==========================================================

    def complete(
        self,
        task_id: int,
    ) -> bool:

        return self.database.complete_task(
            task_id=task_id,
            user_id=self.user_id,
        )

    # ==========================================================
    # DELETE
    # ==========================================================

    def delete(
        self,
        task_id: int,
    ) -> bool:

        return self.database.delete_task(
            task_id=task_id,
            user_id=self.user_id,
        )
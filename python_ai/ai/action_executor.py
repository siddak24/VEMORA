from __future__ import annotations

from ai.action_models import ActionPlan
from memory.manager import MemoryManager
from session.manager import SessionManager
from task.manager import TaskManager
from datetime import datetime

class ActionExecutor:
    """
    Executes structured actions produced by the SLM/LLM.

    The model decides WHAT to do.
    Python decides HOW to do it.
    """

    def __init__(
        self,
        memory: MemoryManager,
        session: SessionManager,
        tasks: TaskManager,
    ) -> None:

        self.memory = memory
        self.session = session
        self.tasks = tasks

    # ==========================================================
    # EXECUTE PLAN
    # ==========================================================

    def execute(
        self,
        plan: ActionPlan,
        allowed_tools: set[str] | None = None,
    ) -> list[dict]:

        if allowed_tools is None:

            allowed_tools = {
                "search_session",
                "search_memory",
                "save_memory",
                "update_memory",
                "delete_memory",
                "create_task",
                "search_task",
                "complete_task",
                "delete_task",
                "get_current_time",
                "get_current_date",
            }

        results: list[dict] = []

        for action in plan.actions:

            # --------------------------------------------------
            # SECURITY / MODE RESTRICTION
            # --------------------------------------------------

            if action.tool not in allowed_tools:

                results.append(
                    {
                        "tool": action.tool,
                        "status": "blocked",
                        "reason": "tool_not_allowed",
                    }
                )

                continue

            # ==================================================
            # SESSION SEARCH
            # ==================================================

            if action.tool == "search_session":

                matches = self.session.search(
                    query=action.query,
                    limit=10,
                )

                results.append(
                    {
                        "tool": "search_session",
                        "results": matches,
                    }
                )

            # ==================================================
            # CURRENT TIME
            # ==================================================

            elif action.tool == "get_current_time":

                now = datetime.now().astimezone()

                results.append(
                    {
                        "tool": "get_current_time",
                        "time": now.strftime("%I:%M %p"),
                        "timezone": now.strftime("%Z"),
                        "status": "success",
                    }
                )


            # ==================================================
            # CURRENT DATE
            # ==================================================

            elif action.tool == "get_current_date":

                now = datetime.now().astimezone()

                results.append(
                    {
                        "tool": "get_current_date",
                        "date": now.strftime("%A, %d %B %Y"),
                        "status": "success",
                    }
                )
            
            # ==================================================
            # LONG-TERM MEMORY SEARCH
            # ==================================================

            elif action.tool == "search_memory":

                matches = self.memory.search(
                    query=action.query,
                    limit=5,
                )

                results.append(
                    {
                        "tool": "search_memory",
                        "results": matches,
                    }
                )

            # ==================================================
            # SAVE MEMORY
            # ==================================================

            elif action.tool == "save_memory":

                if not action.content.strip():

                    results.append(
                        {
                            "tool": "save_memory",
                            "status": "failed",
                            "reason": "empty_content",
                        }
                    )

                    continue

                memory_id = self.memory.save(
                    content=action.content,
                    memory_type=action.memory_type,
                    importance=action.importance,
                    confidence=action.confidence,
                    retention=action.retention,
                )

                results.append(
                    {
                        "tool": "save_memory",
                        "memory_id": memory_id,
                        "status": "saved",
                    }
                )

            # ==================================================
            # UPDATE MEMORY
            # ==================================================

            elif action.tool == "update_memory":

                matches = self.memory.search(
                    query=action.query,
                    limit=3,
                )

                if not matches:

                    results.append(
                        {
                            "tool": "update_memory",
                            "status": "not_found",
                        }
                    )

                    continue

                target = matches[0]

                updated = self.memory.update(
                    memory_id=(
                        action.memory_id
                        or target["id"]
                    ),
                    content=action.content,
                    memory_type=action.memory_type,
                    importance=action.importance,
                    confidence=action.confidence,
                    retention=action.retention,
                )

                results.append(
                    {
                        "tool": "update_memory",
                        "status": (
                            "updated"
                            if updated
                            else "failed"
                        ),
                        "memory_id": target["id"],
                    }
                )

            # ==================================================
            # DELETE MEMORY
            # ==================================================

            elif action.tool == "delete_memory":

                matches = self.memory.search(
                    query=action.query,
                    limit=3,
                )

                if not matches:

                    results.append(
                        {
                            "tool": "delete_memory",
                            "status": "not_found",
                        }
                    )

                    continue

                # Avoid blindly deleting an ambiguous result.
                if (
                    len(matches) > 1
                    and (
                        matches[0]["similarity"]
                        - matches[1]["similarity"]
                        < 0.10
                    )
                ):

                    results.append(
                        {
                            "tool": "delete_memory",
                            "status": "ambiguous",
                            "matches": matches,
                        }
                    )

                    continue

                target = matches[0]

                deleted = self.memory.delete(
                    target["id"]
                )

                results.append(
                    {
                        "tool": "delete_memory",
                        "status": (
                            "deleted"
                            if deleted
                            else "failed"
                        ),
                        "memory_id": target["id"],
                    }
                )

            # ==================================================
            # CREATE TASK
            # ==================================================

            elif action.tool == "create_task":

                title = (
                    action.title.strip()
                    or action.content.strip()
                )

                if not title:

                    results.append(
                        {
                            "tool": "create_task",
                            "status": "failed",
                            "reason": "empty_title",
                        }
                    )

                    continue

                # --------------------------------------------------
                # Prevent duplicate pending tasks.
                # --------------------------------------------------

                existing = self.tasks.find_duplicate(
                    title=title
                )

                if existing is not None:

                    results.append(
                        {
                            "tool": "create_task",
                            "status": "already_exists",
                            "task_id": existing["id"],
                            "title": existing["title"],
                        }
                    )

                    continue

                # --------------------------------------------------
                # Create new task.
                # --------------------------------------------------

                task_id = self.tasks.create(
                    title=title,
                    description=(
                        action.description.strip()
                        or None
                    ),
                    due_at=(
                        action.due_at.strip()
                        or None
                    ),
                    expires_at=(
                        action.expires_at.strip()
                        or None
                    ),
                )

                results.append(
                    {
                        "tool": "create_task",
                        "task_id": task_id,
                        "status": "created",
                    }
                )
            # ==================================================
            # SEARCH TASK
            # ==================================================

            elif action.tool == "search_task":

                matches = self.tasks.search(
                    query=action.query,
                    limit=5,
                )

                results.append(
                    {
                        "tool": "search_task",
                        "results": matches,
                    }
                )

            # ==================================================
            # COMPLETE TASK
            # ==================================================

            elif action.tool == "complete_task":

                task_id = action.task_id

                # If the model did not provide an ID,
                # search for the relevant task first.
                if task_id is None:

                    matches = self.tasks.search(
                        query=action.query,
                        limit=3,
                    )

                    if not matches:

                        results.append(
                            {
                                "tool": "complete_task",
                                "status": "not_found",
                            }
                        )

                        continue

                    if (
                        len(matches) > 1
                        and (
                            matches[0]["score"]
                            == matches[1]["score"]
                        )
                    ):

                        results.append(
                            {
                                "tool": "complete_task",
                                "status": "ambiguous",
                                "matches": matches,
                            }
                        )

                        continue

                    task_id = matches[0]["id"]

                completed = self.tasks.complete(
                    task_id
                )

                results.append(
                    {
                        "tool": "complete_task",
                        "task_id": task_id,
                        "status": (
                            "completed"
                            if completed
                            else "failed"
                        ),
                    }
                )

            # ==================================================
            # DELETE TASK
            # ==================================================

            elif action.tool == "delete_task":

                task_id = action.task_id

                if task_id is None:

                    matches = self.tasks.search(
                        query=action.query,
                        limit=3,
                    )

                    if not matches:

                        results.append(
                            {
                                "tool": "delete_task",
                                "status": "not_found",
                            }
                        )

                        continue

                    if (
                        len(matches) > 1
                        and (
                            matches[0]["score"]
                            == matches[1]["score"]
                        )
                    ):

                        results.append(
                            {
                                "tool": "delete_task",
                                "status": "ambiguous",
                                "matches": matches,
                            }
                        )

                        continue

                    task_id = matches[0]["id"]

                deleted = self.tasks.delete(
                    task_id
                )

                results.append(
                    {
                        "tool": "delete_task",
                        "task_id": task_id,
                        "status": (
                            "deleted"
                            if deleted
                            else "failed"
                        ),
                    }
                )

        return results
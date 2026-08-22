from __future__ import annotations

from memory.manager import MemoryManager
from session.manager import SessionManager
from ai.action_models import ActionPlan


class ActionExecutor:
    """
    Executes the structured plan produced by the SLM/LLM.

    The model decides WHAT to do.
    This class decides HOW to actually do it.
    """

    def __init__(
        self,
        memory: MemoryManager,
        session: SessionManager,
    ) -> None:

        self.memory = memory
        self.session = session

    def execute(
        self,
        plan: ActionPlan,
    ) -> list[dict]:

        results: list[dict] = []

        for action in plan.actions:

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
                    memory_id=target["id"],
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

        return results
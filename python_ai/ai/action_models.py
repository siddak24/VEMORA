from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    One tool action requested by the SLM/LLM.
    """

    tool: Literal[
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
    ]

    # Used by search/update/delete operations.
    query: str = ""

    # Used for memory content.
    content: str = ""

    # Used when creating a task.
    title: str = ""

    description: str = ""

    # Memory fields.
    memory_type: str = "general"

    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    retention: Literal[
        "EPHEMERAL",
        "SHORT_TERM",
        "EVENT_BASED",
        "LONG_TERM",
    ] = "SHORT_TERM"

    # Task fields.
    due_at: str = ""

    expires_at: str = ""

    status: Literal[
        "PENDING",
        "COMPLETED",
        "CANCELLED",
    ] = "PENDING"

    # Optional explicit IDs.
    memory_id: int | None = None
    task_id: int | None = None


class ActionPlan(BaseModel):
    """
    Structured plan produced by the SLM/LLM.
    """

    actions: list[Action] = Field(
        default_factory=list
    )

    should_speak: bool = True

    response_instruction: str = (
        "Answer the user concisely using "
        "the available evidence."
    )
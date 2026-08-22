from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Action(BaseModel):
    tool: Literal[
        "search_session",
        "search_memory",
        "save_memory",
        "update_memory",
        "delete_memory",
    ]

    query: str = ""

    content: str = ""

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


class ActionPlan(BaseModel):
    """
    Plan produced by the SLM/LLM.

    The model decides what tools are needed.
    Python executes them.
    """

    actions: list[Action] = Field(
        default_factory=list
    )

    should_speak: bool = True

    response_instruction: str = Field(
        default=(
            "Answer the user concisely using "
            "the available evidence."
        )
    )
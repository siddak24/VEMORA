from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MemoryDecision(BaseModel):
    action: Literal[
        "CHAT",
        "SAVE_MEMORY",
        "SEARCH_MEMORY",
    ] = Field(
        description="What VEMORA should do with the user's request."
    )

    content: str = Field(
        default="",
        description=(
            "Information to save when action is SAVE_MEMORY. "
            "Otherwise empty."
        ),
    )

    query: str = Field(
        default="",
        description=(
            "What to search for when action is SEARCH_MEMORY. "
            "Otherwise empty."
        ),
    )

    memory_type: str = Field(
        default="general",
        description=(
            "Category of memory, such as preference, "
            "person, event, fact, or general."
        ),
    )
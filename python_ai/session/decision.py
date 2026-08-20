from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionDecision(BaseModel):
    action: Literal[
        "LISTEN",
        "RESPOND",
        "SAVE_MEMORY",
        "SEARCH_MEMORY",
    ] = Field(
        description=(
            "What VEMORA should do with the current "
            "utterance during an active listening session."
        )
    )

    should_speak: bool = Field(
        description=(
            "Whether VEMORA should verbally respond "
            "to the user right now."
        )
    )

    content: str = Field(
        default="",
        description=(
            "Information to save if SAVE_MEMORY is selected."
        ),
    )

    query: str = Field(
        default="",
        description=(
            "Search query if SEARCH_MEMORY is selected."
        ),
    )

    memory_type: str = Field(
        default="general",
        description="Type of memory to save.",
    )
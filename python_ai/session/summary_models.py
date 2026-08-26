from __future__ import annotations

from pydantic import BaseModel, Field


class SectionSummary(BaseModel):
    """
    Summary representing one section of an active session.
    """

    summary: str = ""

    key_points: list[str] = Field(
        default_factory=list
    )

    entities: list[str] = Field(
        default_factory=list
    )

    important_events: list[str] = Field(
        default_factory=list
    )


class SessionSummary(BaseModel):
    """
    High-level summary representing an entire session.
    """

    summary: str = ""

    key_points: list[str] = Field(
        default_factory=list
    )

    important_events: list[str] = Field(
        default_factory=list
    )

    people: list[str] = Field(
        default_factory=list
    )

    topics: list[str] = Field(
        default_factory=list
    )
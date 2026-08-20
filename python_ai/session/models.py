from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SessionState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    FINALIZING = "FINALIZING"


@dataclass
class Session:
    session_id: int
    user_id: str
    session_type: str
    state: SessionState
    started_at: str
    ended_at: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class TranscriptChunk:
    chunk_id: int
    session_id: int
    sequence: int
    text: str
    created_at: str
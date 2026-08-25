from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

from session.manager import SessionManager


@dataclass
class SessionProcessor:
    """
    Controls how transcript chunks from an active session
    are handled.

    Passive speech is buffered locally.

    Direct VEMORA commands are handled immediately by the
    caller.

    The processor itself does not call Gemini. It only decides
    when a passive batch is ready for higher-level processing.
    """

    session: SessionManager

    # Number of passive chunks to accumulate before suggesting
    # background processing.
    batch_size: int = 5

    # Maximum amount of time between passive processing events.
    batch_interval_seconds: float = 20.0

    passive_buffer: list[str] = field(
        default_factory=list
    )

    last_batch_time: float = field(
        default_factory=monotonic
    )

    # ==========================================================
    # ADD PASSIVE CHUNK
    # ==========================================================

    def add_passive_chunk(
        self,
        text: str,
    ) -> None:

        text = text.strip()

        if not text:
            return

        self.passive_buffer.append(
            text
        )

        # Store in the session as PASSIVE.
        self.session.add_transcript(
            text,
            chunk_type="PASSIVE",
        )


    def reset(self) -> None:
        """
        Clear any pending passive chunks and restart
        the processing timer.
        """

        self.passive_buffer.clear()
        self.last_batch_time = monotonic()

    # ==========================================================
    # CHECK WHETHER BACKGROUND PROCESSING IS DUE
    # ==========================================================


    
    def should_process_passive(
        self,
    ) -> bool:

        if not self.passive_buffer:
            return False

        if len(self.passive_buffer) >= self.batch_size:
            return True

        elapsed = (
            monotonic()
            - self.last_batch_time
        )

        return (
            elapsed
            >= self.batch_interval_seconds
        )

    # ==========================================================
    # GET CURRENT PASSIVE BATCH
    # ==========================================================

    def get_passive_batch(self) -> str:

        if not self.passive_buffer:
            return ""

        batch = "\n".join(
            self.passive_buffer
        )

        self.passive_buffer.clear()

        self.last_batch_time = monotonic()

        return batch

    # ==========================================================
    # PENDING DATA
    # ==========================================================

    def has_pending_passive_data(self) -> bool:

        return bool(
            self.passive_buffer
        )

    # ==========================================================
    # FLUSH
    # ==========================================================

    def flush(self) -> str:

        return self.get_passive_batch()
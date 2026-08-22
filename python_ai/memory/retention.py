from __future__ import annotations

from datetime import datetime, timedelta


def calculate_expiry(
    retention: str,
) -> datetime | None:

    now = datetime.now()

    if retention == "EPHEMERAL":
        return now + timedelta(hours=2)

    if retention == "SHORT_TERM":
        return now + timedelta(days=7)

    if retention == "EVENT_BASED":
        # For now, the event itself will determine
        # the real expiry later.
        return None

    if retention == "LONG_TERM":
        return None

    return now + timedelta(days=7)
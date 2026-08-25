from __future__ import annotations

import threading
import time

from task.manager import TaskManager


class TaskScheduler:
    """
    Background task-expiry checker.

    Checks every `interval_seconds` and removes
    tasks whose expires_at has passed.
    """

    def __init__(
        self,
        tasks: TaskManager,
        interval_seconds: int = 30,
    ) -> None:

        self.tasks = tasks
        self.interval_seconds = interval_seconds

        self._stop_event = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

    def start(self) -> None:

        if not self._thread.is_alive():

            print(
                "[TASK SCHEDULER] Started."
            )

            self._thread.start()

    def stop(self) -> None:

        self._stop_event.set()

        if self._thread.is_alive():

            self._thread.join(
                timeout=2
            )

        print(
            "[TASK SCHEDULER] Stopped."
        )

    def _run(self) -> None:

        while not self._stop_event.wait(
            self.interval_seconds
        ):

            try:

                deleted = (
                    self.tasks.database
                    .delete_expired_tasks(
                        user_id=self.tasks.user_id
                    )
                )

                if deleted > 0:

                    print(
                        f"[TASK SCHEDULER] "
                        f"Deleted {deleted} expired task(s)."
                    )

            except Exception as error:

                print(
                    f"[TASK SCHEDULER] ERROR: {error}"
                )
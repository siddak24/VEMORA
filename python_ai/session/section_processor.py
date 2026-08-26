from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

from memory.database import MemoryDatabase
from session.section_manager import (
    SessionSection,
    SessionSectionManager,
)
from session.summarizer import SessionSummarizer


class SectionProcessor:
    """
    Automatically summarizes completed session sections.

    Responsibilities:
        - Determine when a section is complete.
        - Capture its transcript.
        - Summarize it in a background worker.
        - Save section summaries to SQLite.
        - Generate a final session summary when the
          session ends gracefully.

    The normal listening loop is not blocked while a
    section is being summarized.
    """

    def __init__(
        self,
        database: MemoryDatabase,
        section_manager: SessionSectionManager,
        summarizer: SessionSummarizer,
    ) -> None:

        self.database = database
        self.section_manager = section_manager
        self.summarizer = summarizer

        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vemora-summary",
        )

        self.session_id: int | None = None

        self._initialized = False
        self._summary_lock = threading.RLock()

        # ------------------------------------------------------
        # Track sections that have already been submitted.
        # ------------------------------------------------------

        self.last_submitted_section_number = 0

        # ------------------------------------------------------
        # Track background summary jobs.
        #
        # Because max_workers=1, at most one summary actually
        # runs at a time, while later jobs may wait in the queue.
        # ------------------------------------------------------

        self._pending_futures: list[Future] = []

    # ==========================================================
    # START SESSION
    # ==========================================================

    def start(
        self,
        session_id: int,
        start_time: datetime,
    ) -> None:

        with self._summary_lock:

            # Safety: don't start a new section processor while
            # an old one is still active.
            if self._initialized:

                raise RuntimeError(
                    "SectionProcessor is already active."
                )

            self.session_id = session_id

            self.section_manager.reset()

            self.section_manager.create_first_section(
                start_time
            )

            self.last_submitted_section_number = 0

            self._pending_futures.clear()

            self._initialized = True

            print(
                "[SECTION] "
                f"Started {self.section_manager.current_section.label}"
            )

    # ==========================================================
    # CHECK FOR COMPLETED SECTION
    # ==========================================================

    def check(self) -> None:
        """
        Lightweight check called by the main listening loop.

        If the current section has ended, its transcript is
        submitted to the background summarizer and the next
        overlapping section immediately becomes active.
        """

        with self._summary_lock:

            if not self._initialized:
                return

            if self.session_id is None:
                return

            current = (
                self.section_manager.current_section
            )

            if current is None:
                return

            # --------------------------------------------------
            # Current time in local timezone.
            # --------------------------------------------------

            now = datetime.now(
                timezone.utc
            ).astimezone()

            if now < current.end_time:
                return

            # --------------------------------------------------
            # Prevent duplicate submission.
            # --------------------------------------------------

            if (
                current.number
                <= self.last_submitted_section_number
            ):
                return

            session_id = self.session_id

            # --------------------------------------------------
            # Capture the section transcript BEFORE moving
            # to the next overlapping section.
            # --------------------------------------------------

            section_text = (
                self.section_manager.get_section_text(
                    session_id=session_id,
                    section=current,
                )
            )

            self.last_submitted_section_number = (
                current.number
            )

            # --------------------------------------------------
            # Submit background summarization.
            # --------------------------------------------------

            if section_text.strip():

                print()
                print(
                    "[SECTION] "
                    f"{current.label} is complete."
                )

                print(
                    "[SECTION] "
                    "Starting background summarization..."
                )

                future = self.executor.submit(
                    self._summarize_and_save,
                    session_id,
                    current,
                    section_text,
                )

                self._pending_futures.append(
                    future
                )

                # Remove completed futures so this list does
                # not grow forever during a long session.
                self._cleanup_finished_futures()

            else:

                print()
                print(
                    "[SECTION] "
                    f"{current.label} contains no usable speech."
                )

            # --------------------------------------------------
            # Immediately create the next overlapping section.
            # --------------------------------------------------

            next_section = (
                self.section_manager.create_next_section()
            )

            print(
                "[SECTION] "
                f"Now collecting {next_section.label}"
            )

    # ==========================================================
    # CLEAN FINISHED FUTURES
    # ==========================================================

    def _cleanup_finished_futures(
        self,
    ) -> None:

        self._pending_futures = [
            future
            for future in self._pending_futures
            if not future.done()
        ]

    # ==========================================================
    # BACKGROUND SECTION SUMMARIZATION
    # ==========================================================

    def _summarize_and_save(
        self,
        session_id: int,
        section: SessionSection,
        section_text: str,
    ) -> None:

        try:

            summary = (
                self.summarizer.summarize(
                    section_text
                )
            )

            content = (
                summary.model_dump_json()
            )

            # --------------------------------------------------
            # Save the exact sequence range too.
            # This will be useful when we later retrieve
            # section summaries alongside raw transcript data.
            # --------------------------------------------------

            section_chunks = (
                self.section_manager.get_section_chunks(
                    session_id=session_id,
                    section=section,
                )
            )

            if section_chunks:

                start_sequence = min(
                    chunk["sequence"]
                    for chunk in section_chunks
                )

                end_sequence = max(
                    chunk["sequence"]
                    for chunk in section_chunks
                )

            else:

                start_sequence = None
                end_sequence = None

            summary_id = (
                self.database.add_session_summary(
                    session_id=session_id,
                    summary_type="SECTION",
                    start_sequence=start_sequence,
                    end_sequence=end_sequence,
                    start_time=(
                        section.start_time.isoformat()
                    ),
                    end_time=(
                        section.end_time.isoformat()
                    ),
                    content=content,
                )
            )

            print()
            print(
                "[SECTION] "
                f"Saved summary #{summary_id}: "
                f"{section.label}"
            )

        except Exception as error:

            print()
            print(
                "[SECTION] "
                "Background summarization failed: "
                f"{error}"
            )

    # ==========================================================
    # WAIT FOR BACKGROUND SECTION SUMMARIES
    # ==========================================================

    def wait_for_pending_summaries(
        self,
    ) -> None:
        """
        Gracefully wait for all section-summary jobs that
        have already been submitted.

        This is used when the user intentionally ends a session.
        """

        with self._summary_lock:

            self._cleanup_finished_futures()

            if not self._pending_futures:

                return

            print()
            print(
                "[SECTION] "
                "Waiting for pending section summaries..."
            )

            futures = list(
                self._pending_futures
            )

        # Wait outside the lock so the worker can continue.
        for future in futures:

            try:

                future.result()

            except Exception as error:

                print(
                    "[SECTION] "
                    f"Pending summary failed: {error}"
                )

        with self._summary_lock:

            self._pending_futures.clear()

    # ==========================================================
    # FINAL SESSION SUMMARY
    # ==========================================================

    def finalize_session_summary(
        self,
    ) -> int | None:
        """
        Finalize the current session.

        Before creating the overall SESSION summary:

            1. Capture the current unfinished section, if any.
            2. Submit it for background summarization.
            3. Wait for all section summaries to finish.
            4. Combine all SECTION summaries into one SESSION summary.
            5. Save the SESSION summary.

        Returns:
            Database ID of the SESSION summary,
            or None if there is nothing to summarize.
        """

        with self._summary_lock:

            if self.session_id is None:
                return None

            session_id = self.session_id

            current = (
                self.section_manager.current_section
            )

            # ------------------------------------------------------
            # FINAL PARTIAL SECTION
            # ------------------------------------------------------
            #
            # If the current section has not naturally reached
            # its end, we still need to preserve everything that
            # happened in it before generating the final summary.
            # ------------------------------------------------------

            if current is not None:

                # Don't submit it again if it has already been
                # submitted as a completed section.
                if (
                    current.number
                    > self.last_submitted_section_number
                ):

                    section_text = (
                        self.section_manager.get_section_text(
                            session_id=session_id,
                            section=current,
                        )
                    )

                    if section_text.strip():

                        print()
                        print(
                            "[SECTION] "
                            f"Final partial section detected: "
                            f"{current.label}"
                        )

                        print(
                            "[SECTION] "
                            "Submitting final partial section "
                            "for summarization..."
                        )

                        future = (
                            self.executor.submit(
                                self._summarize_and_save,
                                session_id,
                                current,
                                section_text,
                            )
                        )

                        self._pending_futures.append(
                            future
                        )

                        self.last_submitted_section_number = (
                            current.number
                        )

            # ------------------------------------------------------
            # WAIT FOR ALL SECTION SUMMARIES
            # ------------------------------------------------------

        self.wait_for_pending_summaries()

        # ----------------------------------------------------------
        # GET ALL SECTION SUMMARIES
        # ----------------------------------------------------------

        section_rows = (
            self.database.get_session_summaries(
                session_id=session_id,
                summary_type="SECTION",
            )
        )

        if not section_rows:

            print()
            print(
                "[SESSION SUMMARY] "
                "No section summaries available."
            )

            return None

        # ----------------------------------------------------------
        # Sort sections chronologically
        # ----------------------------------------------------------

        section_rows.sort(
            key=lambda row: (
                row.get("start_time")
                or "",
                row.get("id", 0),
            )
        )

        section_summaries = [
            row["content"]
            for row in section_rows
            if row.get("content")
        ]

        if not section_summaries:

            print()
            print(
                "[SESSION SUMMARY] "
                "No usable section summaries available."
            )

            return None

        # ----------------------------------------------------------
        # GENERATE OVERALL SESSION SUMMARY
        # ----------------------------------------------------------

        print()
        print(
            "[SESSION SUMMARY] "
            "Generating final session summary..."
        )

        final_summary = (
            self.summarizer.summarize_session(
                section_summaries
            )
        )

        content = (
            final_summary.model_dump_json()
        )

        # ----------------------------------------------------------
        # DETERMINE OVERALL TIME RANGE
        # ----------------------------------------------------------

        start_time = (
            section_rows[0].get(
                "start_time"
            )
        )

        end_time = (
            section_rows[-1].get(
                "end_time"
            )
        )

        start_sequences = [
            row["start_sequence"]
            for row in section_rows
            if row.get("start_sequence")
            is not None
        ]

        end_sequences = [
            row["end_sequence"]
            for row in section_rows
            if row.get("end_sequence")
            is not None
        ]

        start_sequence = (
            min(start_sequences)
            if start_sequences
            else None
        )

        end_sequence = (
            max(end_sequences)
            if end_sequences
            else None
        )

        # ----------------------------------------------------------
        # SAVE FINAL SESSION SUMMARY
        # ----------------------------------------------------------

        summary_id = (
            self.database.add_session_summary(
                session_id=session_id,
                summary_type="SESSION",
                start_sequence=start_sequence,
                end_sequence=end_sequence,
                start_time=start_time,
                end_time=end_time,
                content=content,
            )
        )

        print()
        print(
            "[SESSION SUMMARY] "
            f"Saved final session summary #{summary_id}"
        )

        return summary_id
    # ==========================================================
    # GRACEFUL STOP
    # ==========================================================

    def stop(
        self,
        finalize: bool = False,
    ) -> int | None:
        """
        Stop the section processor.

        finalize=True:
            Used when the user intentionally ends the session.
            Waits for pending section summaries and generates
            a final SESSION summary.

        finalize=False:
            Used for emergency/application shutdown.
            Does not wait for Gemini jobs.
        """

        with self._summary_lock:

            if not self._initialized:

                return None

        final_summary_id: int | None = None

        # ------------------------------------------------------
        # Normal/user-requested shutdown.
        # ------------------------------------------------------

        if finalize:

            final_summary_id = (
                self.finalize_session_summary()
            )

            print()
            print(
                "[SECTION] "
                "Section processor finalized."
            )

            # After all submitted work has completed,
            # now it is safe to shut down the executor.
            self.executor.shutdown(
                wait=True,
                cancel_futures=False,
            )

        # ------------------------------------------------------
        # Emergency shutdown.
        # ------------------------------------------------------

        else:

            print()
            print(
                "[SECTION] "
                "Stopping section processor..."
            )

            self.executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

        # ------------------------------------------------------
        # Reset state.
        # ------------------------------------------------------

        with self._summary_lock:

            self.session_id = None

            self._initialized = False

            self.last_submitted_section_number = 0

            self._pending_futures.clear()

            self.section_manager.reset()

        return final_summary_id
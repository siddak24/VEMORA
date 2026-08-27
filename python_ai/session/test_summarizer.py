from __future__ import annotations
from pathlib import Path
from ai.providers import create_llm_provider
from memory.database import MemoryDatabase
from session.section_manager import SessionSectionManager
from session.summarizer import SessionSummarizer


def main() -> None:

    print()
    print(
        "========================================"
    )
    print(
        "      VEMORA SECTION SUMMARIZER TEST"
    )
    print(
        "========================================"
    )
    print()

    # ----------------------------------------------------------
    # AI
    # ----------------------------------------------------------

    llm = create_llm_provider()

    if not hasattr(
        llm,
        "client",
    ):

        raise RuntimeError(
            "This test requires the Gemini provider."
        )

    # ----------------------------------------------------------
    # DATABASE
    # ----------------------------------------------------------

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    database = MemoryDatabase(
        db_path=(
            project_root
            / "data"
            / "vemora.db"
        )
    )

    # ----------------------------------------------------------
    # SECTION MANAGER
    # ----------------------------------------------------------

    section_manager = (
        SessionSectionManager(
            database=database,
        )
    )

    # ----------------------------------------------------------
    # USE EXISTING TEST SESSION
    # ----------------------------------------------------------

    session_id = 28

    # Your existing session #28 is useful because it
    # already contains transcript data.
    sections = []

    # Pick a section starting from the earliest timestamp
    # that exists in the session.
    chunks = database.get_session_chunks(
        session_id=session_id
    )

    if not chunks:

        print(
            "No transcript chunks found."
        )

        database.close()

        return

    first_timestamp = (
        chunks[0]["recorded_at"]
    )

    from datetime import datetime

    start = datetime.fromisoformat(
        first_timestamp
    )

    first_section = (
        section_manager.create_first_section(
            start
        )
    )

    sections.append(
        first_section
    )

    print(
        first_section.label
    )

    print()

    # ----------------------------------------------------------
    # GET SECTION CONTENT
    # ----------------------------------------------------------

    section_text = (
        section_manager.get_section_text(
            session_id=session_id,
            section=first_section,
        )
    )

    if not section_text.strip():

        print(
            "The section contains no transcript."
        )

        database.close()

        return

    print(
        "[SECTION TEXT]"
    )

    print(
        section_text[:3000]
    )

    if len(section_text) > 3000:

        print(
            "... [truncated for display]"
        )

    print()

    # ----------------------------------------------------------
    # SUMMARIZE
    # ----------------------------------------------------------

    summarizer = SessionSummarizer(
        llm=llm
    )

    print(
        "[VEMORA] Summarizing section..."
    )

    summary = (
        summarizer.summarize(
            section_text
        )
    )

    print()

    print(
        "[SECTION SUMMARY]"
    )

    print(
        summary.model_dump_json(
            indent=2
        )
    )

    # ----------------------------------------------------------
    # SAVE SUMMARY
    # ----------------------------------------------------------

    summary_text = (
        summary.model_dump_json()
    )

    summary_id = (
        database.add_session_summary(
            session_id=session_id,
            summary_type="SECTION",
            start_sequence=(
                chunks[0]["sequence"]
            ),
            end_sequence=(
                chunks[-1]["sequence"]
            ),
            start_time=(
                first_section
                .start_time
                .isoformat()
            ),
            end_time=(
                first_section
                .end_time
                .isoformat()
            ),
            content=summary_text,
        )
    )

    print()

    print(
        f"[VEMORA] Saved section summary "
        f"#{summary_id}"
    )

    database.close()


if __name__ == "__main__":
    main()
from memory.database import MemoryDatabase
from session.manager import SessionManager


def main() -> None:

    database = MemoryDatabase(
        "data/vemora.db"
    )

    session = SessionManager(
        database=database,
        user_id="default_user",
    )

    print()
    print("==============================")
    print("    VEMORA SESSION TEST")
    print("==============================")
    print()

    session.start(
        session_type="seminar"
    )

    session.add_transcript(
        "Today we will discuss distributed systems."
    )

    session.add_transcript(
        "The final project deadline is September 15."
    )

    session.add_transcript(
        "We will also discuss database scalability."
    )

    print()
    print("Recent context:")
    print("------------------------------")

    print(
        session.recent_context()
    )

    print()
    print("Full transcript:")
    print("------------------------------")

    print(
        session.full_transcript()
    )

    session.end(
        summary=(
            "Seminar about distributed systems "
            "and database scalability."
        )
    )

    database.close()

    print()
    print("Session test complete.")


if __name__ == "__main__":
    main()
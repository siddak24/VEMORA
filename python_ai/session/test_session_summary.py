from ai.providers import create_llm_provider
from session.summarizer import SessionSummarizer


def main() -> None:

    llm = create_llm_provider()

    summarizer = SessionSummarizer(
        llm=llm
    )

    section_summaries = [
        """
        John enters a village and meets Sarah.
        Sarah tells John about a mysterious shop.
        """,

        """
        John visits the shop with Sarah.
        The shopkeeper performs strange magic tricks.
        Several unusual objects appear.
        """,

        """
        John and Sarah leave the shop.
        When they look back, the shop has disappeared.
        """,
    ]

    print()
    print(
        "[VEMORA] Creating session summary..."
    )

    summary = (
        summarizer.summarize_session(
            section_summaries
        )
    )

    print()
    print(
        "[SESSION SUMMARY]"
    )

    print(
        summary.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()
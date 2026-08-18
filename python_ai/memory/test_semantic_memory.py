from memory.manager import MemoryManager


def main() -> None:

    memory = MemoryManager()

    print()
    print("====================================")
    print("     VEMORA SEMANTIC MEMORY TEST")
    print("====================================")
    print()

    # Add a new memory.
    memory.save(
        "My final year project presentation "
        "is scheduled for Friday.",
        memory_type="event",
    )

    # Search using completely different wording.
    query = (
        "When do I have to present my "
        "final year project?"
    )

    print("Query:")
    print(query)

    print()
    print("Semantic results:")
    print("------------------------------------")

    results = memory.search(
        query,
        limit=5,
    )

    for result in results:

        print(
            f"{result['similarity']:.4f} | "
            f"{result['content']}"
        )

    memory.close()


if __name__ == "__main__":
    main()
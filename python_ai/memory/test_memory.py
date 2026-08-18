from memory.manager import MemoryManager


def main() -> None:

    memory = MemoryManager()

    print()
    print("==============================")
    print("      VEMORA MEMORY TEST")
    print("==============================")
    print()

    # Save
    memory.save(
        "My FYP presentation is on Friday.",
        memory_type="event",
    )

    memory.save(
        "My favorite programming language is Java.",
        memory_type="preference",
    )

    print("Memories saved.")

    # Search
    results = memory.search(
        "FYP presentation"
    )

    print()
    print("Search results:")
    print()

    for result in results:

        print(
            f"- {result['content']}"
        )

    print()
    print("All memories:")
    print()

    for result in memory.all():

        print(
            f"- [{result['memory_type']}] "
            f"{result['content']}"
        )

    memory.close()


if __name__ == "__main__":
    main()
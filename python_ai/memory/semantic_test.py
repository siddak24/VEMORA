from sentence_transformers import SentenceTransformer


def main() -> None:
    print()
    print("======================================")
    print("       VEMORA SEMANTIC SEARCH TEST")
    print("======================================")
    print()

    print("[VEMORA] Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    memories = [
        "My FYP presentation is scheduled for Friday.",
        "My favorite programming language is Java.",
        "Rahul will send me the report tomorrow.",
        "I usually work on VEMORA at night.",
    ]

    query = (
        "When is my final year project presentation?"
    )

    print()
    print("Query:")
    print(query)

    # Convert memories into vectors.
    memory_embeddings = model.encode(
        memories,
        normalize_embeddings=True,
    )

    # Convert query into a vector.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # Cosine similarity becomes a dot product when
    # embeddings are normalized.
    similarities = (
        memory_embeddings @ query_embedding
    )

    ranked = sorted(
        zip(memories, similarities),
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print("Results:")
    print("--------------------------------------")

    for memory, score in ranked:

        print(
            f"{score:.4f}  |  {memory}"
        )


if __name__ == "__main__":
    main()
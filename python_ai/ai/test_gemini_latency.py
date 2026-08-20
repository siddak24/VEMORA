import time

from ai.providers import create_llm_provider


def main() -> None:
    ai = create_llm_provider()

    prompts = [
        "Reply: A",
        "Reply: B",
        "Reply: C",
    ]

    for i, prompt in enumerate(prompts, start=1):
        start = time.perf_counter()

        response = ai.generate_response(prompt)

        elapsed = time.perf_counter() - start

        print(
            f"Request {i}: {elapsed:.2f}s -> {response}"
        )


if __name__ == "__main__":
    main()
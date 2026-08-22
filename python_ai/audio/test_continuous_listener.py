from audio.continuous_listener import (
    ContinuousListener,
)


def main() -> None:

    listener = ContinuousListener()

    print()
    print("========================================")
    print("     VEMORA CONTINUOUS LISTENER TEST")
    print("========================================")
    print()
    print("Speak normally.")
    print("You do NOT need to press Enter.")
    print("Press Ctrl+C to stop.")
    print()

    try:

        while True:

            audio_file = (
                listener.listen_once()
            )

            print()
            print(
                f"[TEST] Captured: {audio_file}"
            )
            print()

    except KeyboardInterrupt:

        print()
        print("[TEST] Stopped.")


if __name__ == "__main__":
    main()
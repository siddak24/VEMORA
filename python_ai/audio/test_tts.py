import pyttsx3


def main() -> None:
    print("Starting TTS test...")

    engine = pyttsx3.init()

    voices = engine.getProperty("voices")

    print(f"Voices found: {len(voices)}")

    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name}")

    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    print("Speaking now...")

    engine.say(
        "Hello. This is VEMORA. "
        "This is a text to speech test."
    )

    engine.runAndWait()

    print("TTS finished.")


if __name__ == "__main__":
    main()
    
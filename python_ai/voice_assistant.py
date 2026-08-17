from pathlib import Path

from ai.providers import create_llm_provider
from audio.local_stt import LocalSTT
from audio.recorder import AudioRecorder
from audio.tts import LocalTTS


def main() -> None:
    print()
    print("========================================")
    print("          VEMORA VOICE ASSISTANT")
    print("========================================")
    print()
    print("Pipeline:")
    print("Microphone -> STT -> Gemini -> TTS")
    print()

    # ---------------------------------------------------------
    # 1. Microphone
    # ---------------------------------------------------------

    recorder = AudioRecorder()

    # ---------------------------------------------------------
    # 2. Local STT
    # ---------------------------------------------------------

    model_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "faster-whisper-base"
    )

    stt = LocalSTT(
        model_path=model_path,
        device="cpu",
        compute_type="int8",
    )

    # ---------------------------------------------------------
    # 3. Gemini
    # ---------------------------------------------------------

    llm = create_llm_provider()

    # ---------------------------------------------------------
    # 4. Local TTS
    # ---------------------------------------------------------

    tts = LocalTTS()

    print()
    print("[VEMORA] All systems initialized.")
    print()

    while True:

        try:
            # =================================================
            # RECORD
            # =================================================

            audio_file = recorder.record_until_enter(
                "data/voice_input.wav"
            )

            # =================================================
            # STT
            # =================================================

            print()
            print("[1/3] Speech-to-text...")

            user_text = stt.transcribe(
                audio_file
            )

            if not user_text:
                print(
                    "[VEMORA] No speech detected."
                )
                continue

            print()
            print("YOU:")
            print(user_text)

            # =================================================
            # GEMINI
            # =================================================

            print()
            print("[2/3] Gemini thinking...")

            response = llm.generate_response(
                user_text
            )

            print()
            print("VEMORA:")
            print(response)

            # =================================================
            # TTS
            # =================================================

            print()
            print("[3/3] Speaking...")

            tts.speak(response)

            print()
            print("----------------------------------------")
            print("Ready for next question.")
            print("----------------------------------------")

        except KeyboardInterrupt:

            print()
            print("[VEMORA] Stopped by user.")
            break

        except Exception as error:

            print()
            print("[VEMORA] ERROR:")
            print(error)
            print()

            # Keep the assistant alive after an individual
            # request fails.
            continue


if __name__ == "__main__":
    main()
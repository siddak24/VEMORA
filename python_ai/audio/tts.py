from __future__ import annotations

import pyttsx3


class LocalTTS:
    """
    Windows local TTS for VEMORA.

    A fresh engine is created for every utterance.
    This is slightly less efficient, but much more reliable
    for our current prototype.
    """

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
    ) -> None:
        self.rate = rate
        self.volume = volume

    def speak(self, text: str) -> None:

        text = text.strip()

        if not text:
            print("[VEMORA TTS] Empty response.")
            return

        print("[VEMORA TTS] Starting...")
        print("[VEMORA TTS]", text)

        engine = None

        try:
            engine = pyttsx3.init()

            engine.setProperty(
                "rate",
                self.rate,
            )

            engine.setProperty(
                "volume",
                self.volume,
            )

            engine.say(text)
            engine.runAndWait()

            print("[VEMORA TTS] Finished.")

        except Exception as error:

            print(
                f"[VEMORA TTS] ERROR: {error}"
            )

        finally:

            if engine is not None:

                try:
                    engine.stop()
                except Exception:
                    pass
from __future__ import annotations

import pyttsx3


class LocalTTS:
    """
    Offline text-to-speech for VEMORA's laptop prototype.
    """

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
    ) -> None:
        self.engine = pyttsx3.init()

        self.engine.setProperty(
            "rate",
            rate,
        )

        self.engine.setProperty(
            "volume",
            volume,
        )

    def speak(self, text: str) -> None:
        text = text.strip()

        if not text:
            return

        print("[VEMORA TTS]", text)

        self.engine.say(text)
        self.engine.runAndWait()
from __future__ import annotations

from pathlib import Path


class SpeechToText:
    """
    Interface for speech-to-text providers.
    """

    def transcribe(self, audio_file: Path) -> str:
        raise NotImplementedError
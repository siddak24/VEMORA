from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


class LocalSTT:

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        print(
            f"[VEMORA] Loading STT from:\n"
            f"{self.model_path}"
        )

        self.model = WhisperModel(
            str(self.model_path),
            device=device,
            compute_type=compute_type,
        )

        print("[VEMORA] STT ready.")

    def transcribe(
        self,
        audio_file: str | Path,
    ) -> str:

        audio_file = Path(audio_file)

        segments, info = self.model.transcribe(
            str(audio_file),

            # For our current project we're speaking English.
            language="en",

            beam_size=3,

            # Useful for short conversational commands.
            condition_on_previous_text=False,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 500,
            },

            temperature=0.0,
        )

        parts: list[str] = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                parts.append(text)

        return " ".join(parts).strip()
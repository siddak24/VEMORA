from __future__ import annotations

from pathlib import Path

import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def record_until_enter(
        self,
        output_path: str | Path,
    ) -> Path:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        input("Press ENTER to start speaking...")

        print("Listening... Speak now.")
        print("Press ENTER again when you finish.")

        chunks: list = []

        def callback(
            indata,
            frames,
            time,
            status,
        ):
            if status:
                print(
                    f"[MIC] {status}"
                )

            chunks.append(
                indata.copy()
            )

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
            blocksize=1024,
        ):

            input()

        if not chunks:
            raise RuntimeError(
                "No audio was recorded."
            )

        audio = __import__("numpy").concatenate(
            chunks,
            axis=0,
        )

        # Normalize very quiet recordings.
        peak = float(
            abs(audio).max()
        )

        if peak > 0:
            target_peak = 0.95

            if peak < 0.5:
                audio = (
                    audio
                    * (target_peak / peak)
                )

        sf.write(
            output_path,
            audio,
            self.sample_rate,
        )

        duration = (
            len(audio)
            / self.sample_rate
        )

        print(
            f"[MIC] Recorded {duration:.2f} seconds."
        )

        print(
            f"[MIC] Saved: {output_path}"
        )

        return output_path
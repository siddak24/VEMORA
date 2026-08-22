from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


class ContinuousListener:
    """
    Continuously monitors the microphone and captures
    one speech segment at a time.

    This is intentionally separate from AudioRecorder.

    AudioRecorder:
        Manual start/stop with Enter.

    ContinuousListener:
        Automatic speech start/stop using audio energy.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_duration: float = 0.25,
        silence_duration: float = 0.8,
        energy_threshold: float = 0.015,
        output_dir: str | Path = "data/stream",
    ) -> None:

        self.sample_rate = sample_rate
        self.channels = channels

        self.block_size = max(
            1,
            int(
                sample_rate
                * block_duration
            ),
        )

        self.silence_blocks = max(
            1,
            int(
                silence_duration
                / block_duration
            ),
        )

        self.energy_threshold = (
            energy_threshold
        )

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # AUDIO ENERGY
    # ==========================================================

    @staticmethod
    def _energy(
        audio: np.ndarray,
    ) -> float:
        """
        Calculate RMS energy of an audio block.
        """

        if audio.size == 0:
            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )

    # ==========================================================
    # CAPTURE ONE SPEECH SEGMENT
    # ==========================================================

    def listen_once(self) -> Path:
        """
        Wait for speech automatically.

        Starts recording when the audio energy goes above
        the threshold and stops after enough consecutive
        silent blocks.

        Returns:
            Path to the recorded WAV segment.
        """

        print(
            "[LISTENER] Waiting for speech..."
        )

        speech_started = False
        silence_count = 0

        chunks: list[np.ndarray] = []

        # Keep a little audio before speech starts so the
        # first word isn't clipped.
        pre_buffer: deque[np.ndarray] = deque(
            maxlen=3
        )

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block_size,
        ) as stream:

            while True:

                audio, _ = stream.read(
                    self.block_size
                )

                audio = np.asarray(
                    audio,
                    dtype=np.float32,
                )

                # sounddevice may return shape
                # (frames, 1). Keep it consistent.
                if audio.ndim == 1:

                    audio = audio.reshape(
                        -1,
                        1,
                    )

                energy = self._energy(
                    audio
                )

                is_speech = (
                    energy
                    >= self.energy_threshold
                )

                # --------------------------------------------------
                # WAITING FOR SPEECH
                # --------------------------------------------------

                if not speech_started:

                    pre_buffer.append(
                        audio.copy()
                    )

                    if is_speech:

                        speech_started = True
                        silence_count = 0

                        chunks.extend(
                            list(pre_buffer)
                        )

                        pre_buffer.clear()

                        print(
                            "[LISTENER] Speech detected."
                        )

                    continue

                # --------------------------------------------------
                # RECORDING SPEECH
                # --------------------------------------------------

                chunks.append(
                    audio.copy()
                )

                if is_speech:

                    silence_count = 0

                else:

                    silence_count += 1

                # --------------------------------------------------
                # END AFTER SILENCE
                # --------------------------------------------------

                if (
                    silence_count
                    >= self.silence_blocks
                ):

                    print(
                        "[LISTENER] Speech ended."
                    )

                    break

        if not chunks:

            raise RuntimeError(
                "No speech captured."
            )

        audio_data = np.concatenate(
            chunks,
            axis=0,
        )

        timestamp = int(
            time.time() * 1000
        )

        output_path = (
            self.output_dir
            / f"segment_{timestamp}.wav"
        )

        sf.write(
            output_path,
            audio_data,
            self.sample_rate,
        )

        duration = (
            len(audio_data)
            / self.sample_rate
        )

        print(
            f"[LISTENER] Captured "
            f"{duration:.2f} seconds."
        )

        print(
            f"[LISTENER] Saved: "
            f"{output_path}"
        )

        return output_path
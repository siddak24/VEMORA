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
    speech segments automatically.

    Passive listening:
        Uses the normal silence duration (default 0.8s).

    Direct command listening:
        Can wait for speech for a configurable amount of time
        after the wake word, e.g. 4 seconds.

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

        self.block_duration = block_duration

        # Normal passive-listening silence threshold.
        self.silence_duration = silence_duration

        self.silence_blocks = max(
            1,
            int(
                silence_duration
                / block_duration
            ),
        )

        self.energy_threshold = energy_threshold

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
    # INTERNAL AUDIO CAPTURE
    # ==========================================================

    def _capture_segment(
        self,
        silence_duration: float,
        wait_timeout: float | None = None,
        output_prefix: str = "segment",
        status_label: str = "speech",
    ) -> Path | None:
        """
        Internal speech-segment capture.

        If wait_timeout is provided, the listener will wait only
        that long for speech to START.

        Once speech starts, it keeps recording until the configured
        amount of silence is detected.

        Returns:
            Path to WAV file, or None if speech never started
            before wait_timeout expired.
        """

        silence_blocks = max(
            1,
            int(
                silence_duration
                / self.block_duration
            ),
        )

        print(
            "[LISTENER] Waiting for speech..."
        )

        speech_started = False
        silence_count = 0

        chunks: list[np.ndarray] = []

        # Keep a small amount of audio before speech starts so
        # the first word does not get clipped.
        pre_buffer: deque[np.ndarray] = deque(
            maxlen=3
        )

        start_time = time.monotonic()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block_size,
        ) as stream:

            while True:

                # --------------------------------------------------
                # WAIT TIMEOUT
                # --------------------------------------------------

                if (
                    wait_timeout is not None
                    and not speech_started
                    and (
                        time.monotonic()
                        - start_time
                        >= wait_timeout
                    )
                ):

                    print(
                        "[LISTENER] No speech detected "
                        f"within {wait_timeout:.1f} seconds."
                    )

                    return None

                # --------------------------------------------------
                # READ MICROPHONE
                # --------------------------------------------------

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
                            f"[LISTENER] "
                            f"{status_label.capitalize()} detected."
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
                # SPEECH ENDED
                # --------------------------------------------------

                if (
                    silence_count
                    >= silence_blocks
                ):

                    print(
                        "[LISTENER] Speech ended."
                    )

                    break

        if not chunks:

            return None

        # ------------------------------------------------------
        # COMBINE AUDIO
        # ------------------------------------------------------

        audio_data = np.concatenate(
            chunks,
            axis=0,
        )

        timestamp = int(
            time.time() * 1000
        )

        output_path = (
            self.output_dir
            / f"{output_prefix}_{timestamp}.wav"
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

    # ==========================================================
    # NORMAL PASSIVE LISTENING
    # ==========================================================

    def listen_once(
        self,
        silence_duration: float | None = None,
    ) -> Path:
        """
        Capture one normal speech segment.

        Default:
            0.8 seconds of silence ends the segment.

        This is used for passive session listening.
        """

        active_silence_duration = (
            silence_duration
            if silence_duration is not None
            else self.silence_duration
        )

        result = self._capture_segment(
            silence_duration=active_silence_duration,
            wait_timeout=None,
            output_prefix="segment",
            status_label="speech",
        )

        if result is None:

            raise RuntimeError(
                "No speech captured."
            )

        return result

    # ==========================================================
    # WAIT FOR COMMAND SPEECH
    # ==========================================================

    def wait_for_speech(
        self,
        timeout: float = 4.0,
        silence_duration: float | None = None,
    ) -> Path | None:
        """
        After VEMORA's wake word is detected, wait for the user
        to BEGIN speaking for up to `timeout` seconds.

        Example:

            "VEMORA..."
                 ↓
              pause 2 sec
                 ↓
            "what is my class?"

        This method allows that pause.

        Once speech starts, recording continues until silence.

        Args:
            timeout:
                Maximum time to wait for speech to START.

            silence_duration:
                Silence required AFTER speech has started.
                Defaults to the normal 0.8 seconds.

        Returns:
            Path to command WAV, or None if no speech starts
            within the timeout.
        """

        active_silence_duration = (
            silence_duration
            if silence_duration is not None
            else self.silence_duration
        )

        print()
        print(
            "[LISTENER] VEMORA is waiting for your command..."
        )

        result = self._capture_segment(
            silence_duration=active_silence_duration,
            wait_timeout=timeout,
            output_prefix="command",
            status_label="command speech",
        )

        return result
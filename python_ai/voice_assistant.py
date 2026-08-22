from __future__ import annotations

import re
from pathlib import Path

from ai.action_executor import ActionExecutor
from ai.providers import create_llm_provider
from audio.continuous_listener import ContinuousListener
from audio.local_stt import LocalSTT
from audio.tts import LocalTTS
from memory.database import MemoryDatabase
from memory.manager import MemoryManager
from session.manager import SessionManager


# ==============================================================
# WAKE WORD
# ==============================================================

WAKE_WORD = "vemora"

WAKE_WORD_VARIANTS = {
    "vemora",
    "vemura",
    "vimora",
    "vimura",
    "vemuro",
    "vimuro",
}


def normalize_word(text: str) -> str:
    return re.sub(
        r"[^a-z]",
        "",
        text.lower(),
    )


def is_wake_word(word: str) -> bool:
    """
    Detect common Whisper variations of VEMORA.
    """

    from difflib import SequenceMatcher

    normalized = normalize_word(word)

    if not normalized:
        return False

    if normalized in WAKE_WORD_VARIANTS:
        return True

    similarity = SequenceMatcher(
        None,
        normalized,
        WAKE_WORD,
    ).ratio()

    return similarity >= 0.72


def contains_wake_word(text: str) -> bool:
    """
    Check whether the transcript contains a word
    that is likely to be VEMORA.
    """

    words = re.findall(
        r"[A-Za-z]+",
        text,
    )

    return any(
        is_wake_word(word)
        for word in words
    )


def remove_wake_word(text: str) -> str:
    """
    Remove the first detected VEMORA wake word
    from a transcript.
    """

    words = text.split()

    result: list[str] = []
    removed = False

    for word in words:

        cleaned = normalize_word(word)

        if not removed and is_wake_word(cleaned):

            removed = True
            continue

        result.append(word)

    return " ".join(result).strip()


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    print()
    print("========================================")
    print("          VEMORA ASSISTANT")
    print("========================================")
    print()
    print("S = start session")
    print("Q = quit")
    print()

    # ==========================================================
    # PROJECT PATHS
    # ==========================================================

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    model_path = (
        project_root
        / "models"
        / "faster-whisper-base"
    )

    database_path = (
        project_root
        / "data"
        / "vemora.db"
    )

    # ==========================================================
    # CONTINUOUS MICROPHONE
    # ==========================================================

    listener = ContinuousListener(
        sample_rate=16000,
        channels=1,
        block_duration=0.25,
        silence_duration=0.8,
        energy_threshold=0.015,
        output_dir=(
            project_root
            / "data"
            / "stream"
        ),
    )

    # ==========================================================
    # STT
    # ==========================================================

    stt = LocalSTT(
        model_path=model_path,
        device="cpu",
        compute_type="int8",
    )

    # ==========================================================
    # AI
    # ==========================================================

    llm = create_llm_provider()

    # ==========================================================
    # TTS
    # ==========================================================

    tts = LocalTTS()

    # ==========================================================
    # DATABASE
    # ==========================================================

    database = MemoryDatabase(
        db_path=database_path
    )

    # ==========================================================
    # LONG-TERM MEMORY
    # ==========================================================

    memory = MemoryManager(
        user_id="default_user"
    )

    # ==========================================================
    # SESSION
    # ==========================================================

    session = SessionManager(
        database=database,
        user_id="default_user",
        embedding_model=memory.embedding_model,
    )

    # ==========================================================
    # ACTION EXECUTOR
    # ==========================================================

    executor = ActionExecutor(
        memory=memory,
        session=session,
    )

    print()
    print("[VEMORA] All systems ready.")
    print()

    # ==========================================================
    # COMMAND LOOP
    # ==========================================================

    while True:

        try:

            command = input(
                "[VEMORA] Command: "
            ).strip().lower()

            # ==================================================
            # QUIT
            # ==================================================

            if command == "q":

                if session.session_id is not None:

                    print(
                        "[SESSION] Ending active session..."
                    )

                    session.end()

                print(
                    "[VEMORA] Shutting down."
                )

                break

            # ==================================================
            # START SESSION
            # ==================================================

            if command != "s":

                print(
                    "[VEMORA] Press S to start "
                    "or Q to quit."
                )

                continue

            if session.session_id is not None:

                print(
                    "[VEMORA] A session is already active."
                )

                continue

            session.start(
                session_type="conversation"
            )

            print()
            print(
                "[VEMORA] SESSION ACTIVE."
            )
            print(
                "VEMORA is now continuously listening."
            )
            print(
                "Say 'VEMORA' whenever you want a response."
            )
            print(
                "Say 'VEMORA, stop listening' to end the session."
            )
            print()

            # ==================================================
            # ACTIVE SESSION LOOP
            # ==================================================

            while session.session_id is not None:

                # ------------------------------------------------
                # Wait for the next speech segment.
                # No Enter required.
                # ------------------------------------------------

                try:

                    audio_file = (
                        listener.listen_once()
                    )

                except Exception as error:

                    print(
                        f"[LISTENER] ERROR: {error}"
                    )

                    continue

                # ------------------------------------------------
                # STT
                # ------------------------------------------------

                print()
                print(
                    "[SESSION] Transcribing..."
                )

                try:

                    user_text = stt.transcribe(
                        audio_file
                    )

                except Exception as error:

                    print(
                        f"[STT] ERROR: {error}"
                    )

                    continue

                user_text = user_text.strip()

                if not user_text:

                    print(
                        "[SESSION] Empty transcript."
                    )

                    continue

                print()
                print("YOU:")
                print(user_text)

                # ------------------------------------------------
                # Check whether VEMORA was addressed.
                # ------------------------------------------------

                wake_word_detected = contains_wake_word(
                    user_text
                )

                # ------------------------------------------------
                # Classify the transcript chunk.
                # ------------------------------------------------

                chunk_type = (
                    "DIRECT_COMMAND"
                    if wake_word_detected
                    else "PASSIVE"
                )

                # ------------------------------------------------
                # Store transcript exactly once.
                # ------------------------------------------------

                session.add_transcript(
                    user_text,
                    chunk_type=chunk_type,
                )

                # ------------------------------------------------
                # Passive listening
                # ------------------------------------------------

                if not wake_word_detected:

                    print(
                        "[SESSION] Passive listening..."
                    )

                    continue
                # ==================================================
                # DIRECT VEMORA INTERACTION DETECTED
                # ==================================================

                print()
                print(
                    "[SESSION] VEMORA detected."
                )

                command_text = remove_wake_word(
                    user_text
                )

                # ------------------------------------------------
                # Important:
                # If the passive speech segment contains VEMORA,
                # the first segment may contain only the wake word
                # or the start of the command.
                #
                # We therefore listen for another segment so the
                # user's full command can be captured naturally.
                # ------------------------------------------------

                if not command_text:

                    print(
                        "[SESSION] Listening for command..."
                    )

                    try:

                        command_audio = (
                            listener.listen_once()
                        )

                        command_text = (
                            stt.transcribe(
                                command_audio
                            ).strip()
                        )

                    except Exception as error:

                        print(
                            f"[COMMAND] ERROR: {error}"
                        )

                        continue

                print()
                print(
                    "[COMMAND]"
                )
                print(command_text)

                # ------------------------------------------------
                # Store the direct command in the session too.
                # ------------------------------------------------
                # ==================================================
                # SESSION CONTEXT
                # ==================================================

                session_context = (
                    session.recent_context(
                        limit=10
                    )
                )

                # ==================================================
                # ACTION PLAN
                # ==================================================

                print()
                print(
                    "[VEMORA] Creating action plan..."
                )

                plan = llm.create_action_plan(
                    user_text=command_text,
                    session_context=session_context,
                )

                print()
                print(
                    "[ACTION PLAN]"
                )

                print(
                    plan.model_dump_json(
                        indent=2
                    )
                )

                # ==================================================
                # EXECUTE TOOLS
                # ==================================================

                print()
                print(
                    "[VEMORA] Executing actions..."
                )

                try:

                    tool_results = (
                        executor.execute(
                            plan
                        )
                    )

                except Exception as error:

                    print(
                        f"[TOOLS] ERROR: {error}"
                    )

                    continue

                print()
                print(
                    "[TOOL RESULTS]"
                )
                print(tool_results)

                # ==================================================
                # SHOULD VEMORA SPEAK?
                # ==================================================

                if not plan.should_speak:

                    print(
                        "[VEMORA] No spoken response required."
                    )

                    continue

                # ==================================================
                # FINAL GROUNDED RESPONSE
                # ==================================================

                print()
                print(
                    "[VEMORA] Generating response..."
                )

                try:

                    response = (
                        llm.generate_grounded_response(
                            user_text=command_text,
                            tool_results=tool_results,
                            instruction=(
                                plan.response_instruction
                            ),
                        )
                    )

                except Exception as error:

                    print(
                        f"[AI] ERROR: {error}"
                    )

                    continue

                response = response.strip()

                if not response:

                    print(
                        "[VEMORA] Empty response."
                    )

                    continue

                print()
                print(
                    "VEMORA:"
                )
                print(response)

                # ==================================================
                # TTS
                # ==================================================

                print()
                print(
                    "[VEMORA] Speaking..."
                )

                try:

                    tts.speak(
                        response
                    )

                except Exception as error:

                    print(
                        f"[TTS] ERROR: {error}"
                    )

                print()
                print(
                    "[VEMORA] Back to listening."
                )
                print()

                # ==================================================
                # SPECIAL COMMAND:
                # STOP LISTENING
                # ==================================================

                normalized_command = (
                    command_text.lower().strip()
                )

                if (
                    "stop listening"
                    in normalized_command
                    or "stop the session"
                    in normalized_command
                    or "end the session"
                    in normalized_command
                ):

                    print()
                    print(
                        "[SESSION] Stop command detected."
                    )

                    session.end()

                    print(
                        "[VEMORA] Back to IDLE."
                    )

                    break

            print()

        except KeyboardInterrupt:

            print()
            print(
                "[VEMORA] Interrupted."
            )

            if session.session_id is not None:

                session.end()

            break

        except Exception as error:

            print()
            print(
                "[VEMORA] ERROR:"
            )
            print(error)

            print()
            print(
                "[VEMORA] Returning to command mode."
            )

    # ==========================================================
    # CLEANUP
    # ==========================================================

    try:
        memory.close()
    except Exception:
        pass

    try:
        database.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
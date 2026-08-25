from __future__ import annotations
from task.scheduler import TaskScheduler
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

from ai.action_executor import ActionExecutor
from ai.providers import create_llm_provider

from audio.continuous_listener import ContinuousListener
from audio.local_stt import LocalSTT
from audio.tts import LocalTTS

from memory.database import MemoryDatabase
from memory.manager import MemoryManager

from task.manager import TaskManager

from session.manager import SessionManager
from session.processor import SessionProcessor


load_dotenv()


# ==============================================================
# CONFIGURATION
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

PASSIVE_SILENCE_DURATION = 0.8
COMMAND_START_GRACE_PERIOD = 4.0

WAKE_WORD_SIMILARITY_THRESHOLD = 0.72


# ==============================================================
# WAKE WORD HELPERS
# ==============================================================

def normalize_word(text: str) -> str:
    """
    Normalize a word for wake-word comparison.
    """

    return re.sub(
        r"[^a-z]",
        "",
        text.lower(),
    )


def is_wake_word(word: str) -> bool:
    """
    Detect VEMORA and common Whisper variations.
    """

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

    return (
        similarity
        >= WAKE_WORD_SIMILARITY_THRESHOLD
    )


def contains_wake_word(text: str) -> bool:
    """
    Check whether the transcript contains a likely
    VEMORA wake word.
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
    Remove the first detected VEMORA wake word.
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
# AUDIO CLEANUP
# ==============================================================

def cleanup_audio_file(
    audio_file: Path,
) -> None:
    """
    Delete a temporary audio segment after successful STT,
    unless KEEP_AUDIO_SEGMENTS=true is enabled.
    """

    keep_audio = (
        os.getenv(
            "KEEP_AUDIO_SEGMENTS",
            "false",
        ).lower()
        == "true"
    )

    if keep_audio:

        print(
            f"[AUDIO] Keeping segment: {audio_file}"
        )

        return

    try:

        if audio_file.exists():

            audio_file.unlink()

            print(
                f"[AUDIO] Deleted temporary segment: "
                f"{audio_file}"
            )

    except OSError as error:

        print(
            f"[AUDIO] Could not delete "
            f"{audio_file}: {error}"
        )


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

    stream_directory = (
        project_root
        / "data"
        / "stream"
    )

    # ==========================================================
    # CONTINUOUS MICROPHONE
    # ==========================================================

    listener = ContinuousListener(
        sample_rate=16000,
        channels=1,
        block_duration=0.25,
        silence_duration=PASSIVE_SILENCE_DURATION,
        energy_threshold=0.015,
        output_dir=stream_directory,
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
        db_path=database_path,
    )

    # ==========================================================
    # LONG-TERM MEMORY
    # ==========================================================

    memory = MemoryManager(
        user_id="default_user",
    )

    # ==========================================================
    # TASKS
    # ==========================================================

    tasks = TaskManager(
        database=database,
        user_id="default_user",
    )

    task_scheduler = TaskScheduler(
        tasks=tasks,
        interval_seconds=30,
    )

    task_scheduler.start()
    # ==========================================================
    # SESSION
    # ==========================================================

    session = SessionManager(
        database=database,
        user_id="default_user",
        embedding_model=memory.embedding_model,
    )

    # ==========================================================
    # SESSION PROCESSOR
    # ==========================================================

    processor = SessionProcessor(
        session=session,
        batch_size=5,
        batch_interval_seconds=20.0,
    )

    # ==========================================================
    # ACTION EXECUTOR
    # ==========================================================

    executor = ActionExecutor(
        memory=memory,
        session=session,
        tasks=tasks,
    )

    print()
    print(
        "[VEMORA] All systems ready."
    )
    print()

    # ==========================================================
    # MAIN COMMAND LOOP
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

                    if processor.has_pending_passive_data():

                        pending = (
                            processor.flush()
                        )

                        if pending:

                            print(
                                "[SESSION] "
                                "Pending passive data remains "
                                "unprocessed."
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

            processor.reset()

            session.start(
                session_type="conversation",
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
                "You can also state useful tasks naturally "
                "without saying VEMORA."
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
                # LISTEN FOR SPEECH
                # ------------------------------------------------

                try:

                    audio_file = (
                        listener.listen_once(
                            silence_duration=(
                                PASSIVE_SILENCE_DURATION
                            )
                        )
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

                    user_text = (
                        stt.transcribe(
                            audio_file
                        ).strip()
                    )

                except Exception as error:

                    print(
                        f"[STT] ERROR: {error}"
                    )

                    print(
                        "[AUDIO] Keeping failed STT "
                        "segment for debugging."
                    )

                    continue

                cleanup_audio_file(
                    audio_file
                )

                if not user_text:

                    print(
                        "[SESSION] Empty transcript."
                    )

                    continue

                print()
                print("YOU:")
                print(user_text)

                # ------------------------------------------------
                # WAKE WORD DETECTION
                # ------------------------------------------------

                wake_word_detected = (
                    contains_wake_word(
                        user_text
                    )
                )

                # ==================================================
                # PASSIVE SPEECH
                # ==================================================

                if not wake_word_detected:

                    processor.add_passive_chunk(
                        user_text
                    )

                    print(
                        "[SESSION] Passive listening..."
                    )

                    # ------------------------------------------------
                    # PASSIVE AI BATCH
                    # ------------------------------------------------

                    if processor.should_process_passive():

                        passive_batch = (
                            processor.get_passive_batch()
                        )

                        if passive_batch:

                            print()
                            print(
                                "[SESSION] "
                                "Processing passive batch..."
                            )

                            print(
                                "--------------------------------"
                            )

                            print(
                                passive_batch
                            )

                            try:

                                passive_plan = (
                                    llm.process_passive_batch(
                                        passive_batch
                                    )
                                )

                                print()
                                print(
                                    "[PASSIVE AI PLAN]"
                                )

                                print(
                                    passive_plan.model_dump_json(
                                        indent=2
                                    )
                                )

                                # Passive processing can:
                                #   - save memories
                                #   - update memories
                                #   - delete memories
                                #   - create tasks
                                #
                                # It cannot search or speak.

                                passive_results = (
                                    executor.execute(
                                        passive_plan,
                                        allowed_tools={
                                            "save_memory",
                                            "update_memory",
                                            "delete_memory",
                                            "create_task",
                                            "complete_task",
                                        },
                                    )
                                )

                                print()
                                print(
                                    "[PASSIVE ACTION RESULTS]"
                                )

                                print(
                                    passive_results
                                )

                            except Exception as error:

                                print(
                                    f"[PASSIVE AI] ERROR: {error}"
                                )

                    continue

                # ==================================================
                # DIRECT VEMORA INTERACTION
                # ==================================================

                print()
                print(
                    "[SESSION] VEMORA detected."
                )

                command_text = (
                    remove_wake_word(
                        user_text
                    )
                ).strip()

                # ==================================================
                # WAKE WORD ONLY / VERY SHORT COMMAND
                # ==================================================

                if len(command_text.split()) <= 2:

                    print()
                    print(
                        "[SESSION] "
                        "VEMORA is listening for your command..."
                    )

                    try:

                        command_audio = (
                            listener.wait_for_speech(
                                timeout=(
                                    COMMAND_START_GRACE_PERIOD
                                )
                            )
                        )

                    except Exception as error:

                        print(
                            f"[COMMAND] ERROR: {error}"
                        )

                        continue

                    if command_audio is None:

                        print(
                            "[SESSION] "
                            "No command detected."
                        )

                        continue

                    try:

                        follow_up_text = (
                            stt.transcribe(
                                command_audio
                            ).strip()
                        )

                    except Exception as error:

                        print(
                            f"[COMMAND STT] ERROR: {error}"
                        )

                        print(
                            "[AUDIO] Keeping failed command "
                            "segment for debugging."
                        )

                        continue

                    cleanup_audio_file(
                        command_audio
                    )

                    if not follow_up_text:

                        print(
                            "[SESSION] "
                            "No command was transcribed."
                        )

                        continue

                    command_text = follow_up_text

                # ==================================================
                # DIRECT COMMAND
                # ==================================================

                command_text = command_text.strip()

                if not command_text:

                    print(
                        "[SESSION] Empty command."
                    )

                    continue

                print()
                print(
                    "[COMMAND]"
                )
                print(command_text)

                # ------------------------------------------------
                # Store the direct command once.
                # ------------------------------------------------

                session.add_transcript(
                    command_text,
                    chunk_type="DIRECT_COMMAND",
                )

                # ==================================================
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

                try:

                    plan = (
                        llm.create_action_plan(
                            user_text=command_text,
                            session_context=session_context,
                        )
                    )

                except Exception as error:

                    print(
                        f"[AI PLAN] ERROR: {error}"
                    )

                    continue

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

                print(
                    tool_results
                )

                # ==================================================
                # SHOULD VEMORA SPEAK?
                # ==================================================

                if not plan.should_speak:

                    print(
                        "[VEMORA] "
                        "No spoken response required."
                    )

                    continue

                # ==================================================
                # GROUNDED RESPONSE
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
                        f"[AI RESPONSE] ERROR: {error}"
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

                print(
                    response
                )

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

                    continue

                print()
                print(
                    "[VEMORA] Back to listening."
                )

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

            print(
                error
            )

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
        task_scheduler.stop()
    except Exception:
        pass
    
    try:
        database.close()
    except Exception:
        pass
    

if __name__ == "__main__":
    main()
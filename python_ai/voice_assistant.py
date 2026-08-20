from __future__ import annotations

from pathlib import Path

from ai.providers import create_llm_provider
from audio.local_stt import LocalSTT
from audio.recorder import AudioRecorder
from audio.tts import LocalTTS
from memory.manager import MemoryManager
from session.manager import SessionManager
from memory.database import MemoryDatabase


def main() -> None:
    print()
    print("========================================")
    print("          VEMORA ASSISTANT")
    print("========================================")
    print()
    print("S = start session")
    print("E = end session")
    print("Q = quit")
    print()

    # ---------------------------------------------------------
    # Components
    # ---------------------------------------------------------

    recorder = AudioRecorder()

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

    llm = create_llm_provider()

    tts = LocalTTS()

    database = MemoryDatabase(
        db_path=(
            Path(__file__).resolve().parents[1]
            / "data"
            / "vemora.db"
        )
    )

    memory = MemoryManager(
        user_id="default_user"
    )

    session = SessionManager(
        database=database,
        user_id="default_user",
    )

    print("[VEMORA] All systems ready.")
    print()

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------

    while True:

        try:

            command = input(
                "[VEMORA] Command: "
            ).strip().lower()

            # =================================================
            # QUIT
            # =================================================

            if command == "q":

                if session.session_id is not None:
                    session.end()

                print(
                    "[VEMORA] Shutting down."
                )

                break

            # =================================================
            # START SESSION
            # =================================================

            if command == "s":

                if session.session_id is not None:

                    print(
                        "[VEMORA] A session is "
                        "already active."
                    )

                    continue

                session.start(
                    session_type="conversation"
                )

                print()
                print(
                    "[VEMORA] SESSION ACTIVE"
                )
                print(
                    "Press ENTER to record speech."
                )
                print(
                    "Press E when you want to end "
                    "the session."
                )
                print()

                # ---------------------------------------------
                # Session loop
                # ---------------------------------------------

                while (
                    session.session_id
                    is not None
                ):

                    command = input(
                        "[SESSION] Press ENTER "
                        "to speak, or E to end: "
                    ).strip().lower()

                    # -----------------------------------------
                    # END SESSION
                    # -----------------------------------------

                    if command == "e":

                        print(
                            "[SESSION] Ending..."
                        )

                        transcript = (
                            session.full_transcript()
                        )

                        print()
                        print(
                            "Session transcript:"
                        )
                        print(
                            "------------------------------"
                        )
                        print(transcript)

                        session.end()

                        print()
                        print(
                            "[VEMORA] Back to IDLE."
                        )
                        print()

                        break

                    # -----------------------------------------
                    # RECORD
                    # -----------------------------------------

                    audio_file = (
                        recorder.record_until_enter(
                            "data/session_input.wav"
                        )
                    )

                    # -----------------------------------------
                    # STT
                    # -----------------------------------------

                    print()
                    print(
                        "[SESSION] Transcribing..."
                    )

                    user_text = stt.transcribe(
                        audio_file
                    )

                    if not user_text:

                        print(
                            "[SESSION] "
                            "No speech detected."
                        )

                        continue

                    print()
                    print(
                        "YOU:"
                    )
                    print(user_text)

                    # -----------------------------------------
                    # SAVE TO SESSION
                    # -----------------------------------------

                    session.add_transcript(
                        user_text
                    )

                    print(
                        "[SESSION] Transcript saved."
                    )

                    # -----------------------------------------
                    # NORMAL VEMORA RESPONSE
                    # -----------------------------------------

                    # ---------------------------------------------------------
                    # SESSION DECISION
                    # ---------------------------------------------------------

                    context = session.recent_context(
                        limit=10
                    )

                    decision = llm.decide_session_action(
                        user_text=user_text,
                        session_context=context,
                    )

                    print(
                        f"[SESSION] Action: {decision.action}"
                    )

                    # ---------------------------------------------------------
                    # SAVE MEMORY
                    # ---------------------------------------------------------

                    if decision.action == "SAVE_MEMORY":

                        if decision.content.strip():

                            memory_id = memory.save(
                                content=decision.content,
                                memory_type=decision.memory_type,
                            )

                            print(
                                f"[MEMORY] Saved #{memory_id}: "
                                f"{decision.content}"
                            )

                    # ---------------------------------------------------------
                    # SEARCH MEMORY
                    # ---------------------------------------------------------

                    elif decision.action == "SEARCH_MEMORY":

                        query = (
                            decision.query.strip()
                            or user_text
                        )

                        results = memory.search(
                            query=query,
                            limit=5,
                        )

                        if results:

                            memory_context = "\n".join(
                                f"- {item['content']}"
                                for item in results
                            )

                            response = llm.generate_response(
                                f"""
                    Answer the user's question using ONLY these memories.

                    User:
                    {user_text}

                    Memories:
                    {memory_context}

                    Keep the answer concise.
                    Do not invent information.
                    """
                            )

                        else:

                            response = (
                                "I don't have anything relevant saved."
                            )

                        print()
                        print("VEMORA:")
                        print(response)

                        tts.speak(response)

                    # ---------------------------------------------------------
                    # DIRECT RESPONSE
                    # ---------------------------------------------------------

                    elif decision.action == "RESPOND":

                        response = llm.generate_response(
                            f"""
                    You are VEMORA in an active listening session.

                    Recent context:
                    {context}

                    User:
                    {user_text}

                    Answer briefly and naturally.
                    """
                        )

                        print()
                        print("VEMORA:")
                        print(response)

                        tts.speak(response)

                    # ---------------------------------------------------------
                    # LISTEN
                    # ---------------------------------------------------------

                    else:

                        print(
                            "[SESSION] Listening silently..."
                        )

                    print()

                continue

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
                f"[VEMORA] ERROR: {error}"
            )

    database.close()
    memory.close()


if __name__ == "__main__":
    main()
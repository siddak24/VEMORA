from __future__ import annotations

from pathlib import Path

from ai.providers import create_llm_provider
from audio.local_stt import LocalSTT
from audio.recorder import AudioRecorder
from audio.tts import LocalTTS
from memory.manager import MemoryManager


def main() -> None:
    print()
    print("========================================")
    print("        VEMORA SEMANTIC MEMORY")
    print("========================================")
    print()

    # ---------------------------------------------------------
    # Audio
    # ---------------------------------------------------------

    recorder = AudioRecorder()

    # ---------------------------------------------------------
    # Local STT
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Gemini
    # ---------------------------------------------------------

    llm = create_llm_provider()

    # ---------------------------------------------------------
    # TTS
    # ---------------------------------------------------------

    tts = LocalTTS()

    # ---------------------------------------------------------
    # Semantic Memory
    # ---------------------------------------------------------

    memory = MemoryManager(
        user_id="default_user"
    )

    print()
    print("[VEMORA] All systems ready.")
    print()

    while True:

        try:
            # =================================================
            # 1. RECORD
            # =================================================

            audio_file = recorder.record_until_enter(
                "data/voice_input.wav"
            )

            # =================================================
            # 2. STT
            # =================================================

            print()
            print("[1] Transcribing...")

            user_text = stt.transcribe(
                audio_file
            )

            if not user_text:

                print(
                    "[VEMORA] No speech detected."
                )

                continue

            print()
            print("YOU:")
            print(user_text)

            # =================================================
            # 3. MEMORY DECISION
            # =================================================

            print()
            print("[2] Understanding request...")

            decision = llm.decide_memory_action(
                user_text
            )

            print(
                f"[MEMORY] Action: {decision.action}"
            )

            # =================================================
            # 4. SAVE MEMORY
            # =================================================

            if decision.action == "SAVE_MEMORY":

                if not decision.content.strip():

                    response = (
                        "I understood that you want "
                        "me to remember something, "
                        "but I couldn't determine "
                        "what to save."
                    )

                else:

                    memory_id = memory.save(
                        content=decision.content,
                        memory_type=(
                            decision.memory_type
                        ),
                    )

                    print(
                        f"[MEMORY] Saved #{memory_id}: "
                        f"{decision.content}"
                    )

                    response = (
                        "Okay, I'll remember that."
                    )

            # =================================================
            # 5. SEARCH MEMORY
            # =================================================

            elif decision.action == "SEARCH_MEMORY":

                query = (
                    decision.query.strip()
                    or user_text
                )

                print(
                    f"[MEMORY] Searching for: "
                    f"{query}"
                )

                results = memory.search(
                    query=query,
                    limit=5,
                )

                print(
                    f"[MEMORY] Found "
                    f"{len(results)} result(s)."
                )

                if not results:

                    response = (
                        "I don't have anything "
                        "relevant saved about that."
                    )

                else:

                    print()
                    print(
                        "[MEMORY] Relevant memories:"
                    )

                    for result in results:

                        print(
                            f"  "
                            f"{result['similarity']:.3f} "
                            f"| "
                            f"{result['content']}"
                        )

                    # -------------------------------------------------
                    # Give only the retrieved memories to Gemini.
                    # -------------------------------------------------

                    memory_context = "\n".join(
                        f"- {result['content']}"
                        for result in results
                    )

                    response = (
                        llm.generate_response(
                            f"""
You are VEMORA, a concise personal AI assistant.

Answer the user's question using ONLY the
relevant memories provided below.

User question:
{user_text}

Relevant memories:
{memory_context}

Rules:
- Do not invent information.
- Do not mention the memory system.
- Give a natural, concise answer.
- If the memories are insufficient, say that
  you don't have enough saved information.
"""
                        )
                    )

            # =================================================
            # 6. NORMAL CHAT
            # =================================================

            else:

                response = llm.generate_response(
                    user_text
                )

            # =================================================
            # 7. OUTPUT
            # =================================================

            print()
            print("VEMORA:")
            print(response)

            # =================================================
            # 8. TTS
            # =================================================

            print()
            print("[3] Speaking...")

            tts.speak(response)

            print()
            print("----------------------------------------")
            print("Ready for next question.")
            print("----------------------------------------")

        except KeyboardInterrupt:

            print()
            print("[VEMORA] Exiting.")

            memory.close()

            break

        except Exception as error:

            print()
            print(
                f"[VEMORA] ERROR: {error}"
            )

            print()
            print(
                "VEMORA is still running."
            )

    # Safety close.
    memory.close()


if __name__ == "__main__":
    main()
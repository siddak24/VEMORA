from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ai.action_models import ActionPlan
from memory.models import MemoryDecision
from session.decision import SessionDecision


load_dotenv()


class DemoProvider:
    """
    Local provider used when VEMORA is running in demo mode.
    """

    def generate_response(
        self,
        user_text: str,
    ) -> str:

        return (
            f'[DEMO AI] I received: "{user_text}"'
        )

    def decide_memory_action(
        self,
        user_text: str,
    ) -> MemoryDecision:

        return MemoryDecision(
            action="CHAT",
            content="",
            query="",
            memory_type="general",
            importance=0.0,
            confidence=1.0,
            retention="SHORT_TERM",
        )

    def decide_session_action(
        self,
        user_text: str,
        session_context: str = "",
    ) -> SessionDecision:

        return SessionDecision(
            action="LISTEN",
            should_speak=False,
        )

    def create_action_plan(
        self,
        user_text: str,
        session_context: str = "",
    ) -> ActionPlan:

        return ActionPlan(
            actions=[],
            should_speak=True,
            response_instruction=(
                "Give a concise response."
            ),
        )

    def generate_grounded_response(
        self,
        user_text: str,
        tool_results: list[dict],
        instruction: str = "",
    ) -> str:

        return (
            "[DEMO AI] "
            "I could not use real tool results "
            "while demo mode is enabled."
        )


class GeminiProvider:
    """
    Gemini API provider for VEMORA.
    """

    def __init__(self) -> None:

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = "gemini-3.5-flash-lite"

    # ==========================================================
    # MEMORY DECISION
    # ==========================================================

    def decide_memory_action(
        self,
        user_text: str,
    ) -> MemoryDecision:

        prompt = f"""
You are the memory controller for VEMORA,
an AI wearable assistant.

Decide what VEMORA should do with the user's message.

Possible actions:

1. CHAT
Use for normal questions, conversation,
explanations, translations, calculations,
and general requests.

2. SAVE_MEMORY
Use when the user wants VEMORA to remember
useful information for later.

3. SEARCH_MEMORY
Use when the user asks about something
VEMORA may have remembered previously.

4. UPDATE_MEMORY
Use when the user is correcting or changing
a previously stored memory.

5. DELETE_MEMORY
Use when the user explicitly asks VEMORA
to forget or delete something.

Rules:

- Do not invent memories.
- Do not save every ordinary conversation.
- Save information that is useful and likely
  to matter later.
- Preserve the user's meaning.
- Keep saved content concise.
- For SEARCH_MEMORY, create a useful query.
- For UPDATE_MEMORY, create a useful query
  to identify the old memory and provide
  the corrected content.
- For DELETE_MEMORY, create a useful query
  to identify the memory to remove.
- For CHAT, leave content and query empty.
- Estimate importance and confidence carefully.
- Choose an appropriate retention policy.

User message:
{user_text}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MemoryDecision,
            ),
        )

        return MemoryDecision.model_validate_json(
            response.text
        )

    # ==========================================================
    # SESSION DECISION
    # ==========================================================

    def decide_session_action(
        self,
        user_text: str,
        session_context: str = "",
    ) -> SessionDecision:

        prompt = f"""
You are VEMORA, a wearable AI assistant
in ACTIVE LISTENING mode.

The user has intentionally asked VEMORA
to listen to an ongoing conversation,
lecture, seminar, meeting, or real-world
situation.

VEMORA should normally remain SILENT.

Possible actions:

1. LISTEN
The utterance is part of the ongoing
conversation. Record useful context
but do not respond.

2. RESPOND
The user is directly asking VEMORA
something or explicitly requesting
a response.

3. SAVE_MEMORY
The utterance contains useful persistent
personal information worth remembering.

4. SEARCH_MEMORY
The user is asking about information
VEMORA may already remember.

5. UPDATE_MEMORY
The user is correcting stored information.

6. DELETE_MEMORY
The user asks VEMORA to forget stored
information.

Rules:

- Default to LISTEN.
- Do not speak merely because something
  interesting was said.
- Do not answer every utterance.
- should_speak should normally be false.
- If the user explicitly addresses VEMORA,
  use RESPOND or an appropriate memory action
  and set should_speak=true.
- For SAVE_MEMORY, should_speak is normally false
  unless the user asks for confirmation.
- Never invent information.
- Keep memory content concise.
- For UPDATE_MEMORY and DELETE_MEMORY,
  provide a useful query.

Recent session context:
{session_context}

Current utterance:
{user_text}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SessionDecision,
            ),
        )

        return SessionDecision.model_validate_json(
            response.text
        )

    # ==========================================================
    # ACTION PLAN
    # ==========================================================

    def create_action_plan(
        self,
        user_text: str,
        session_context: str = "",
    ) -> ActionPlan:

        prompt = f"""
You are VEMORA, a wearable AI assistant.

Your job is to decide which tools are needed
to answer the user's request.

Available tools:

1. search_session
Search the current active session for recent
information.

2. search_memory
Search persistent long-term semantic memory.

3. save_memory
Save useful information for future use.

4. update_memory
Update an existing persistent memory.

5. delete_memory
Delete a persistent memory.

Rules:

- Use multiple tools when useful.
- You may request session and long-term
  memory searches together.
- Do not invent information.
- Do not save ordinary conversation unless
  it is useful for future recall or
  personalization.
- If the user directly asks VEMORA something,
  should_speak should normally be true.
- During passive listening, should_speak
  can be false.
- Keep the action plan minimal.
- Only request tools that are actually useful.

Current session context:
{session_context}

User request:
{user_text}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ActionPlan,
            ),
        )

        return ActionPlan.model_validate_json(
            response.text
        )

    # ==========================================================
    # GROUNDED RESPONSE
    # ==========================================================

    def generate_grounded_response(
        self,
        user_text: str,
        tool_results: list[dict],
        instruction: str = "",
    ) -> str:

        prompt = f"""
You are VEMORA, a concise wearable AI assistant.

Answer the user's request using ONLY the
evidence returned by the tools below.

User request:
{user_text}

Tool results:
{tool_results}

Instructions:
{instruction}

Rules:
- Do not invent information.
- If the evidence is insufficient, say so.
- Keep the answer concise.
- Normally use 1 to 3 sentences.
- Give the most useful information first.
- Do not mention tools, databases,
  prompts, or internal architecture.
- The response will be spoken aloud.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text.strip()

    # ==========================================================
    # NORMAL RESPONSE
    # ==========================================================

    def generate_response(
        self,
        user_text: str,
    ) -> str:

        prompt = f"""
You are VEMORA, a smart wearable AI assistant.

Answer the user's request clearly and naturally.

Response rules:
- Keep the answer SHORT and concise.
- Normally use 1 to 3 sentences.
- Give the most important information first.
- Do not add unnecessary background or explanations.
- Do not repeat the user's question.
- For simple factual questions, answer in 1 sentence.
- Only give a long or detailed answer when the
  user explicitly asks for an explanation,
  details, steps, or a longer response.
- The response will be spoken aloud,
  so make it natural for speech.

User:
{user_text}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text.strip()


# ==============================================================
# PROVIDER FACTORY
# ==============================================================

def create_llm_provider():
    demo_mode = os.getenv(
        "DEMO_MODE",
        "true",
    ).lower() == "true"

    provider = os.getenv(
        "AI_PROVIDER",
        "gemini",
    ).lower()

    if demo_mode:
        print(
            "[VEMORA] Demo AI enabled."
        )
        return DemoProvider()

    if provider == "gemini":
        print(
            "[VEMORA] Gemini API enabled."
        )
        return GeminiProvider()

    raise ValueError(
        f"Unsupported AI_PROVIDER: {provider}"
    )
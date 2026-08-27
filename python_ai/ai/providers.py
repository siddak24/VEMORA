from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ai.action_models import (
    ActionPlan,
    FollowUpDecision,
)
from memory.models import MemoryDecision
from session.decision import SessionDecision

load_dotenv()


# ==============================================================
# DEMO PROVIDER
# ==============================================================

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

    def decide_follow_up(
        self,
        user_text: str,
        conversation_context: str,
    ) -> bool:

        return False

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

    def process_passive_batch(
        self,
        transcript_batch: str,
    ) -> ActionPlan:

        return ActionPlan(
            actions=[],
            should_speak=False,
            response_instruction="",
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


# ==============================================================
# GEMINI PROVIDER
# ==============================================================

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
- For UPDATE_MEMORY, provide a query identifying
  the old memory and provide the corrected content.
- For DELETE_MEMORY, provide a query identifying
  the memory to remove.
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
    # FOLLOW-UP DETECTION
    # ==========================================================

    def decide_follow_up(
        self,
        user_text: str,
        conversation_context: str,
    ) -> bool:

        prompt = f"""
You are VEMORA's conversation-state classifier.

Determine whether the user's NEW utterance is a
follow-up to the recent direct conversation with VEMORA.

Return only the structured FollowUpDecision.

Rules:

- A follow-up may refer to something from the previous
  VEMORA answer using:
  "it", "that", "there", "this", "those",
  "who", "when", "where", "why", "how",
  "what about", "and where", "and when",
  "and what", etc.

- A short question that naturally continues the previous
  topic is a follow-up.

- Questions beginning with "and", "also", "what about",
  "where", "when", "who", "why", or "how" can be
  follow-ups when the previous conversation provides
  a clear topic.

Examples:

Previous:
VEMORA: It is 02:41 PM.

New:
"And what day is it today?"

This is TRUE.

Previous:
VEMORA: Your class is tomorrow at 8 AM.

New:
"And where is it?"

This is TRUE.

Previous:
VEMORA: The narrator and Gip entered the magic shop.

New:
"What happened next?"

This is TRUE.

- An unrelated statement or conversation is NOT a
  follow-up.

- Do not classify every short utterance as a follow-up.

Recent direct conversation:
{conversation_context}

New user utterance:
{user_text}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FollowUpDecision,
            ),
        )

        decision = FollowUpDecision.model_validate_json(
            response.text
        )

        return decision.is_follow_up

    # ==========================================================
    # PASSIVE SESSION PROCESSING
    # ==========================================================

    def process_passive_batch(
        self,
        transcript_batch: str,
    ) -> ActionPlan:

        prompt = f"""
You are VEMORA's passive-session processor.

The user has intentionally activated an ongoing
listening session.

The transcript below contains speech from that
session.

Your job is to identify information that should
be acted on silently.

- If the user says they completed, submitted, finished,
  sent, paid, attended, or otherwise completed something
  that clearly corresponds to an existing task, use
  complete_task.

Available tools:

1. save_memory
2. update_memory
3. delete_memory
4. create_task
5. complete_task

Do NOT use:
- search_session
- search_memory
- search_task
- delete_task

Task vs event rules:

- "I have a presentation on Monday."
  This is normally an EVENT or MEMORY.

- "I need to prepare my presentation by Monday."
  This is a TASK.

- "I have to submit my assignment by Friday."
  This is a TASK.

- "Remind me to call Rahul tomorrow."
  This is a TASK.

Memory rules:

- Do not save ordinary chatter.
- Do not create an action for every sentence.
- Prefer useful personal information, preferences,
  schedules, deadlines, commitments, important facts,
  and recurring information.
- Do not invent dates, times, people, or facts.
- Preserve uncertainty when the transcript is ambiguous.
- Estimate importance and confidence conservatively.
- Choose an appropriate retention policy.
- Never speak to the user from passive processing.
- should_speak MUST be false.

For create_task:

- title should clearly describe the action.
- description should contain useful additional context.
- due_at should contain the due date/time when known.
- expires_at should be set when the task should no longer
  remain relevant.

Transcript:
{transcript_batch}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ActionPlan,
            ),
        )

        plan = ActionPlan.model_validate_json(
            response.text
        )

        plan.should_speak = False

        return plan

    # ==========================================================
    # DIRECT ACTION PLANNER
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

AVAILABLE TOOLS:

1. search_session
   Search raw transcript chunks from the current session.
   Use for exact or highly specific details.

2. search_memory
   Search persistent long-term memory.

3. save_memory
   Save useful information for future recall.

4. update_memory
   Correct an existing memory.

5. delete_memory
   Forget an existing memory.

6. create_task
   Create something the user needs to do.

7. search_task
   Find an existing task.

8. complete_task
   Mark an existing task as completed.

9. delete_task
   Delete an existing task.

10. get_current_time
    Use when the user asks for the current time.

11. get_current_date
    Use when the user asks what day/date it is.

12. get_session_summary
    Retrieve the already-generated summary of the ENTIRE
    current session.

13. search_section_summaries
    Search summaries representing portions of the
    current session.

============================================================
HIERARCHICAL SESSION RETRIEVAL
============================================================

VEMORA has three levels of session information:

LEVEL 1:
search_session
    Raw transcript.
    Best for exact or very specific details.

LEVEL 2:
search_section_summaries
    Summaries of portions of the session.
    Best for broad topics, events, or parts of a
    long session.

LEVEL 3:
get_session_summary
    Summary of the ENTIRE session.
    Best for whole-session questions.

============================================================
WHOLE-SESSION QUESTIONS
============================================================

If the user asks to summarize, recap, overview,
or describe the ENTIRE current story/session/
lecture/meeting/seminar, use:

get_session_summary

Examples:

- "Summarize the story."
- "Summarize me the story."
- "Summarize the entire story."
- "Can you summarize the story?"
- "Can you summarize it?"
- "Give me a summary."
- "Give me an overview."
- "Give me an overview of the entire story."
- "Give me a recap."
- "Tell me the whole story."
- "What happened in the whole story?"

These are WHOLE-SESSION requests.

DO NOT use:
- search_task
- search_memory
- search_session

for a whole-session summary when the session
summary tool is available.

============================================================
SECTION QUESTIONS
============================================================

Use search_section_summaries for:

- "What happened in the middle?"
- "What happened in the first part?"
- "What happened near the end?"
- "What happened after they entered the shop?"
- "What was discussed about the project?"

============================================================
RAW DETAIL QUESTIONS
============================================================

Use search_session for:

- exact wording
- exact names
- exact statements
- specific details
- precise moments

Examples:

- "What exactly did the shopman say?"
- "What was written on the package?"
- "What exact name was mentioned?"

============================================================
IMPORTANT
============================================================

NEVER use search_task for a question about:
- a story
- a lecture
- a meeting
- a seminar
- the current session

unless the user is explicitly asking about a task.

NEVER return an empty action list if the user clearly
asks VEMORA to retrieve or summarize current session
information.

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

        plan = ActionPlan.model_validate_json(
            response.text
        )

        # ======================================================
        # DETERMINISTIC SESSION RETRIEVAL SAFEGUARD
        # ======================================================
        #
        # Gemini is still the planner, but obvious retrieval
        # requests are enforced here so a bad tool choice cannot
        # send the request to search_task or produce no action.
        # ======================================================

        retrieval_intent = (
            self._classify_session_retrieval_intent(
                user_text
            )
        )

        if retrieval_intent == "SESSION_SUMMARY":

            plan = self._force_retrieval_plan(
                original_plan=plan,
                tool="get_session_summary",
                query="",
            )

        elif retrieval_intent == "SECTION_SUMMARY":

            plan = self._force_retrieval_plan(
                original_plan=plan,
                tool="search_section_summaries",
                query=user_text,
            )

        return plan

    # ==========================================================
    # CLASSIFY SESSION RETRIEVAL INTENT
    # ==========================================================

    @staticmethod
    def _classify_session_retrieval_intent(
        user_text: str,
    ) -> str | None:
        """
        Deterministically identify obvious retrieval requests.

        Returns:
            SESSION_SUMMARY
            SECTION_SUMMARY
            None
        """

        text = (
            user_text
            .strip()
            .lower()
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # ------------------------------------------------------
        # Whole-session summary patterns.
        # ------------------------------------------------------

        whole_session_patterns = (
            "summarize the story",
            "summarise the story",
            "summarize me the story",
            "summarise me the story",
            "summarize this story",
            "summarise this story",
            "summarize me this story",
            "summarise me this story",
            "summarize the entire story",
            "summarise the entire story",
            "summarize the whole story",
            "summarise the whole story",

            "summarize the session",
            "summarise the session",
            "summarize the entire session",
            "summarise the entire session",
            "summarize the whole session",
            "summarise the whole session",

            "summary of the story",
            "summary of this story",
            "summary of the session",

            "give me a summary",
            "give me the summary",

            "give me an overview",
            "give me the overview",
            "give me an overview of the story",
            "give me an overview of the entire story",
            "give me an overview of this story",
            "give me an overview of the session",

            "give me a recap",
            "give me the recap",
            "recap the story",
            "recap this story",
            "recap the session",

            "tell me the whole story",
            "tell me about the whole story",
            "tell me what happened in the whole story",
            "what happened in the whole story",
        )

        if any(
            phrase in text
            for phrase in whole_session_patterns
        ):

            return "SESSION_SUMMARY"

        # ------------------------------------------------------
        # Conversational forms.
        # ------------------------------------------------------

        if (
            (
                "i said" in text
                or "i asked" in text
            )
            and (
                "summarize" in text
                or "summarise" in text
                or "summary" in text
            )
            and (
                "story" in text
                or "session" in text
                or "lecture" in text
                or "meeting" in text
                or "seminar" in text
            )
        ):

            return "SESSION_SUMMARY"

        # ------------------------------------------------------
        # Short conversational summary requests referring
        # to recent context.
        # ------------------------------------------------------

        if (
            (
                "summarize it" in text
                or "summarise it" in text
                or "summarize this" in text
                or "summarise this" in text
            )
        ):

            return "SESSION_SUMMARY"

        # ------------------------------------------------------
        # Section-level questions.
        # ------------------------------------------------------

        section_patterns = (
            "what happened in the middle",
            "what happened in the middle of the story",
            "what happened in the middle of the session",

            "what happened in the beginning",
            "what happened at the beginning",
            "what happened at the start",
            "what happened in the start",

            "what happened near the end",
            "what happened at the end",

            "what happened in the first part",
            "what happened in the second part",
            "what happened in the final part",

            "what happened earlier",
            "what happened later",

            "what was discussed in the middle",
            "what was discussed earlier",
            "what was discussed later",
        )

        if any(
            phrase in text
            for phrase in section_patterns
        ):

            return "SECTION_SUMMARY"

        return None

    # ==========================================================
    # FORCE RETRIEVAL PLAN
    # ==========================================================

    @staticmethod
    def _force_retrieval_plan(
        original_plan: ActionPlan,
        tool: str,
        query: str,
    ) -> ActionPlan:
        """
        Replace Gemini's retrieval choice with a deterministic,
        validated retrieval action.

        Other fields from the original plan are intentionally
        discarded because retrieval-only requests should have
        one clear action.
        """

        return ActionPlan.model_validate(
            {
                "actions": [
                    {
                        "tool": tool,
                        "query": query,
                        "content": "",
                        "title": "",
                        "description": "",
                        "memory_type": "general",
                        "importance": 0.5,
                        "confidence": 1.0,
                        "retention": "SHORT_TERM",
                        "due_at": "",
                        "expires_at": "",
                        "status": "PENDING",
                        "memory_id": None,
                        "task_id": None,
                    }
                ],
                "should_speak": True,
                "response_instruction": (
                    "Use the retrieved session evidence "
                    "to answer the user's request."
                ),
            }
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
- Do not mention tools, databases, prompts,
  or internal architecture.
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
- Do not repeat the user's question.
- For simple factual questions, answer in 1 sentence.
- Only give a long or detailed answer when the user
  explicitly asks for an explanation, details, steps,
  or a longer response.
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

    demo_mode = (
        os.getenv(
            "DEMO_MODE",
            "true",
        ).lower()
        == "true"
    )

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
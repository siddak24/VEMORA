from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ai.action_models import ActionPlan
from memory.models import MemoryDecision
from session.decision import SessionDecision
from ai.action_models import (
    ActionPlan,
    FollowUpDecision,
)

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
        """
        Decide whether the user's new utterance is a
        follow-up to the recent direct VEMORA conversation.
        """

        prompt = f"""
    You are VEMORA's conversation-state classifier.

    Determine whether the user's new utterance is a natural
    follow-up to the recent direct conversation with VEMORA.

    Return structured JSON.

    Rules:

    - A follow-up may refer to something from the previous
    VEMORA answer using words such as:
    "it", "that", "there", "who", "when", "what about",
    "and where", "and when", etc.

    - A short question that clearly continues the previous
    topic is a follow-up.

    - An unrelated statement or conversation is NOT a follow-up.

    - Do not assume every short utterance is a follow-up.

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
    # PASSIVE SESSION PROCESSING
    # ==========================================================

    def process_passive_batch(
        self,
        transcript_batch: str,
    ) -> ActionPlan:
        """
        Analyze passive session speech.

        The user is not directly asking VEMORA
        for a spoken response.

        The model may:
            - save useful memories
            - update memories
            - delete memories when explicitly requested
            - create tasks

        It must never speak.
        """

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

- For complete_task, provide a useful query describing
the completed task.

- Do not mark a task completed unless the transcript
clearly indicates completion.

Available tools:

1. save_memory
   Save useful information that may matter later.

2. update_memory
   Update an existing long-term memory when the
   transcript clearly corrects existing information.

3. delete_memory
   Delete a memory only when the user explicitly
   asks VEMORA to forget something.

4. create_task
   Create a task when the user expresses an action,
   obligation, or reminder that they need to perform.

5. complete_task
   Mark an existing task as completed when the user
   indicates that they have finished an outstanding task.

Do NOT use:
- search_session
- search_memory
- search_task
- delete_task

from passive processing.

Task vs event rules:

- "I have a presentation on Monday."
  This is normally an EVENT or MEMORY.

- "I need to prepare my presentation by Monday."
  This is a TASK.

- "I have to submit my assignment by Friday."
  This is a TASK.

- "Remind me to call Rahul tomorrow."
  This is a TASK.

- A task represents something the user needs to do.

- An event represents something that happens to
  the user.


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

        # Safety rule: passive processing never speaks.
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

Available tools:

1. search_session
   Search the current active session.

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

Time rules:

- Never guess the current time or date.
- For current time, use get_current_time.
- For current date/day, use get_current_date.
- These tools should be preferred over general model knowledge.

Task rules:

- "I have a presentation on Monday" is normally
  an EVENT or memory, not a task.

- "I need to prepare my presentation by Monday"
  is a TASK.

- "Remind me to prepare my presentation by Monday"
  is a TASK.

- "I need to submit my assignment by Friday"
  is a TASK.

- A task represents an action the user needs to perform.

- An event represents something that happens to the user.

- Use create_task only when the user expresses an
  action, obligation, or reminder.

- Use complete_task when the user says they finished
  an existing task.

- Use search_task before completing or deleting a task
  when the task ID is not known.

- Do not create duplicate tasks unnecessarily.

Memory rules:

- Use save_memory only when persistent information
  is genuinely useful.
- Use update_memory when the user is correcting
  something already remembered.
- Use delete_memory when the user explicitly asks
  VEMORA to forget something.

Retrieval rules:

- Use search_session when the answer may exist in
  the current session.
- Use search_memory when the answer may exist in
  long-term memory.
- You may request both searches when the source is unclear.
- Do not invent information.

For task dates:

- When an exact date/time is known, provide due_at in ISO format:
  YYYY-MM-DDTHH:MM:SS

- Do not use "10 PM" when an absolute date/time can be determined.
- Do not invent the date if it is unknown.

Response rules:

- If the user directly asks VEMORA something,
  should_speak should normally be true.
- Keep the action plan minimal.
- Only request tools that are actually useful.

For create_task:
- title
- description
- due_at
- expires_at

For complete_task:
- task_id if known
- otherwise query

For delete_task:
- task_id if known
- otherwise query

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

    def decide_follow_up(
            self,
            user_text: str,
            conversation_context: str,
        ) -> bool:
            """
            Decide whether the user's new utterance is a
            follow-up to the recent direct VEMORA conversation.
            """
    
            prompt = f"""
        You are VEMORA's conversation-state classifier.
    
        Determine whether the user's new utterance is a natural
        follow-up to the recent direct conversation with VEMORA.
    
        Return structured JSON.
    
        Rules:
    
        - A follow-up may refer to something from the previous
        VEMORA answer using words such as:
        "it", "that", "there", "who", "when", "what about",
        "and where", "and when", etc.
    
        - A short question that clearly continues the previous
        topic is a follow-up.
    
        - An unrelated statement or conversation is NOT a follow-up.
    
        - Do not assume every short utterance is a follow-up.
    
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
- Do not add unnecessary background or explanations.
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
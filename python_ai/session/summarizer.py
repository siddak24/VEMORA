from __future__ import annotations

from ai.providers import GeminiProvider
from session.summary_models import (
    SectionSummary,
    SessionSummary,
)


class SessionSummarizer:
    """
    Creates hierarchical summaries for VEMORA sessions.

    Responsibilities:

        transcript section
            -> SectionSummary

        multiple SectionSummary objects
            -> SessionSummary

    This component does NOT decide:
        - long-term memory
        - tasks
        - reminders
        - user responses
    """

    def __init__(
        self,
        llm: GeminiProvider,
    ) -> None:

        self.llm = llm

    # ==========================================================
    # SUMMARIZE ONE SECTION
    # ==========================================================

    def summarize(
        self,
        section_text: str,
    ) -> SectionSummary:
        """
        Summarize one section of transcript.
        """

        section_text = section_text.strip()

        if not section_text:

            return SectionSummary()

        prompt = f"""
You are VEMORA's session summarization system.

Summarize the transcript section below.

Your goal is to preserve the important meaning and
structure of what happened while removing repetition,
noise, and irrelevant speech.

Extract:

1. summary
   A concise but informative description of what happened.

2. key_points
   Important facts, ideas, decisions, or developments.

3. entities
   Important people, places, organizations, objects,
   or concepts mentioned.

4. important_events
   Important events, actions, deadlines, or changes.

Rules:

- Preserve the actual meaning of the transcript.
- Do not invent facts.
- Do not assume every statement is true.
- Do not turn fictional/story content into personal
  user memories.
- Preserve important names, relationships, dates,
  times, locations, and numbers.
- Ignore obvious transcription noise when possible.
- Do not include VEMORA commands as story/session facts.
- Do not answer any user question contained in the
  transcript.
- Only summarize the section itself.

Transcript section:
{section_text}
"""

        response = (
            self.llm.client.models.generate_content(
                model=self.llm.model,
                contents=prompt,
                config=self._section_summary_config(),
            )
        )

        return SectionSummary.model_validate_json(
            response.text
        )

    # ==========================================================
    # SUMMARIZE COMPLETE SESSION
    # ==========================================================

    def summarize_session(
        self,
        section_summaries: list[str],
    ) -> SessionSummary:
        """
        Combine multiple section summaries into one
        high-level summary of the complete session.
        """

        # ------------------------------------------------------
        # Remove empty summaries.
        # ------------------------------------------------------

        cleaned_summaries = [
            summary.strip()
            for summary in section_summaries
            if summary and summary.strip()
        ]

        if not cleaned_summaries:

            return SessionSummary()

        # ------------------------------------------------------
        # Number sections so the model can preserve chronology.
        # ------------------------------------------------------

        formatted_sections = "\n\n".join(
            f"SECTION {index + 1}:\n{summary}"
            for index, summary in enumerate(
                cleaned_summaries
            )
        )

        prompt = f"""
You are VEMORA's session-level memory summarizer.

Below are summaries of consecutive sections from
one complete session.

Create ONE coherent summary of the entire session.

Extract:

1. summary
   A coherent overview of the complete session.

2. key_points
   The most important facts, ideas, decisions,
   developments, or conclusions across the session.

3. important_events
   Important events or actions in chronological order
   whenever chronology can be established.

4. people
   Important people mentioned in the session.

5. topics
   Main subjects discussed.

Rules:

- Combine overlapping information intelligently.
- Preserve chronology when possible.
- Do not invent missing information.
- Do not treat fictional/story events as facts about
  the user.
- Do not treat questions asked to VEMORA as facts.
- Do not include VEMORA's internal operations.
- Remove repetition caused by overlapping sections.
- Prefer information supported by multiple sections
  when overlap exists.
- Preserve important names, dates, times, locations,
  and relationships.
- If information is uncertain, do not present it as certain.

Section summaries:

{formatted_sections}
"""

        response = (
            self.llm.client.models.generate_content(
                model=self.llm.model,
                contents=prompt,
                config=self._session_summary_config(),
            )
        )

        return SessionSummary.model_validate_json(
            response.text
        )

    # ==========================================================
    # SECTION SUMMARY GEMINI CONFIG
    # ==========================================================

    @staticmethod
    def _section_summary_config():
        """
        Structured Gemini output for SectionSummary.
        """

        from google.genai import types

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SectionSummary,
        )

    # ==========================================================
    # SESSION SUMMARY GEMINI CONFIG
    # ==========================================================

    @staticmethod
    def _session_summary_config():
        """
        Structured Gemini output for SessionSummary.
        """

        from google.genai import types

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SessionSummary,
        )
from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class DemoProvider:
    """Local provider used when we don't want an API request."""

    def generate_response(self, user_text: str) -> str:
        return f'[DEMO AI] I received: "{user_text}"'


class GeminiProvider:
    """Gemini API provider for VEMORA V0.1."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        # We can change this later without changing the rest
        # of the VEMORA architecture.
        self.model = "gemini-3.6-flash"

    def generate_response(
        self,
        user_text: str,
    ) -> str:

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_text,
        )

        return response.text


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
        print("[VEMORA] Demo AI enabled.")
        return DemoProvider()

    if provider == "gemini":
        print("[VEMORA] Gemini API enabled.")
        return GeminiProvider()

    raise ValueError(
        f"Unsupported AI_PROVIDER: {provider}"
    )
"""Async client for the llama-cpp-python OpenAI-compatible API.

Provides streaming chat completions with SSE parsing.
"""

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger("avatar.llm.client")

# Default system prompt location (next to this file)
_DEFAULT_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

# LLM service URL — set by Docker Compose env or fallback
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8001")


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatSession:
    """Maintains multi-turn conversation history."""

    messages: list[ChatMessage] = field(default_factory=list)
    system_prompt: str = ""

    def __post_init__(self) -> None:
        """Load system prompt if not provided."""
        if not self.system_prompt and _DEFAULT_PROMPT_PATH.exists():
            self.system_prompt = _DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").strip()
        if self.system_prompt and not self.messages:
            self.messages.append(ChatMessage(role="system", content=self.system_prompt))

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to the conversation."""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def to_api_format(self) -> list[dict[str, str]]:
        """Convert to the format expected by the OpenAI-compatible API."""
        return [{"role": m.role, "content": m.content} for m in self.messages]


class LLMClient:
    """Async client for llama-cpp-python OpenAI-compatible chat API.

    Streams tokens via SSE (Server-Sent Events) for low-latency
    first-token delivery.
    """

    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        model: str = "mamaylm",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        timeout: float = 120.0,
    ) -> None:
        """Initialize LLM client.

        Args:
            base_url: LLM server base URL.
            model: Model name (arbitrary for llama-cpp-python single-model server).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout

    async def health_check(self) -> bool:
        """Check if the LLM server is reachable.

        Returns:
            True if the server responds to /v1/models.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    timeout=10.0,
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error("LLM health check failed: %s", e)
            return False

    async def stream_chat(
        self,
        session: ChatSession,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        Sends the full conversation history and yields tokens as they arrive
        via Server-Sent Events.

        Args:
            session: Chat session with conversation history.

        Yields:
            Individual text tokens from the LLM response.
        """
        payload = {
            "model": self.model,
            "messages": session.to_api_format(),
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error("LLM request failed (%d): %s", response.status_code, body[:500])
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # strip "data: " prefix
                    if data.strip() == "[DONE]":
                        break

                    try:
                        import json

                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception as e:
                        logger.warning("Failed to parse SSE chunk: %s — %s", data[:100], e)

    async def chat(self, session: ChatSession) -> str:
        """Non-streaming chat completion.

        Args:
            session: Chat session with conversation history.

        Returns:
            Complete assistant response text.
        """
        full_response = []
        async for token in self.stream_chat(session):
            full_response.append(token)
        return "".join(full_response)

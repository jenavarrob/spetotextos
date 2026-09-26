import asyncio
import inspect
from typing import Any

import requests
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from app.models import Message
from ollama_client import create_ollama_client


class OllamaService:
    """Synchronous adapter around AutoGen's asynchronous Ollama client."""

    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")

    def check_connection(self, model: str | None = None) -> dict[str, Any]:
        """Check the Ollama HTTP API and whether the configured model exists."""
        requested_model = model or self.model
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {
                "connected": False,
                "host": self.host,
                "model": requested_model,
                "model_available": False,
                "error": str(exc),
            }

        models = payload.get("models", [])
        names = {item.get("name") for item in models if isinstance(item, dict)}
        available = requested_model in names or requested_model.removesuffix(":latest") in names
        return {
            "connected": True,
            "host": self.host,
            "model": requested_model,
            "model_available": available,
            "available_models": sorted(name for name in names if name),
            "error": None if available else f"Model {requested_model} is not installed",
        }

    def complete(self, messages: list[Message], *, model: str | None = None) -> Any:
        requested_model = model or self.model
        health = self.check_connection(requested_model)
        if not health["connected"]:
            raise RuntimeError(health["error"] or f"Cannot connect to {self.host}")
        if not health["model_available"]:
            raise RuntimeError(health["error"] or f"Model {requested_model} is not installed")
        try:
            # AutoGen's Ollama client owns an async HTTP client. Create and
            # close it inside the same event loop to avoid reusing a client
            # bound to an event loop that has already been closed.
            return asyncio.run(self._complete_async(messages, requested_model))
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed at {self.host} for model {requested_model}: {exc}"
            ) from exc

    async def _complete_async(self, messages: list[Message], model: str) -> Any:
        client = create_ollama_client(model=model, host=self.host)
        try:
            return await client.create(self._to_autogen_messages(messages))
        finally:
            close = getattr(client, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result

    @staticmethod
    def _to_autogen_messages(messages: list[Message]):
        converted = []
        for message in messages:
            if message.role == "system":
                converted.append(SystemMessage(content=message.content))
            elif message.role == "assistant":
                converted.append(AssistantMessage(content=message.content, source="assistant"))
            else:
                converted.append(UserMessage(content=message.content, source="user"))
        return converted

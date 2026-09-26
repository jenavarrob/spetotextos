"""Factory for the local Ollama model client.

The optional Ollama dependency is imported lazily so Azure-only deployments
can still import and test the application without installing local-LLM extras.
"""

from typing import Any


def create_ollama_client(
    model: str = "llama3.2:latest",
    host: str = "http://localhost:11434",
) -> Any:
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    return OllamaChatCompletionClient(
        model=model,
        host=host,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": "unknown",
        },
    )

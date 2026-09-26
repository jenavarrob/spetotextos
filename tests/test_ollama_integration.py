import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.factory import create_app
from app.models import Message
from app.services.application import ApplicationServices
from app.services.ollama import OllamaService


HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")


def ollama_service_or_skip():
    try:
        service = OllamaService(MODEL, HOST)
    except Exception as exc:
        pytest.skip(f"Ollama Python client is not installed: {exc}")
    health = service.check_connection()
    if not health["connected"]:
        pytest.skip(f"Ollama is not reachable: {health['error']}")
    if not health["model_available"]:
        pytest.skip(f"Ollama model is not installed: {health['error']}")
    return service


@pytest.mark.integration
def test_live_ollama_chat_completion():
    service = ollama_service_or_skip()
    result = service.complete([Message(role="user", content="Reply with exactly: ollama-ok")])
    assert isinstance(result.content, str)
    assert result.content.strip()


@pytest.mark.integration
def test_live_chabela_ollama_connection_and_chat():
    service = ollama_service_or_skip()
    settings = Settings(
        azure_openai_endpoint="https://openai.example",
        azure_openai_api_key="openai-key",
        azure_openai_core_model_deployment="chat",
        azure_search_endpoint="https://search.example",
        azure_search_admin_key="search-key",
        azure_storage_connection_string="UseDevelopmentStorage=true",
        blob_container_name="rag-data",
        azure_search_index_name="products",
        azure_search_indexer_name="products-indexer",
        azure_openai_speech_to_text_endpoint="https://speech.example",
        azure_openai_speech_to_text_key="speech-key",
        azure_openai_speech_to_text_deployment="speech",
        app_username="admin",
        app_password="secret",
        ollama_host=HOST,
        ollama_model=MODEL,
    )
    app_services = ApplicationServices(settings=settings, azure=object(), ollama=service)
    client = TestClient(create_app(settings=settings, services=app_services))

    health = client.get("/health/ollama", auth=("admin", "secret"))
    assert health.status_code == 200
    assert health.json()["connected"] is True
    assert health.json()["model_available"] is True

    chat = client.post(
        "/chat/user1",
        auth=("admin", "secret"),
        json={
            "provider": "ollama",
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: chabela-ollama-ok"}],
        },
    )
    assert chat.status_code == 200
    assert chat.json()["reply"].strip()

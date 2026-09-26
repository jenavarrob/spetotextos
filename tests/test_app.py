import base64

from fastapi.testclient import TestClient

from app.config import Settings, load_settings
from app.factory import create_app


class FakeResponse:
    def __init__(self, text):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": text})()})()]


class FakeTranscription:
    text = "transcribed text"


class FakeServices:
    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    def complete(self, messages, *, provider="azure", use_rag=False, model=None):
        if provider == "ollama" and use_rag:
            raise ValueError("Ollama is available for text-only chat; RAG uses Azure.")
        self.calls.append((messages, use_rag, provider, model))
        return FakeResponse("assistant reply")

    class FakeOllama:
        def check_connection(self):
            return {"connected": True, "host": "http://localhost:11434", "model": "llama3.2:latest", "model_available": True}

    ollama = FakeOllama()

    def describe_image(self, contents, content_type):
        return FakeResponse(f"described {content_type}")

    def transcribe(self, audio_file):
        return FakeTranscription()

    def upload_pdf(self, filename, contents):
        self.uploaded = (filename, contents)

    def run_indexer(self):
        self.indexer_started = True


def make_client():
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
    )
    services = FakeServices(settings)
    return TestClient(create_app(settings=settings, services=services)), services


def auth():
    token = base64.b64encode(b"admin:secret").decode()
    return {"Authorization": f"Basic {token}"}


def test_health_is_public():
    client, _ = make_client()
    assert client.get("/health").json() == {"status": "healthy"}


def test_local_settings_do_not_require_azure_environment(monkeypatch):
    azure_names = [
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CORE_MODEL_DEPLOYMENT",
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_ADMIN_KEY", "AZURE_STORAGE_CONNECTION_STRING",
        "BLOB_CONTAINER_NAME", "AZURE_SEARCH_INDEX_NAME", "AZURE_SEARCH_INDEXER_NAME",
        "AZURE_OPENAI_SPEECH_TO_TEXT_ENDPOINT", "AZURE_OPENAI_SPEECH_TO_TEXT_KEY",
        "AZURE_OPENAI_SPEECH_TO_TEXT_DEPLOYMENT",
    ]
    for name in azure_names:
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.azure_enabled is False
    assert settings.ollama_host == "http://localhost:11434"


def test_ollama_health_requires_authentication_and_reports_status():
    client, _ = make_client()
    assert client.get("/health/ollama").status_code == 401
    response = client.get("/health/ollama", headers=auth())
    assert response.json()["connected"] is True
    assert response.json()["model_available"] is True


def test_chat_is_authenticated_and_stateless():
    client, services = make_client()
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    assert client.post("/chat/user1", json=payload).status_code == 401
    response = client.post("/chat/user1", json=payload, headers=auth())
    assert response.json() == {"reply": "assistant reply"}
    assert services.calls[0][1] is False
    assert services.calls[0][2] == "azure"


def test_rag_chat_uses_search_and_full_history():
    client, services = make_client()
    payload = {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]}
    response = client.post("/chat/rag_user1", json=payload, headers=auth())
    assert response.status_code == 200
    assert services.calls[0][1] is True
    assert len(services.calls[0][0]) == 2


def test_ollama_provider_is_forwarded_for_text_chat():
    client, services = make_client()
    payload = {"messages": [{"role": "user", "content": "Hello locally"}], "provider": "ollama", "model": "llama3.2:latest"}
    response = client.post("/chat/user1", json=payload, headers=auth())
    assert response.status_code == 200
    assert services.calls[0][2:] == ("ollama", "llama3.2:latest")


def test_ollama_is_rejected_for_rag_chat():
    client, _ = make_client()
    payload = {"messages": [{"role": "user", "content": "Use local"}], "provider": "ollama"}
    response = client.post("/chat/rag_user1", json=payload, headers=auth())
    assert response.status_code == 400


def test_image_and_speech_do_not_create_server_session_state():
    client, _ = make_client()
    image = {"file": ("photo.png", b"image", "image/png")}
    assert client.post("/image-to-text/user1", files=image, headers=auth()).json() == {"reply": "described image/png"}
    audio = {"file": ("recording.webm", b"audio", "audio/webm")}
    assert client.post("/speech-to-text/user1", files=audio, headers=auth()).json() == {"transcription": "transcribed text"}


def test_pdf_upload_requires_authentication():
    client, services = make_client()
    pdf = {"file": ("document.pdf", b"pdf", "application/pdf")}
    assert client.post("/rag/upload", files=pdf).status_code == 401
    assert client.post("/rag/upload", files=pdf, headers=auth()).status_code == 200
    assert services.uploaded[0] == "document.pdf"

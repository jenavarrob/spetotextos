import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_core_model_deployment: str | None = None
    azure_search_endpoint: str | None = None
    azure_search_admin_key: str | None = None
    azure_storage_connection_string: str | None = None
    blob_container_name: str | None = None
    azure_search_index_name: str | None = None
    azure_search_indexer_name: str | None = None
    azure_openai_speech_to_text_endpoint: str | None = None
    azure_openai_speech_to_text_key: str | None = None
    azure_openai_speech_to_text_deployment: str | None = None
    app_username: str = Field(default="admin", min_length=1)
    app_password: str = Field(default="changeme", min_length=1)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    tesseract_cmd: str | None = None
    tesseract_language: str = "eng"
    search_api_version: str = "2025-09-01"

    @property
    def azure_enabled(self) -> bool:
        return all((
            self.azure_openai_endpoint,
            self.azure_openai_api_key,
            self.azure_openai_core_model_deployment,
            self.azure_search_endpoint,
            self.azure_search_admin_key,
            self.azure_storage_connection_string,
            self.blob_container_name,
            self.azure_search_index_name,
            self.azure_search_indexer_name,
            self.azure_openai_speech_to_text_endpoint,
            self.azure_openai_speech_to_text_key,
            self.azure_openai_speech_to_text_deployment,
        ))


def _optional(name: str) -> str | None:
    value = os.getenv(name)
    return value or None


def load_settings() -> Settings:
    """Load and validate deployment configuration at application startup."""
    return Settings(
        azure_openai_endpoint=_optional("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=_optional("AZURE_OPENAI_API_KEY"),
        azure_openai_core_model_deployment=_optional("AZURE_OPENAI_CORE_MODEL_DEPLOYMENT"),
        azure_search_endpoint=_optional("AZURE_SEARCH_ENDPOINT"),
        azure_search_admin_key=_optional("AZURE_SEARCH_ADMIN_KEY"),
        azure_storage_connection_string=_optional("AZURE_STORAGE_CONNECTION_STRING"),
        blob_container_name=_optional("BLOB_CONTAINER_NAME"),
        azure_search_index_name=_optional("AZURE_SEARCH_INDEX_NAME"),
        azure_search_indexer_name=_optional("AZURE_SEARCH_INDEXER_NAME"),
        azure_openai_speech_to_text_endpoint=_optional("AZURE_OPENAI_SPEECH_TO_TEXT_ENDPOINT"),
        azure_openai_speech_to_text_key=_optional("AZURE_OPENAI_SPEECH_TO_TEXT_KEY"),
        azure_openai_speech_to_text_deployment=_optional("AZURE_OPENAI_SPEECH_TO_TEXT_DEPLOYMENT"),
        app_username=os.getenv("APP_USERNAME", "admin"),
        app_password=os.getenv("APP_PASSWORD", "changeme"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
        tesseract_cmd=os.getenv("TESSERACT_CMD") or None,
        tesseract_language=os.getenv("TESSERACT_LANG", "eng"),
        search_api_version=os.getenv("SEARCH_API_VERSION", "2025-09-01"),
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()

import base64
from dataclasses import dataclass

import requests
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

from app.config import Settings
from app.models import Message


@dataclass
class AzureServices:
    settings: Settings
    chat_client: AzureOpenAI
    speech_client: AzureOpenAI
    blob_service: BlobServiceClient

    def complete(self, messages: list[Message], *, use_rag: bool):
        kwargs = {
            "model": self.settings.azure_openai_core_model_deployment,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0.3 if use_rag else 0.7,
            "max_tokens": 500,
        }
        if use_rag:
            kwargs["extra_body"] = {
                "data_sources": [{
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": self.settings.azure_search_endpoint,
                        "index_name": self.settings.azure_search_index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": self.settings.azure_search_admin_key,
                        },
                        "in_scope": True,
                    },
                }],
            }
        return self.chat_client.chat.completions.create(**kwargs)

    def describe_image(self, contents: bytes, content_type: str):
        encoded = base64.b64encode(contents).decode("ascii")
        return self.chat_client.chat.completions.create(
            model=self.settings.azure_openai_core_model_deployment,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image, keep it short."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{content_type};base64,{encoded}"
                    }},
                ],
            }],
            max_tokens=300,
        )

    def transcribe(self, audio_file):
        return self.speech_client.audio.transcriptions.create(
            model=self.settings.azure_openai_speech_to_text_deployment,
            file=audio_file,
            language="es",
            temperature=0.3,
        )

    def upload_pdf(self, filename: str, contents: bytes) -> None:
        blob = self.blob_service.get_blob_client(
            container=self.settings.blob_container_name,
            blob=filename,
        )
        blob.upload_blob(contents, overwrite=True)

    def run_indexer(self) -> None:
        url = (
            f"{self.settings.azure_search_endpoint}/indexers/"
            f"{self.settings.azure_search_indexer_name}/run"
            f"?api-version={self.settings.search_api_version}"
        )
        response = requests.post(
            url,
            headers={"api-key": self.settings.azure_search_admin_key},
            timeout=30,
        )
        response.raise_for_status()


def build_services(settings: Settings) -> AzureServices:
    if not settings.azure_enabled:
        raise RuntimeError("Azure services are not configured")
    return AzureServices(
        settings=settings,
        chat_client=AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version="2024-02-01",
            azure_endpoint=settings.azure_openai_endpoint,
        ),
        speech_client=AzureOpenAI(
            api_key=settings.azure_openai_speech_to_text_key,
            api_version="2025-03-01-preview",
            azure_endpoint=settings.azure_openai_speech_to_text_endpoint,
        ),
        blob_service=BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        ),
    )

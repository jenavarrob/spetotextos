from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.models import Message
from app.services.azure import AzureServices
from app.services.ollama import OllamaService
from app.services.ocr import TesseractService


@dataclass
class ApplicationServices:
    settings: Settings
    azure: AzureServices | None
    ollama: OllamaService
    tesseract: TesseractService = field(default_factory=TesseractService)

    def complete(
        self,
        messages: list[Message],
        *,
        provider: str = "azure",
        use_rag: bool = False,
        model: str | None = None,
    ) -> Any:
        if provider == "ollama":
            if use_rag:
                raise ValueError("Ollama is available for text-only chat; RAG uses Azure.")
            return self.ollama.complete(messages, model=model)
        if provider != "azure":
            raise ValueError(f"Unsupported chat provider: {provider}")
        if self.azure is None:
            raise RuntimeError("Azure services are not configured; use Ollama for local chat")
        return self.azure.complete(messages, use_rag=use_rag)

    def _require_azure(self) -> AzureServices:
        if self.azure is None:
            raise RuntimeError("Azure services are not configured")
        return self.azure

    def describe_image(self, contents: bytes, content_type: str):
        return self._require_azure().describe_image(contents, content_type)

    def extract_text(self, contents: bytes) -> str:
        return self.tesseract.extract_text(contents)

    def transcribe(self, audio_file):
        return self._require_azure().transcribe(audio_file)

    def upload_pdf(self, filename: str, contents: bytes) -> None:
        self._require_azure().upload_pdf(filename, contents)

    def run_indexer(self) -> None:
        self._require_azure().run_indexer()

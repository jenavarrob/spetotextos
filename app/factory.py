from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import Settings, load_settings
from app.routes import router
from app.services import ApplicationServices, AzureServices, OllamaService, build_services
from app.services.ocr import TesseractService


def create_app(
    settings: Settings | None = None,
    services: ApplicationServices | AzureServices | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or load_settings()
        app.state.services = services or ApplicationServices(
            settings=app.state.settings,
            azure=build_services(app.state.settings) if app.state.settings.azure_enabled else None,
            ollama=OllamaService(
                model=app.state.settings.ollama_model,
                host=app.state.settings.ollama_host,
            ),
            tesseract=TesseractService(
                command=app.state.settings.tesseract_cmd,
                language=app.state.settings.tesseract_language,
            ),
        )
        yield

    app = FastAPI(
        title="AI Product Finder API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.include_router(router)
    # This also makes dependency-injected test applications usable without
    # entering TestClient's lifespan context manager.
    if settings is not None:
        app.state.settings = settings
    if services is not None:
        app.state.services = services
    return app

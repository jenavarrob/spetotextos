import os
import tempfile
import logging
from contextlib import suppress

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse

from app.auth import authenticate
from app.models import ChatRequest


router = APIRouter()
logger = logging.getLogger(__name__)


def _service(request: Request):
    return request.app.state.services


def _completion_text(response) -> str:
    try:
        if hasattr(response, "content"):
            return response.content or ""
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as exc:
        raise HTTPException(502, "The selected chat provider returned an invalid response") from exc


@router.get("/health")
def health_check():
    return {"status": "healthy"}


@router.get("/health/ollama")
async def ollama_health(request: Request, _: str = Depends(authenticate)):
    return await run_in_threadpool(_service(request).ollama.check_connection)


@router.get("/")
def read_root(_: str = Depends(authenticate)):
    return FileResponse("static/index.html")


@router.get("/openapi.json")
def secure_openapi(request: Request, _: str = Depends(authenticate)):
    return JSONResponse(get_openapi(title=request.app.title, version="1.0.0", routes=request.app.routes))


@router.get("/docs")
def secure_docs(request: Request, _: str = Depends(authenticate)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{request.app.title} - Docs")


@router.post("/chat/{session_id}")
async def chat_completion(
    session_id: str,
    request_data: ChatRequest,
    request: Request,
    _: str = Depends(authenticate),
):
    use_rag = session_id.startswith("rag_")
    try:
        response = await run_in_threadpool(
            _service(request).complete,
            request_data.messages,
            provider=request_data.provider,
            model=request_data.model,
            use_rag=use_rag,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat completion failed for provider %s", request_data.provider)
        detail = str(exc) if request_data.provider == "ollama" else f"Unable to complete chat with {request_data.provider}"
        raise HTTPException(502, detail) from exc
    return {"reply": _completion_text(response)}


@router.delete("/chat/{session_id}")
async def delete_session(session_id: str, _: str = Depends(authenticate)):
    return {"message": f"Session {session_id} cleared."}


@router.post("/chat/{session_id}/reset")
async def reset_session(session_id: str, _: str = Depends(authenticate)):
    return {"message": f"Session {session_id} reset."}


@router.delete("/chat")
async def clear_all_sessions(_: str = Depends(authenticate)):
    return {"message": "All sessions cleared."}


@router.post("/rag/upload")
async def rag_upload(
    request: Request,
    file: UploadFile = File(...),
    _: str = Depends(authenticate),
):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF supported for this demo.")
    if not file.filename:
        raise HTTPException(400, "A PDF filename is required.")
    contents = await file.read()
    try:
        await run_in_threadpool(_service(request).upload_pdf, file.filename, contents)
        await run_in_threadpool(_service(request).run_indexer)
    except Exception as exc:
        raise HTTPException(502, "Unable to upload the PDF or start indexing") from exc
    return {"message": f"Uploaded {file.filename} to blob search_container {_service(request).settings.blob_container_name} and started indexing."}


@router.post("/image-to-text/{session_id}")
async def image_to_text(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
    mode: str = Form("ai"),
    _: str = Depends(authenticate),
):
    del session_id
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported.")
    contents = await file.read()
    if mode not in {"ai", "tesseract"}:
        raise HTTPException(400, "Image analysis mode must be 'ai' or 'tesseract'.")
    try:
        if mode == "tesseract":
            text = await run_in_threadpool(_service(request).extract_text, contents)
            return {"reply": text, "mode": "tesseract"}
        response = await run_in_threadpool(_service(request).describe_image, contents, file.content_type)
    except Exception as exc:
        if mode == "tesseract":
            raise HTTPException(
                502,
                "Local Tesseract OCR failed. Install Tesseract or configure TESSERACT_CMD. "
                f"Details: {exc}",
            ) from exc
        raise HTTPException(502, "Unable to describe the image") from exc
    return {"reply": _completion_text(response),}


@router.post("/speech-to-text/{session_id}")
async def speech_to_text(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
    _: str = Depends(authenticate),
):
    del session_id
    suffix = os.path.splitext(file.filename or "recording.webm")[1] or ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name
        with open(temp_path, "rb") as audio_file:
            transcription = await run_in_threadpool(_service(request).transcribe, audio_file)
        return {"transcription": transcription.text}
    except Exception as exc:
        raise HTTPException(502, "Unable to transcribe the audio") from exc
    finally:
        if temp_path:
            with suppress(FileNotFoundError):
                os.remove(temp_path)

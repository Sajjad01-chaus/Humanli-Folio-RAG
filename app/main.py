import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings
from app.container import Services, create_services
from app.errors import ApplicationError, DocumentError, PersistenceError
from app.models import (
    ChatResponse,
    DocumentRecord,
    HistoryResponse,
    IngestResult,
    SessionRecord,
    UploadError,
    UploadResponse,
)
from app.observability import configure_logging, event, failure, request_id

STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1500)
    document_ids: list[str] | None = Field(default=None, max_length=30)


def get_services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


AppServices = Annotated[Services, Depends(get_services)]


def require_session(services: Services, session_id: str) -> SessionRecord:
    session = services.repository.session(session_id)
    if session is None:
        raise HTTPException(404, "Conversation not found.")
    return session


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    configure_logging()
    container = services or create_services(settings or Settings())
    settings = container.settings
    application = FastAPI(title="Folio — Document Intelligence", version="0.1.0")
    application.state.services = container
    application.state.store = container.repository
    application.state.engine = container.engine

    @application.middleware("http")
    async def protect(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        correlation = uuid4().hex
        token = request_id.set(correlation)
        start = time.perf_counter()
        try:
            supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if (
                request.url.path.startswith("/api/")
                and settings.access_token
                and not secrets.compare_digest(supplied, settings.access_token)
            ):
                response: Response = JSONResponse(
                    {"detail": "Workspace access token required."}, status_code=401
                )
            else:
                try:
                    response = await call_next(request)
                except Exception as exc:
                    failure("request.unhandled_failure", exc)
                    response = JSONResponse(
                        {
                            "detail": "An unexpected application error occurred. Check the request ID in server logs.",
                            "code": "internal_error",
                            "request_id": correlation,
                        },
                        status_code=500,
                    )
            response.headers["X-Request-ID"] = correlation
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'self'"
            )
            event(
                "request.completed",
                method=request.method,
                route=getattr(request.scope.get("route"), "path", "unmatched"),
                status_code=response.status_code,
                duration_ms=round((time.perf_counter() - start) * 1000),
            )
            return response
        finally:
            request_id.reset(token)

    @application.exception_handler(ApplicationError)
    async def application_error(request: Request, exc: ApplicationError) -> JSONResponse:
        event("request.dependency_failed", level=logging.ERROR, code=exc.code, error_type=type(exc).__name__)
        return JSONResponse(
            {"detail": str(exc), "code": exc.code, "request_id": request_id.get()},
            status_code=exc.status_code,
        )

    @application.get("/api/health")
    def health(services: AppServices) -> dict[str, str | bool | int]:
        config = services.settings
        return {
            "status": "ok",
            "llm_configured": services.engine.llm.enabled,
            "provider": config.provider,
            "model": config.model,
            "embedding_backend": config.embeddings,
            "documents": len(services.repository.documents()),
            "scope": "single-user workspace",
            "provider_connectivity": "not checked",
        }

    @application.get("/api/documents")
    def documents(services: AppServices) -> list[DocumentRecord]:
        return services.repository.documents()

    @application.post("/api/documents")
    def upload(services: AppServices, files: list[UploadFile] = File(...)) -> UploadResponse:
        if len(files) > 10:
            raise HTTPException(400, "Upload at most 10 PDFs at a time.")
        results: list[IngestResult] = []
        errors: list[UploadError] = []
        for file in files:
            try:
                data = file.file.read(services.settings.max_upload_bytes + 1)
                results.append(services.documents.ingest(file.filename or "document.pdf", data))
            except DocumentError as exc:
                event("document.rejected", error_type=type(exc).__name__)
                errors.append({"name": file.filename or "document.pdf", "error": str(exc)})
            except OSError as exc:
                failure("document.upload_read_failed", exc)
                raise PersistenceError("Cannot read the uploaded file. Retry the upload.") from exc
        return {"documents": results, "errors": errors}

    @application.delete("/api/documents/{document_id}")
    def delete_document(document_id: str, services: AppServices) -> dict[str, str]:
        with services.documents.lock:
            if document_id not in {d["id"] for d in services.repository.documents()}:
                raise HTTPException(404, "Document not found.")
            services.repository.delete_document(document_id)
        event("document.deleted", document_id=document_id)
        return {"deleted": document_id}

    @application.get("/api/documents/{document_id}/pdf")
    def pdf(document_id: str, services: AppServices) -> FileResponse:
        if document_id not in {d["id"] for d in services.repository.documents()}:
            raise HTTPException(404, "Document not found.")
        path = services.repository.root / "uploads" / f"{document_id}.pdf"
        try:
            if not path.is_file():
                raise PersistenceError("The stored PDF is missing. Remove the document and upload it again.")
        except OSError as exc:
            failure("document.download_failed", exc)
            raise PersistenceError("Cannot access the stored PDF. Check file permissions and retry.") from exc
        return FileResponse(path, media_type="application/pdf")

    @application.post("/api/sessions")
    def create_session(services: AppServices) -> SessionRecord:
        return services.repository.create_session()

    @application.get("/api/sessions")
    def sessions(services: AppServices) -> list[SessionRecord]:
        return services.repository.sessions()

    @application.get("/api/sessions/{session_id}")
    def history(session_id: str, services: AppServices) -> HistoryResponse:
        return {
            "session": require_session(services, session_id),
            "turns": services.repository.history(session_id),
        }

    @application.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str, services: AppServices) -> dict[str, str]:
        require_session(services, session_id)
        services.engine.delete_session(session_id)
        return {"deleted": session_id}

    @application.post("/api/sessions/{session_id}/chat")
    def chat(session_id: str, body: Question, services: AppServices) -> ChatResponse:
        require_session(services, session_id)
        if not body.question.strip():
            raise HTTPException(422, "Enter a question.")
        if body.document_ids is not None:
            known = {d["id"] for d in services.repository.documents()}
            if any(i not in known for i in body.document_ids):
                raise HTTPException(404, "One of the selected documents no longer exists.")
        return services.engine.ask(session_id, body.question.strip(), body.document_ids)

    @application.post("/api/demo")
    def demo(services: AppServices, corpus: str = "synthetic") -> dict[str, list[IngestResult]]:
        if corpus not in {"synthetic", "public"}:
            raise HTTPException(400, "Unknown sample corpus.")
        sample_dir = Path(__file__).parent.parent / "samples" / ("public" if corpus == "public" else "pdfs")
        paths = sorted(sample_dir.glob("*.pdf"))
        if not paths:
            raise HTTPException(
                404, "Sample PDFs are missing. Download or regenerate the selected sample corpus."
            )
        results: list[IngestResult] = []
        for path in paths:
            try:
                results.append(services.documents.ingest(path.name, path.read_bytes()))
            except DocumentError as exc:
                raise HTTPException(422, str(exc)) from exc
            except OSError as exc:
                failure("document.sample_read_failed", exc)
                raise PersistenceError("Cannot read the sample PDFs. Check local file permissions.") from exc
        return {"documents": results}

    application.mount("/static", StaticFiles(directory=STATIC), name="static")

    @application.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return application


app = create_app()

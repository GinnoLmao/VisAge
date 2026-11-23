"""Memory-only raw-image HTTP API; models are loaded once per server process."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
import logging
import math
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect

from .faces import FaceError
from .runtime import configure_caches, seed_everything
from .service import InferenceService, ROOT
from .model_registry import ARTIFACTS, EVALUATION


LOGGER = logging.getLogger("visage.api")
from .image_formats import IMAGE_TYPES, FORMAT_DESCRIPTION


@dataclass(frozen=True)
class APISettings:
    artifacts_dir: Path = field(default_factory=lambda: ARTIFACTS)
    detector_path: Path | None = None
    device: str = "cpu"
    max_pending: int = 2
    queue_timeout_seconds: float = 15.0
    upload_timeout_seconds: float = 15.0
    allowed_origins: tuple[str, ...] = ()
    evaluation_dir: Path = EVALUATION
    max_upload_bytes: int | None = None

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda.")
        if self.max_upload_bytes is not None and (type(self.max_upload_bytes) is not int or not 32 <= self.max_upload_bytes <= 8 * 1024**2):
            raise ValueError("Upload override must be between 32 bytes and 8 MiB.")
        if self.max_pending < 0 or self.max_pending > 16:
            raise ValueError("max_pending must be between zero and sixteen.")
        for duration in (self.queue_timeout_seconds, self.upload_timeout_seconds):
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Timeouts must be finite and positive.")
        if "*" in self.allowed_origins:
            raise ValueError("Use explicit allowed origins, not a wildcard.")


def create_app(settings: APISettings | None = None, *, service_factory=None) -> FastAPI:
    settings = settings or APISettings()

    def load_service():
        configure_caches(ROOT)
        seed_everything(42, threads=4)
        return InferenceService(settings.artifacts_dir, device=settings.device, detector_path=settings.detector_path)

    factory = service_factory or load_service

    @asynccontextmanager
    async def lifespan(application):
        application.state.service = await run_in_threadpool(factory)
        try:
            yield
        finally:
            application.state.service.close()
            application.state.service = None

    application = FastAPI(title="VisAge model API", version="0.4.0", lifespan=lifespan,
                          description=f"Send raw {FORMAT_DESCRIPTION} bytes, not multipart form data. Images are never saved.")
    application.state.service = None
    application.state.admitted = 0
    application.state.work_lock = asyncio.Lock()
    if settings.allowed_origins:
        application.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
                                   expose_headers=["X-Request-ID"], allow_credentials=False)

    def error_response(request, status, code, message, details=None):
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(status_code=status, content={"error": {
            "code": code, "message": message, "details": details or {},
            "request_id": request_id}}, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                                 "X-Request-ID": request_id or ""})

    @application.middleware("http")
    async def response_policy(request, call_next):
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @application.exception_handler(FaceError)
    async def face_error(request, exc):
        return error_response(request, exc.status_code, exc.code, exc.message, exc.details)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo supplied image/query contents in error responses or logs.
        details = {"fields": [{"location": list(item["loc"]), "type": item["type"]} for item in exc.errors()]}
        return error_response(request, 422, "INVALID_REQUEST", "Check the request parameters.", details)

    @application.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(request, exc.status_code, "HTTP_ERROR", "Request could not be handled.")

    @application.exception_handler(Exception)
    async def unexpected_error(request, exc):
        LOGGER.error("Inference request failed: request_id=%s error_type=%s", request.state.request_id, type(exc).__name__)
        return error_response(request, 500, "INFERENCE_FAILED", "Image processing failed. Try again or contact the operator.")

    def service_or_error():
        service = application.state.service
        if service is None or not service.ready:
            raise FaceError("NOT_READY", "Models are not ready.", status_code=503)
        return service

    @application.get("/health")
    async def health():
        service = service_or_error()
        return {"status": "ready", "models_loaded": len(service.predictors), "device": service.device}

    @application.get("/v1/models")
    async def models():
        return service_or_error().describe()

    @application.get("/v1/evaluation")
    async def evaluation():
        from .evaluation_registry import evaluation_snapshot
        try:
            return await run_in_threadpool(evaluation_snapshot, settings.evaluation_dir, service_or_error())
        except (ValueError, KeyError, OSError) as exc:
            raise FaceError("EVALUATION_UNAVAILABLE", "Evaluation records are unavailable or do not match the served models.", status_code=503) from exc

    async def read_image(request, limit):
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in IMAGE_TYPES:
            raise FaceError("UNSUPPORTED_MEDIA_TYPE", f"Send raw {FORMAT_DESCRIPTION} image bytes with an image Content-Type.", status_code=415)
        if request.headers.get("content-encoding", "identity").lower() != "identity":
            raise FaceError("UNSUPPORTED_CONTENT_ENCODING", "Send uncompressed image request bytes.", status_code=415)
        length = request.headers.get("content-length")
        if length is not None:
            try:
                numeric_length = int(length)
            except ValueError:
                raise FaceError("INVALID_CONTENT_LENGTH", "Invalid Content-Length.", status_code=400) from None
            if numeric_length < 0:
                raise FaceError("INVALID_CONTENT_LENGTH", "Invalid Content-Length.", status_code=400)
            if numeric_length > limit:
                raise FaceError("IMAGE_TOO_LARGE", f"Image exceeds the {limit / 1024**2:g} MiB input limit.", status_code=413)
        buffer = bytearray()
        try:
            async with asyncio.timeout(settings.upload_timeout_seconds):
                async for chunk in request.stream():
                    if len(buffer) + len(chunk) > limit:
                        raise FaceError("IMAGE_TOO_LARGE", f"Image exceeds the {limit / 1024**2:g} MiB input limit.", status_code=413)
                    buffer.extend(chunk)
            if not buffer:
                raise FaceError("EMPTY_IMAGE", "Upload an image.", status_code=400)
            return bytes(buffer)
        except TimeoutError:
            raise FaceError("UPLOAD_TIMEOUT", "Image upload timed out.", status_code=408) from None
        except ClientDisconnect:
            raise FaceError("UPLOAD_DISCONNECTED", "Image upload was interrupted.", status_code=400) from None
        finally:
            buffer.clear()

    async def execute(request, operation, **kwargs):
        service = service_or_error()
        # Admission occurs before reading image bytes. At most one active and
        # max_pending requests can hold images or wait for the model lock.
        if application.state.admitted >= settings.max_pending + 1:
            raise FaceError("SERVER_BUSY", "The model queue is full. Try again shortly.", status_code=503)
        application.state.admitted += 1
        acquired = False
        data = None
        started = time.perf_counter()
        try:
            data = await read_image(request, min(service.config.max_bytes, settings.max_upload_bytes or service.config.max_bytes))
            uploaded = time.perf_counter()
            try:
                await asyncio.wait_for(application.state.work_lock.acquire(), timeout=settings.queue_timeout_seconds)
                acquired = True
            except TimeoutError:
                raise FaceError("QUEUE_TIMEOUT", "The model queue timed out. Try again shortly.", status_code=503) from None
            running = time.perf_counter()
            # Shield the executor future and finish it even if the HTTP task is
            # cancelled. Releasing this lock while native inference still runs
            # could let another request mutate YuNet's input shape concurrently.
            worker = asyncio.get_running_loop().run_in_executor(
                None, partial(getattr(service, operation), data, **kwargs))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                except BaseException:
                    if cancelled:
                        raise asyncio.CancelledError from None
                    raise
            if cancelled:
                raise asyncio.CancelledError
            result["request_id"] = request.state.request_id
            result.setdefault("timings_ms", {}).update({
                "upload": (uploaded - started) * 1000,
                "queue": (running - uploaded) * 1000,
                "request_total": (time.perf_counter() - started) * 1000,
            })
            return result
        finally:
            data = None
            if acquired:
                application.state.work_lock.release()
            application.state.admitted -= 1

    # Expose raw body schemas in OpenAPI while reading streams ourselves.
    image_body = {"requestBody": {"required": True, "content": {
        kind: {"schema": {"type": "string", "format": "binary"}} for kind in sorted(IMAGE_TYPES - {"application/octet-stream"})}}}

    @application.post("/v1/faces", openapi_extra=image_body)
    async def faces(request: Request):
        return await execute(request, "faces")

    @application.post("/v1/predict", openapi_extra=image_body)
    async def predict(request: Request,
                      method: Literal["all", "convnextv2_nano", "efficientnetv2_s", "swinv2_tiny"] | None = None,
                      face_index: int | None = Query(default=None, ge=0)):
        selected = method if method is not None else service_or_error().describe()["default_method"]
        return await execute(request, "predict", method=selected, face_index=face_index)

    from .web_ui import install_web_routes
    install_web_routes(application)
    return application


app = create_app()

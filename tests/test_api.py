"""HTTP admission, lifetime, structured errors and serialization checks."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest

from visage.api import APISettings, create_app
from visage.faces import FaceError


class StubService:
    ready = True
    device = "cpu"
    config = SimpleNamespace(max_bytes=16)
    predictors = {name: None for name in ("efficientnetv2_s", "convnextv2_nano", "swinv2_tiny")}

    def __init__(self):
        self.calls = []
        self.closed = False

    def close(self):
        self.closed = True
        self.ready = False

    def describe(self):
        return {"default_method": "convnextv2_nano", "methods": list(self.predictors)}

    def predict(self, data, *, method, face_index):
        self.calls.append((data, method, face_index))
        if data == b"no-face":
            raise FaceError("NO_FACE", "No face was detected.")
        if data == b"multiple":
            raise FaceError("MULTIPLE_FACES", "Choose one face.", details={"faces": [{"index": 0}, {"index": 1}]})
        if data == b"broken":
            raise RuntimeError("secret internal failure")
        return {"predictions": [{"method": method, "class_id": 1}], "timings_ms": {"processing_total": 1.0}}

    def faces(self, data):
        self.calls.append((data, "faces", None))
        return {"faces": [{"index": 0}], "selection_required": False}


def test_lifespan_loads_once_reuses_service_and_closes():
    loaded = []
    service = StubService()

    def factory():
        loaded.append(True)
        return service

    app = create_app(service_factory=factory)
    assert not loaded
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ready", "models_loaded": 3, "device": "cpu"}
        assert client.get("/v1/models").json()["default_method"] == "convnextv2_nano"
        for _ in range(2):
            response = client.post("/v1/predict", content=b"valid", headers={"Content-Type": "image/png"})
            assert response.status_code == 200
            result = response.json()
            assert result["predictions"][0]["method"] == "convnextv2_nano"
            assert result["request_id"] == response.headers["X-Request-ID"]
            assert response.headers["Cache-Control"] == "no-store"
            assert all(value >= 0 for value in result["timings_ms"].values())
        response = client.post("/v1/faces", content=b"valid", headers={"Content-Type": "image/png"})
        assert response.status_code == 200 and response.json()["faces"] == [{"index": 0}]
        assert len(loaded) == 1 and len(service.calls) == 3
    assert service.closed and app.state.service is None


def test_evaluation_endpoint_is_readonly_and_pending_without_records(tmp_path):
    service = StubService()
    app = create_app(APISettings(evaluation_dir=tmp_path), service_factory=lambda: service)
    with TestClient(app) as client:
        response = client.get("/v1/evaluation")
        assert response.status_code == 200
        assert response.json()["classification"]["status"] == "pending"
        assert response.json()["detection"]["status"] == "pending"
        assert not service.calls
        assert response.headers["Cache-Control"] == "no-store"


def test_evaluation_endpoint_rejects_mismatched_records(tmp_path):
    import json
    service = StubService()
    service.manifest_sha256 = "a" * 64
    (tmp_path / "classification_roc.json").write_text(json.dumps({"ok": True, "manifest_sha256": "b" * 64}))
    app = create_app(APISettings(evaluation_dir=tmp_path), service_factory=lambda: service)
    with TestClient(app) as client:
        response = client.get("/v1/evaluation")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "EVALUATION_UNAVAILABLE"
        assert not service.calls


@pytest.mark.parametrize("data,headers,status,code", [
    (b"abc", {}, 415, "UNSUPPORTED_MEDIA_TYPE"),
    (b"abc", {"Content-Type": "multipart/form-data"}, 415, "UNSUPPORTED_MEDIA_TYPE"),
    (b"abc", {"Content-Type": "image/png", "Content-Encoding": "gzip"}, 415, "UNSUPPORTED_CONTENT_ENCODING"),
    (b"", {"Content-Type": "image/jpeg"}, 400, "EMPTY_IMAGE"),
    (b"a" * 17, {"Content-Type": "image/png"}, 413, "IMAGE_TOO_LARGE"),
    (b"a", {"Content-Type": "image/png", "Content-Length": "invalid"}, 400, "INVALID_CONTENT_LENGTH"),
    (b"a", {"Content-Type": "image/png", "Content-Length": "-1"}, 400, "INVALID_CONTENT_LENGTH"),
    (b"no-face", {"Content-Type": "image/png"}, 422, "NO_FACE"),
    (b"multiple", {"Content-Type": "image/png"}, 422, "MULTIPLE_FACES"),
])
def test_structured_errors_and_no_model_call_on_admission_errors(data, headers, status, code):
    service = StubService()
    with TestClient(create_app(service_factory=lambda: service)) as client:
        response = client.post("/v1/predict", content=data, headers=headers)
        assert response.status_code == status
        error = response.json()["error"]
        assert error["code"] == code and error["request_id"]
        if status in {400, 413, 415}:
            assert not service.calls
        # Rejections release the queue slot for later requests.
        assert client.post("/v1/predict", content=b"valid", headers={"Content-Type": "image/png"}).status_code == 200


def test_streamed_byte_limit_is_enforced_without_content_length():
    service = StubService()
    with TestClient(create_app(service_factory=lambda: service)) as client:
        response = client.post("/v1/predict", content=iter([b"a" * 10, b"b" * 10]),
                               headers={"Content-Type": "image/png"})
        assert response.status_code == 413 and not service.calls


def test_parameter_validation_and_safe_internal_error():
    service = StubService()
    with TestClient(create_app(service_factory=lambda: service), raise_server_exceptions=False) as client:
        for query in ("method=invalid", "face_index=-1", "face_index=not-integer"):
            response = client.post("/v1/predict?" + query, content=b"valid", headers={"Content-Type": "image/png"})
            assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_REQUEST"
        response = client.post("/v1/predict?method=all&face_index=1", content=b"valid", headers={"Content-Type": "image/png"})
        assert response.status_code == 200 and service.calls[-1][1:] == ("all", 1)
        failed = client.post("/v1/predict", content=b"broken", headers={"Content-Type": "image/png"})
        assert failed.status_code == 500 and "secret" not in failed.text
        assert failed.json()["error"]["code"] == "INFERENCE_FAILED"


def test_cors_requires_explicit_origins_and_defaults_closed():
    with pytest.raises(ValueError, match="wildcard"):
        APISettings(allowed_origins=("*",))
    for settings, allowed in ((APISettings(), False), (APISettings(allowed_origins=("http://localhost:5173",)), True)):
        with TestClient(create_app(settings, service_factory=StubService)) as client:
            response = client.get("/health", headers={"Origin": "http://localhost:5173"})
            assert (response.headers.get("access-control-allow-origin") == "http://localhost:5173") is allowed


def test_queue_capacity_and_serialized_inference():
    started, release = Event(), Event()
    state_lock = Lock()
    service = StubService()
    service.active = service.max_active = 0
    original = service.predict

    def blocking(data, **kwargs):
        with state_lock:
            service.active += 1
            service.max_active = max(service.max_active, service.active)
        started.set()
        assert release.wait(5)
        try:
            return original(data, **kwargs)
        finally:
            with state_lock:
                service.active -= 1

    service.predict = blocking
    app = create_app(APISettings(max_pending=0), service_factory=lambda: service)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.post, "/v1/predict", content=b"valid", headers={"Content-Type": "image/png"})
        assert started.wait(5)
        busy = client.post("/v1/predict", content=b"valid", headers={"Content-Type": "image/png"})
        assert busy.status_code == 503 and busy.json()["error"]["code"] == "SERVER_BUSY"
        release.set()
        assert first.result(timeout=5).status_code == 200
        assert service.max_active == 1 and app.state.admitted == 0


def test_waiting_request_times_out_without_starting_second_inference():
    started, release = Event(), Event()
    service = StubService()
    original = service.predict

    def blocking(data, **kwargs):
        started.set()
        assert release.wait(5)
        return original(data, **kwargs)

    service.predict = blocking
    app = create_app(APISettings(max_pending=1, queue_timeout_seconds=0.05), service_factory=lambda: service)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.post, "/v1/predict", content=b"valid", headers={"Content-Type": "image/png"})
        assert started.wait(5)
        waiting = client.post("/v1/predict", content=b"valid", headers={"Content-Type": "image/png"})
        assert waiting.status_code == 503 and waiting.json()["error"]["code"] == "QUEUE_TIMEOUT"
        release.set()
        assert first.result(timeout=5).status_code == 200
        assert len(service.calls) == 1 and app.state.admitted == 0


def test_not_ready_and_startup_failure_are_not_silently_accepted():
    app = create_app(service_factory=StubService)
    # ASGI callers which fail to run lifespan see an explicit readiness error.
    with TestClient(app) as client:
        app.state.service.ready = False
        assert client.get("/health").status_code == 503
    def broken_factory():
        raise ValueError("missing artifacts")
    with pytest.raises(ValueError, match="missing artifacts"):
        with TestClient(create_app(service_factory=broken_factory)):
            pass


def test_cancelled_request_keeps_model_lock_until_native_worker_finishes():
    started, release = Event(), Event()
    service = StubService()
    original = service.predict

    def blocking(data, **kwargs):
        started.set()
        assert release.wait(5)
        return original(data, **kwargs)

    service.predict = blocking
    app = create_app(APISettings(max_pending=0), service_factory=lambda: service)

    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/v1/predict", content=b"valid", headers={"Content-Type": "image/png"}))
                assert await asyncio.to_thread(started.wait, 5)
                first.cancel()
                await asyncio.sleep(0.02)
                assert not first.done()
                assert app.state.admitted == 1 and app.state.work_lock.locked()
                busy = await client.post("/v1/predict", content=b"valid", headers={"Content-Type": "image/png"})
                assert busy.status_code == 503
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert app.state.admitted == 0 and not app.state.work_lock.locked()
                assert (await client.post("/v1/predict", content=b"valid", headers={"Content-Type": "image/png"})).status_code == 200
    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_upload_timeout_does_not_leave_a_queue_slot_or_start_inference():
    service = StubService()
    app = create_app(APISettings(upload_timeout_seconds=0.02), service_factory=lambda: service)

    async def chunks():
        yield b"a"
        await asyncio.sleep(0.1)
        yield b"b"

    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/v1/predict", content=chunks(), headers={"Content-Type": "image/png"})
                assert response.status_code == 408 and response.json()["error"]["code"] == "UPLOAD_TIMEOUT"
                assert not service.calls and app.state.admitted == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("kwargs", [{"device": "invalid"}, {"max_pending": -1}, {"queue_timeout_seconds": 0},
                                    {"upload_timeout_seconds": float("nan")}])
def test_invalid_settings_fail_before_server_start(kwargs):
    with pytest.raises(ValueError):
        APISettings(**kwargs)


@pytest.mark.parametrize("legacy_method", ["classical", "visagenet", "convnext_ordinal"])
def test_legacy_method_ids_are_rejected_without_running_a_different_model(legacy_method):
    service = StubService()
    with TestClient(create_app(service_factory=lambda: service)) as client:
        response = client.post(f"/v1/predict?method={legacy_method}", content=b"valid", headers={"Content-Type": "image/png"})
        assert response.status_code == 422
        assert not service.calls


def test_hosted_upload_override_rejects_before_inference_and_keeps_local_limit():
    service = StubService()
    service.config = SimpleNamespace(max_bytes=64)
    with TestClient(create_app(APISettings(max_upload_bytes=32), service_factory=lambda: service)) as client:
        rejected = client.post("/v1/predict", content=b"x" * 33, headers={"Content-Type":"image/jpeg"})
        assert rejected.status_code == 413 and rejected.json()["error"]["code"] == "IMAGE_TOO_LARGE"
        assert service.calls == []
        assert client.post("/v1/predict", content=b"x" * 31, headers={"Content-Type":"image/jpeg"}).status_code == 200
    assert APISettings().max_upload_bytes is None

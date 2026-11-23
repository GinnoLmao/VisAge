"""Isolated browser QA server. Never imported by the production application.

The fixture must be a local validation photo, not a private camera image.
Run from the repository root: python tests/web/serve_camera_qa.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from visage.api import create_app
from visage.web_ui import PAGE_HEADERS, WEB_ROOT


def create_qa_app():
    app = create_app()
    events = []

    @app.middleware("http")
    async def qa_shell(request: Request, call_next):
        if request.url.path == "/":
            html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
            html = html.replace('<script type="module" src="/assets/app.js">',
                                '<link rel="stylesheet" href="/__qa__/camera.css">'
                                '<script type="module" src="/__qa__/camera.js"></script>'
                                '<script type="module" src="/assets/app.js">')
            return HTMLResponse(html, headers=PAGE_HEADERS)
        return await call_next(request)

    @app.get("/__qa__/camera.js")
    def camera_script():
        return FileResponse(Path(__file__).with_name("browser-camera.js"), media_type="text/javascript")

    @app.get("/__qa__/camera.css")
    def camera_style():
        return FileResponse(Path(__file__).with_name("browser-camera.css"), media_type="text/css")

    @app.get("/__qa__/upload-tests.js")
    def upload_tests():
        return FileResponse(Path(__file__).with_name("browser-upload.js"), media_type="text/javascript")

    @app.get("/__qa__/fixture.jpg")
    def fixture():
        return FileResponse(ROOT / ".cache/web_qa/single.jpg", media_type="image/jpeg")

    @app.post("/__qa__/events")
    async def add_event(request: Request):
        # Diagnostics only: no image bytes, filename, or participant information.
        record = await request.json()
        if isinstance(record, dict) and len(events) < 2000:
            events.append(record)
        return {"ok": True}

    @app.get("/__qa__/fixture.heic")
    def heic_fixture():
        return FileResponse(ROOT / ".cache/web_qa/single.heic", media_type="image/heic")

    @app.get("/__qa__/events")
    def get_events():
        return events

    return app


if __name__ == "__main__":
    uvicorn.run(create_qa_app(), host="127.0.0.1", port=8001, access_log=False)

"""Self-contained browser assets served alongside the local inference API."""
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WEB_ROOT = Path(__file__).with_name("web")
PAGE_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}


def install_web_routes(application):
    application.mount("/assets", StaticFiles(directory=WEB_ROOT), name="web_assets")

    @application.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def web_application():
        return FileResponse(WEB_ROOT / "index.html", media_type="text/html", headers=PAGE_HEADERS)

"""Static browser shell, same-origin routing and privacy policy checks."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from visage.api import create_app


class Service:
    ready = True
    device = "cpu"
    predictors = {"efficientnetv2_s": None, "convnextv2_nano": None, "swinv2_tiny": None}
    config = SimpleNamespace(max_bytes=8 * 1024**2)

    def close(self):
        self.ready = False


def test_web_shell_and_modules_share_the_api_origin_and_disable_photo_caching():
    service = Service()
    with TestClient(create_app(service_factory=lambda: service)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "One face. Three perspectives." in response.text
        assert "id=\"face-overlay\"" in response.text
        assert "id=\"camera-preview\"" in response.text
        assert "id=\"eval-classification\"" in response.text
        assert "script-src 'self'" in response.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert response.headers["permissions-policy"] == "camera=(self), microphone=(), geolocation=()"
        assert response.headers["cache-control"] == "no-store"
        for name in ("styles.css", "app.js", "core.js", "dom.js", "charts.js", "evaluation.js", "brand.svg"):
            asset = client.get("/assets/" + name)
            assert asset.status_code == 200, name
            assert asset.headers["cache-control"] == "no-store"
        assert client.head("/").status_code == 200
        assert client.get("/health").json()["models_loaded"] == 3
    assert not service.ready


def test_static_routes_do_not_expose_artifacts_research_photos_or_source_files():
    with TestClient(create_app(service_factory=Service)) as client:
        for path in ("/assets/%2e%2e/api.py", "/assets/%2e%2e/%2e%2e/%2e%2e/README.md",
                     "/data/raw/UTKFace", "/artifacts/convnextv2_nano/best.pt", "/assets/missing.js"):
            assert client.get(path).status_code == 404, path

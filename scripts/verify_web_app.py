"""Verify a running local web app, saved dashboard evidence and optional QA fixtures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import sha256_file, utc_now, write_json

from visage.model_registry import METHODS as ACTIVE_METHODS
METHODS = list(ACTIVE_METHODS)


def verify(url, fixtures=None):
    checks = {}
    with httpx.Client(base_url=url, trust_env=False, timeout=45) as client:
        response = client.get("/")
        response.raise_for_status()
        assert 'id="face-overlay"' in response.text and '/assets/app.js' in response.text
        assert "/__qa__/" not in response.text, "Use the production server for this check."
        assert "script-src 'self'" in response.headers["content-security-policy"]
        assert response.headers["cache-control"] == "no-store"
        checks["shell_security"] = True
        for name in ("styles.css", "app.js", "core.js", "dom.js", "charts.js", "evaluation.js", "brand.svg"):
            asset = client.get("/assets/" + name)
            asset.raise_for_status()
            assert asset.content == (ROOT / "src/visage/web" / name).read_bytes()
        checks["assets_match_source"] = True
        health = client.get("/health").json()
        assert health["status"] == "ready" and health["models_loaded"] == 3
        data = client.get("/v1/evaluation").json()
        classification = data["classification"]
        assert classification["status"] == "complete" and classification["samples"] == 3492
        dashboard = {}
        for method in METHODS:
            value = classification["methods"][method]
            matrix = value["confusion_matrix"]
            assert sum(map(sum, matrix)) == classification["samples"]
            assert abs(sum(matrix[i][i] for i in range(3)) / 3492 * 100 - value["accuracy_percent"]) < 1e-9
            assert 0 <= value["macro_f1"] <= 1
            assert all(0 <= c["auc"] <= 1 and len(c["fpr"]) <= 501 for c in value["one_vs_rest_roc"].values())
            dashboard[method] = {key: value[key] for key in ("accuracy_percent", "macro_f1", "macro_auc", "confusion_matrix")}
        detector = data["detection"]
        assert detector["status"] == "complete" and detector["images"] == 3226
        assert detector["metrics"]["AP50"]["ground_truth_faces"] == 39112
        assert len(detector["metrics"]["AP50"]["froc"]["recall"]) <= 501
        pipeline = data["face_pipeline"]
        assert pipeline["attempted"] == pipeline["accepted"] + pipeline["rejected"] == 3492
        assert abs(pipeline["coverage"] - pipeline["accepted"] / pipeline["attempted"]) < 1e-12
        checks["dashboard"] = {"classification": dashboard, "AP50": detector["metrics"]["AP50"]["ap_percent"], "AP75": detector["metrics"]["AP75"]["ap_percent"], "coverage": pipeline["coverage"]}
        if fixtures:
            fixtures = Path(fixtures)
            outcomes = {}

            def post(name, path):
                file = fixtures / name
                kind = "image/png" if file.suffix == ".png" else "image/jpeg"
                return client.post(path, content=file.read_bytes(), headers={"Content-Type": kind})

            for name in ("single.jpg", "class_0.jpg", "class_1.jpg", "rotated_exif.jpg"):
                detected = post(name, "/v1/faces"); detected.raise_for_status()
                value = detected.json()
                assert len(value["faces"]) == 1
                result = post(name, "/v1/predict?method=all&face_index=0")
                result.raise_for_status(); result = result.json()
                assert [p["method"] for p in result["predictions"]] == METHODS
                assert all(0 <= p["class_id"] <= 2 and p["inference_and_preprocessing_ms"] >= 0 for p in result["predictions"])
                assert result["image"] == value["image"]
                individual = post(name, f"/v1/predict?method={METHODS[0]}&face_index=0").json()
                assert individual["predictions"][0]["class_id"] == result["predictions"][0]["class_id"]
                assert individual["face_processing"]["pixel_sha256"] == result["face_processing"]["pixel_sha256"]
                outcomes[name] = {"image": result["image"], "labels": [p["class_id"] for p in result["predictions"]], "crop_sha256": result["face_processing"]["pixel_sha256"], "source_sha256": sha256_file(fixtures / name)}
            multiple = post("multiple.png", "/v1/faces").json()
            assert len(multiple["faces"]) == 2
            no_selection = post("multiple.png", "/v1/predict?method=all")
            assert no_selection.status_code >= 400 and no_selection.json()["error"]["code"] == "MULTIPLE_FACES"
            selections = []
            for face in multiple["faces"]:
                result = post("multiple.png", f"/v1/predict?method=all&face_index={face['index']}")
                result.raise_for_status(); result = result.json()
                assert result["selected_face"]["index"] == face["index"]
                selections.append({"face": face["index"], "crop": result["face_processing"]["pixel_sha256"], "labels": [p["class_id"] for p in result["predictions"]]})
            assert selections[0]["crop"] != selections[1]["crop"]
            outcomes["multiple.png"] = selections
            for name, expected in (("blank.png", "NO_FACE"), ("invalid.jpg", "INVALID_IMAGE"), ("oversized.jpg", "IMAGE_TOO_LARGE")):
                response = post(name, "/v1/faces")
                assert response.status_code >= 400 and response.json()["error"]["code"] == expected
                outcomes[name] = {"status": response.status_code, "code": expected}
            checks["fixtures"] = outcomes
    return {"ok": True, "verified_utc": utc_now(), "url": url, "scope": "Functional smoke checks on existing validation fixtures; no new accuracy experiment.", "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pretrained/web_app/http_verification.json")
    args = parser.parse_args()
    record = verify(args.url, args.fixtures)
    write_json(args.output, record)
    print(json.dumps({"ok": record["ok"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

"""Offline real-artifact integration on validation images, never test photos.

These checks run when local trained artifacts/data and the pinned detector are
present. A source-only installation can still run all unit tests without fetching
research images or network weights.
"""

from __future__ import annotations

import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from visage.api import create_app
from visage.faces import decode_image
from visage.runtime import seed_everything
from visage.service import InferenceService, METHODS
from visage.pretrained_data import pixels


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "artifacts/pretrained/serving/efficientnetv2_s/best.pt", "artifacts/pretrained/serving/convnextv2_nano/best.pt",
    "artifacts/pretrained/serving/swinv2_tiny/best.pt", ".cache/face_detector/provenance.json",
    "data/processed/manifest.csv", "data/raw/UTKFace",
)
pytestmark = pytest.mark.skipif(
    not all((ROOT / item).exists() for item in REQUIRED),
    reason="Real trained artifacts, pinned detector and validation photos are optional local integration fixtures.",
)


def png(image, **kwargs):
    stream = BytesIO()
    image.save(stream, format="PNG", **kwargs)
    return stream.getvalue()


def labels(result):
    return {item["method"]: item["class_id"] for item in result["predictions"]}


@pytest.fixture(scope="module")
def real_application():
    seed_everything(42, threads=4)
    with (ROOT / "data/processed/manifest.csv").open(newline="", encoding="utf-8") as stream:
        row = next(row for row in csv.DictReader(stream) if row["split"] == "validation")
    path = ROOT / "data/raw/UTKFace" / row["relative_path"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["file_sha256"]
    service = InferenceService()
    app = create_app(service_factory=lambda: service)
    with TestClient(app) as client:
        yield client, service, path, data
    assert not service.ready and not service.predictors


def test_real_api_all_methods_match_direct_adapters_on_the_same_face(real_application):
    client, service, _, data = real_application
    identities = {name: id(value) for name, value in service.predictors.items()}
    response = client.post("/v1/predict?method=all", content=data, headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 200, response.text
    result = response.json()
    image = decode_image(data)
    try:
        candidates = service.detector.detect(image)
        prepared = service.detector.prepare(image, candidates=candidates)
        try:
            expected = {method: service.predictors[method].predict(prepared.crop)["class_id"] for method in METHODS}
            assert labels(result) == expected
            for prediction in result['predictions']:
                entries = prediction['class_scores']
                assert prediction['score_type'] == 'uncalibrated_softmax'
                assert [entry['class_id'] for entry in entries] == [0, 1, 2]
                actual = np.array([entry['score'] for entry in entries])
                unchanged = service.predictors[prediction['method']].predict_batch([pixels(prepared.crop)])[0]
                np.testing.assert_allclose(actual, unchanged, atol=1e-7, rtol=1e-6)
                assert int(actual.argmax()) == prediction['class_id']
                assert abs(actual.sum() - 1) < 1e-6
            assert result["face_processing"]["pixel_sha256"] == prepared.metadata["pixel_sha256"]
        finally:
            prepared.crop.close()
    finally:
        image.close()
    assert set(labels(result)) == set(METHODS)
    assert {prediction["manifest_sha256"] for prediction in result["predictions"]} == {service.manifest_sha256}
    assert result["manifest_sha256"] == service.manifest_sha256
    assert all(value >= 0 for value in result["timings_ms"].values())
    assert {name: id(value) for name, value in service.predictors.items()} == identities
    assert "crop" not in result and "image_bytes" not in result


def test_real_no_face_and_multiple_faces_require_selection(real_application):
    client, _, _, data = real_application
    blank = png(Image.new("RGB", (640, 480), (127, 127, 127)))
    result = client.post("/v1/predict", content=blank, headers={"Content-Type": "image/png"})
    assert result.status_code == 422 and result.json()["error"]["code"] == "NO_FACE"
    image = decode_image(data)
    try:
        combined = Image.new("RGB", (image.width * 2 + 100, image.height + 60), (127, 127, 127))
        combined.paste(image, (10, 30))
        combined.paste(image, (image.width + 90, 30))
        two_faces = png(combined)
        combined.close()
    finally:
        image.close()
    detected = client.post("/v1/faces", content=two_faces, headers={"Content-Type": "image/png"})
    assert detected.status_code == 200 and detected.json()["selection_required"]
    assert len(detected.json()["faces"]) == 2
    rejected = client.post("/v1/predict?method=all", content=two_faces, headers={"Content-Type": "image/png"})
    assert rejected.status_code == 422 and rejected.json()["error"]["code"] == "MULTIPLE_FACES"
    selected = client.post("/v1/predict?method=all&face_index=1", content=two_faces, headers={"Content-Type": "image/png"})
    assert selected.status_code == 200, selected.text
    assert selected.json()["selected_face"]["index"] == 1 and len(selected.json()["predictions"]) == 3
    absent = client.post("/v1/predict?face_index=8", content=two_faces, headers={"Content-Type": "image/png"})
    assert absent.status_code == 400 and absent.json()["error"]["code"] == "INVALID_FACE_INDEX"


def test_real_exif_orientation_produces_identical_face_pixels_and_predictions(real_application):
    client, _, _, data = real_application
    image = decode_image(data)
    try:
        plain = png(image)
        stored = image.transpose(Image.Transpose.ROTATE_90)
        exif = stored.getexif()
        exif[274] = 6  # Stored counterclockwise pixels are displayed clockwise.
        oriented = png(stored, exif=exif)
        stored.close()
    finally:
        image.close()
    ordinary = client.post("/v1/predict?method=all", content=plain, headers={"Content-Type": "image/png"})
    rotated = client.post("/v1/predict?method=all", content=oriented, headers={"Content-Type": "image/png"})
    assert ordinary.status_code == rotated.status_code == 200
    assert ordinary.json()["face_processing"]["pixel_sha256"] == rotated.json()["face_processing"]["pixel_sha256"]
    assert labels(ordinary.json()) == labels(rotated.json())


def test_real_http_prediction_matches_image_cli(real_application, tmp_path):
    client, _, path, data = real_application
    response = client.post("/v1/predict?method=all", content=data, headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 200
    output = tmp_path / "prediction.json"
    run = subprocess.run([sys.executable, str(ROOT / "scripts/predict_image.py"), str(path),
                          "--method", "all", "--device", "cpu", "--output", str(output)],
                         cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stderr
    command = json.loads(output.read_text(encoding="utf-8"))
    assert labels(response.json()) == labels(command)
    assert response.json()["face_processing"]["pixel_sha256"] == command["face_processing"]["pixel_sha256"]


def test_existing_crop_cli_matches_prepared_face_adapters(real_application, tmp_path):
    _, service, _, data = real_application
    image = decode_image(data)
    try:
        prepared = service.detector.prepare(image)
        try:
            expected = {method: service.predictors[method].predict(prepared.crop)["class_id"] for method in METHODS}
            crop_path = tmp_path / "prepared.png"
            prepared.crop.save(crop_path)
        finally:
            prepared.crop.close()
    finally:
        image.close()
    output = tmp_path / "aligned_prediction.json"
    run = subprocess.run([sys.executable, str(ROOT / "scripts/predict.py"), str(crop_path),
                          "--method", "all", "--output", str(output)],
                         cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stderr
    assert labels(json.loads(output.read_text(encoding="utf-8"))) == expected


def test_real_default_and_loaded_identity_agree_across_api_and_both_clis(real_application, tmp_path):
    client, service, path, data = real_application
    metadata = client.get("/v1/models").json()
    default = metadata["default_method"]
    assert default in METHODS and len(metadata["methods"]) == 3
    for value in metadata["methods"]:
        assert value["seed"] == 42 and value["selected_epoch"] >= 1
        assert len(value["artifact_sha256"]) == 64
    response = client.post("/v1/predict", content=data, headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 200, response.text
    expected = response.json()["predictions"]
    assert len(expected) == 1 and expected[0]["method"] == default
    image = decode_image(data)
    try:
        prepared = service.detector.prepare(image)
        try:
            crop = tmp_path / "selected.png"
            prepared.crop.save(crop)
        finally:
            prepared.crop.close()
    finally:
        image.close()
    for command, source in (("predict_image.py", path), ("predict.py", crop)):
        output = tmp_path / (command + ".json")
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / command), str(source), "--output", str(output)],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        values = json.loads(output.read_text(encoding="utf-8"))["predictions"]
        assert len(values) == 1 and values[0]["method"] == default
        assert values[0]["class_id"] == expected[0]["class_id"]

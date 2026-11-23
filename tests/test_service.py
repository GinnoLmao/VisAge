"""Shared-crop, persistent-model and cleanup guarantees for the service."""
from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from visage.faces import FaceConfig, FaceError
from visage.service import InferenceService, METHODS


class Candidate:
    index = 0

    def to_dict(self):
        return {"index": self.index, "bbox": [5, 5, 80, 80], "score": 0.99}


class Detector:
    config = FaceConfig()

    def __init__(self):
        self.decoded = None
        self.crop = None
        self.detect_count = 0
        self.prepare_count = 0

    def detect(self, image):
        self.decoded = image
        self.detect_count += 1
        return [Candidate()]

    def prepare(self, image, face_index=None, candidates=None):
        assert candidates is not None and len(candidates) == 1
        self.prepare_count += 1
        self.crop = image.resize((224, 224))
        return SimpleNamespace(crop=self.crop, selected=candidates[0], candidates=candidates,
                               quality={"accepted": True}, metadata={"policy": "test"})


def encoded_image():
    stream = BytesIO()
    Image.fromarray(np.random.default_rng(12).integers(0, 256, (100, 110, 3), dtype=np.uint8)).save(stream, "PNG")
    return stream.getvalue()


def test_models_load_once_and_every_prediction_uses_identical_crop(tmp_path):
    loads, crops = [], []

    class Predictor:
        def __init__(self, method, directory, device):
            loads.append((method, directory, device))
            self.method = method
            self.metadata = {"manifest_sha256": "a" * 64}

        def predict(self, image):
            crops.append(image)
            return {"method": self.method, "class_id": 1, "age_group": "18-59"}

    detector = Detector()
    service = InferenceService(tmp_path, detector=detector, predictor_factory=Predictor)
    assert [item[0] for item in loads] == list(METHODS)
    for _ in range(2):
        result = service.predict(encoded_image(), method="all")
        assert len(result["predictions"]) == 3
        assert crops[-3] is crops[-2] is crops[-1]
        assert result["manifest_sha256"] == "a" * 64
        assert all(value >= 0 for value in result["timings_ms"].values())
        with pytest.raises(ValueError):
            detector.crop.getpixel((0, 0))
        with pytest.raises(ValueError):
            detector.decoded.getpixel((0, 0))
    assert len(loads) == 3 and detector.detect_count == detector.prepare_count == 2
    assert service.describe()["default_method"] == "convnextv2_nano"
    service.close()
    assert not service.ready and not service.predictors
    with pytest.raises(RuntimeError, match="not ready"):
        service.predict(encoded_image())


def test_common_dataset_freeze_is_required(tmp_path):
    def factory(method, directory, device):
        return SimpleNamespace(metadata={"manifest_sha256": method})
    with pytest.raises(ValueError, match="same frozen dataset"):
        InferenceService(tmp_path, detector=Detector(), predictor_factory=factory)


def test_default_method_and_cleanup_when_a_classifier_fails(tmp_path):
    calls = []

    class Predictor:
        metadata = {"manifest_sha256": "a" * 64}

        def __init__(self, method, directory, device):
            self.method = method

        def predict(self, image):
            calls.append(self.method)
            raise RuntimeError("classifier failed")

    detector = Detector()
    service = InferenceService(tmp_path, detector=detector, predictor_factory=Predictor)
    with pytest.raises(RuntimeError, match="classifier failed"):
        service.predict(encoded_image())
    assert calls == ["convnextv2_nano"]
    for image in (detector.crop, detector.decoded):
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_faces_does_not_invoke_classifiers_and_invalid_method_does_not_decode(tmp_path):
    detector = Detector()
    service = InferenceService(tmp_path, detector=detector,
                               predictor_factory=lambda *args, **kwargs: SimpleNamespace(metadata={"manifest_sha256": "a" * 64}))
    result = service.faces(encoded_image())
    assert result["image"] == {"width": 110, "height": 100}
    assert not result["selection_required"] and len(result["faces"]) == 1
    assert detector.prepare_count == 0
    with pytest.raises(FaceError) as error:
        service.predict(b"not an image", method="unknown")
    assert error.value.code == "INVALID_METHOD"
    assert detector.detect_count == 1


def test_no_face_releases_decoded_image(tmp_path):
    class EmptyDetector(Detector):
        def detect(self, image):
            self.decoded = image
            return []

    detector = EmptyDetector()
    service = InferenceService(tmp_path, detector=detector,
                               predictor_factory=lambda *args, **kwargs: SimpleNamespace(metadata={"manifest_sha256": "a" * 64}))
    with pytest.raises(FaceError) as error:
        service.faces(encoded_image())
    assert error.value.code == "NO_FACE"
    with pytest.raises(ValueError):
        detector.decoded.getpixel((0, 0))

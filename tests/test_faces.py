"""Image-admission, original-coordinate geometry, selection and quality checks."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import math
import threading

import cv2
import numpy as np
from PIL import Image
import pytest

from visage.faces import FaceCandidate, FaceConfig, FaceDetector, FaceError, decode_image
from visage.alignment import TRAINING_TEMPLATE


def encoded(image=None, image_format="PNG", **kwargs):
    image = image or Image.new("RGB", (100, 120), "green")
    stream = BytesIO()
    image.save(stream, format=image_format, **kwargs)
    return stream.getvalue()


def candidate(index=0, box=(45.0, 35.0, 100.0, 120.0), eyes=((72.0, 75.0), (118.0, 75.0))):
    return FaceCandidate(index, box, (*eyes, (95.0, 100.0), (78.0, 128.0), (113.0, 128.0)), 0.97)


def textured_image():
    rng = np.random.default_rng(9)
    return Image.fromarray(rng.integers(30, 225, (220, 220, 3), dtype=np.uint8))


def detector_stub(raw=None, config=None):
    class Network:
        def setInputSize(self, size):
            self.size = size

        def detect(self, bgr):
            assert bgr.shape[:2] == (self.size[1], self.size[0])
            return 0, raw

    detector = FaceDetector.__new__(FaceDetector)
    detector.config = config or FaceConfig()
    detector._network = Network()
    detector._lock = threading.Lock()
    return detector


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_allowed_bytes_decode_independently_rgb(image_format):
    result = decode_image(encoded(image_format=image_format))
    assert result.mode == "RGB" and result.size == (100, 120)
    result.load()
    result.close()


def test_exif_orientation_and_alpha_are_handled():
    image = Image.new("RGB", (100, 140), "blue")
    exif = image.getexif()
    exif[274] = 6
    oriented = decode_image(encoded(image, "JPEG", exif=exif))
    assert oriented.size == (140, 100)
    transparent = Image.new("RGBA", (50, 50), (255, 0, 0, 0))
    pixels = np.asarray(decode_image(encoded(transparent)))
    np.testing.assert_array_equal(pixels, np.full((50, 50, 3), 127, dtype=np.uint8))


@pytest.mark.parametrize("data,code", [(b"", "EMPTY_IMAGE"), (b"plain text", "INVALID_IMAGE")])
def test_bad_bytes_have_typed_errors(data, code):
    with pytest.raises(FaceError) as info:
        decode_image(data)
    assert info.value.code == code
    assert info.value.status_code == 400


def test_admission_limits_before_decode_and_unsupported_formats():
    with pytest.raises(FaceError, match="upload limit") as info:
        decode_image(b"x" * 101, replace(FaceConfig(), max_bytes=100))
    assert info.value.status_code == 413
    with pytest.raises(FaceError) as info:
        decode_image(encoded(), replace(FaceConfig(), max_pixels=500))
    assert info.value.code == "IMAGE_DIMENSIONS_TOO_LARGE"
    with pytest.raises(FaceError) as info:
        decode_image(encoded(image_format="BMP"))
    assert info.value.code == "UNSUPPORTED_IMAGE" and info.value.status_code == 415
    with pytest.raises(FaceError) as info:
        decode_image(encoded(Image.new("RGB", (20, 40))))
    assert info.value.code == "IMAGE_TOO_SMALL"


def test_animated_images_and_truncated_pixels_are_rejected():
    image = Image.new("RGB", (100, 100), "red")
    data = encoded(image, "PNG", save_all=True, append_images=[Image.new("RGB", (100, 100), "blue")], duration=100)
    with pytest.raises(FaceError) as info:
        decode_image(data)
    assert info.value.code == "ANIMATED_IMAGE"
    data = encoded(image, "JPEG")
    with pytest.raises(FaceError) as info:
        decode_image(data[:len(data) // 2])
    assert info.value.code == "INVALID_IMAGE"


def test_detection_maps_coordinates_to_original_and_sorts_indexes():
    raw = np.array([
        [60, 20, 20, 30, 65, 30, 75, 30, 70, 35, 66, 40, 74, 40, .99],
        [10, 15, 20, 30, 15, 25, 25, 25, 20, 30, 16, 35, 24, 35, .95],
    ], dtype=np.float32)
    detector = detector_stub(raw, replace(FaceConfig(), max_detection_side=100))
    result = detector.detect(Image.new("RGB", (200, 160)))
    assert [f.index for f in result] == [0, 1]
    assert result[0].bbox == (20.0, 30.0, 40.0, 60.0)
    assert result[0].landmarks[0] == (30.0, 50.0)
    assert detector._network.size == (100, 80)


def test_detection_rejects_invalid_boxes_and_caps_faces():
    row = [20, 20, 80, 100, 40, 50, 75, 50, 60, 65, 45, 85, 75, 85, .99]
    bad = row.copy()
    bad[0] = math.nan
    detector = detector_stub(np.array([bad, row]))
    assert len(detector.detect(Image.new("RGB", (200, 200)))) == 1
    detector = detector_stub(np.array([row, row]), replace(FaceConfig(), max_faces=1))
    with pytest.raises(FaceError) as info:
        detector.detect(Image.new("RGB", (200, 200)))
    assert info.value.code == "TOO_MANY_FACES"


def test_no_face_multiple_faces_and_explicit_selection():
    detector = detector_stub()
    image = textured_image()
    assert detector.detect(image) == []
    with pytest.raises(FaceError) as info:
        detector.prepare(image)
    assert info.value.code == "NO_FACE"
    faces = [candidate(0), candidate(1)]
    with pytest.raises(FaceError) as info:
        detector.prepare(image, candidates=faces)
    assert info.value.code == "MULTIPLE_FACES" and len(info.value.details["faces"]) == 2
    with pytest.raises(FaceError) as info:
        detector.prepare(image, face_index=8, candidates=faces)
    assert info.value.code == "INVALID_FACE_INDEX"
    with pytest.raises(FaceError):
        detector.prepare(image, face_index=True, candidates=faces)
    prepared = detector.prepare(image, face_index=1, candidates=faces)
    assert prepared.selected.index == 1 and prepared.metadata["selection"] == "explicit_index"
    assert prepared.crop.size == (224, 224) and prepared.crop.mode == "RGB"
    prepared.crop.close()


def test_affine_fits_training_template_and_same_pixels_are_reproducible():
    detector = detector_stub()
    image = textured_image()
    template = np.asarray(TRAINING_TEMPLATE)
    source_points = template * 180 + 20
    face = FaceCandidate(0, (20, 20, 180, 180), tuple(map(tuple, source_points)), .97)
    first = detector.prepare(image, candidates=[face])
    second = detector.prepare(image, candidates=[face])
    matrix = np.asarray(first.metadata["affine_original_to_crop"])
    mapped = np.c_[source_points, np.ones(5)] @ matrix.T
    np.testing.assert_allclose(mapped, template * 224, atol=1e-10)
    assert first.quality["alignment_rmse_pixels"] < 1e-10
    np.testing.assert_array_equal(np.asarray(first.crop), np.asarray(second.crop))
    assert first.metadata["pixel_sha256"] == second.metadata["pixel_sha256"]
    assert first.quality["engineering_thresholds_only"]


@pytest.mark.parametrize("face,code", [
    (candidate(box=(40, 40, 30, 35)), "FACE_TOO_SMALL"),
    (candidate(eyes=((72, 75), (73, 75))), "UNUSABLE_FACE"),
    (candidate(eyes=((72, 50), (80, 100))), "UNUSABLE_FACE"),
])
def test_tiny_or_unusable_landmark_geometry_rejected(face, code):
    with pytest.raises(FaceError) as info:
        detector_stub().prepare(textured_image(), candidates=[face])
    assert info.value.code == code


@pytest.mark.parametrize("value", [0, 127, 255])
def test_flat_dark_or_bright_face_is_rejected(value):
    with pytest.raises(FaceError) as info:
        detector_stub().prepare(Image.new("RGB", (220, 220), (value,) * 3), candidates=[candidate()])
    assert info.value.code == "UNUSABLE_FACE"
    assert "low_contrast" in info.value.details["reasons"]


def test_smooth_face_with_contrast_still_fails_blur_gate():
    gradient = np.tile(np.linspace(20, 230, 220, dtype=np.uint8), (220, 1))
    image = Image.fromarray(np.repeat(gradient[:, :, None], 3, axis=2))
    with pytest.raises(FaceError) as info:
        detector_stub().prepare(image, candidates=[candidate()])
    assert "blur" in info.value.details["reasons"]
    assert info.value.details["quality"]["dynamic_range_p95_p5"] > 20


def test_absent_face_pixels_are_rejected_separately_from_optional_context():
    face = FaceCandidate(0, (0, 0, 100, 120),
                         ((1, 60), (16, 60), (8, 80), (3, 100), (18, 100)), .99)
    with pytest.raises(FaceError) as info:
        detector_stub().prepare(textured_image(), candidates=[face])
    assert "face_cut_off" in info.value.details["reasons"]


def test_missing_model_and_invalid_configuration_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaceDetector(tmp_path / "missing.onnx")
    for kwargs in ({"max_bytes": 0}, {"score_threshold": 1.1}, {"crop_expansion": .5},
                   {"min_blur_variance": -1}, {"eye_height_fraction": math.nan},
                   {"max_faces": 2.5}, {"min_face_size": True}):
        with pytest.raises(ValueError):
            FaceConfig(**kwargs)


def test_model_and_license_checksums_are_enforced(tmp_path):
    # No arbitrary ONNX can reach OpenCV merely by altering adjacent metadata.
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not the pinned model")
    (tmp_path / "provenance.json").write_text('{}')
    with pytest.raises(ValueError, match="pinned official"):
        FaceDetector(path)

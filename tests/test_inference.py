"""Image admission and offline artifact-backed inference adapter checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from visage import CLASS_NAMES
from visage.datasets import make_transform
from visage.inference import AgeGroupPredictor, read_face_crop
from visage.methods.classical import ClassicalAgeClassifier, FeatureConfig, LinearSVM
from visage.methods.visagenet import VisAgeNet
from visage.runtime import sha256_file
from visage.training import logits_to_probabilities


MANIFEST_SHA = "a" * 64


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_allowed_decoded_formats_are_rgb_and_independent_of_filename(tmp_path: Path, image_format: str) -> None:
    path = tmp_path / "face.bin"
    Image.new("RGB", (13, 17), "green").save(path, format=image_format)
    result = read_face_crop(path)
    assert result.mode == "RGB" and result.size == (13, 17)
    result.load()  # Pixels remain usable after the input file closes.


def test_image_exif_orientation_is_applied_before_inference(tmp_path: Path) -> None:
    path = tmp_path / "portrait.jpg"
    pixels = np.random.default_rng(21).integers(0, 256, (8, 13, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    with Image.open(path) as original:
        expected = np.asarray(ImageOps.exif_transpose(original).convert("RGB"))
    result = read_face_crop(path)
    assert result.size == (8, 13)
    np.testing.assert_array_equal(np.asarray(result), expected)


def test_nonimage_and_unsupported_decoded_format_are_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "renamed.jpg"
    bogus.write_bytes(b"this is text, not image data")
    with pytest.raises(UnidentifiedImageError):
        read_face_crop(bogus)
    unsupported = tmp_path / "renamed.png"
    Image.new("RGB", (5, 5), "blue").save(unsupported, format="BMP")
    with pytest.raises(ValueError, match="JPEG, PNG or WebP"):
        read_face_crop(unsupported)


def test_input_byte_limit_is_checked_before_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "too_big.jpg"
    with path.open("wb") as stream:
        stream.truncate(8 * 1024**2 + 1)

    def never_decode(*args, **kwargs):
        raise AssertionError("Oversized input must be rejected before image decode.")

    monkeypatch.setattr(Image, "open", never_decode)
    with pytest.raises(ValueError, match="8 MiB"):
        read_face_crop(path)


def test_decoded_pixel_limit_is_checked_before_pixel_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "compressed_large.png"
    path.write_bytes(b"small encoded placeholder")

    class LargeHeader:
        width = 5001
        height = 4000
        format = "PNG"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def load(self):
            raise AssertionError("A too-large decoded image must not be loaded.")

    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: LargeHeader())
    with pytest.raises(ValueError, match="20 megapixel"):
        read_face_crop(path)


def test_adapter_rejects_unknown_method_without_loading_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        AgeGroupPredictor("unknown", tmp_path / "missing", device="cpu")


@pytest.fixture
def classical_artifact(tmp_path: Path):
    images = [
        Image.fromarray(np.random.default_rng(index + 200).integers(0, 256, (20, 20, 3), dtype=np.uint8))
        for index in range(9)
    ]
    model = ClassicalAgeClassifier(
        FeatureConfig(image_size=16, lbp_grid=2, lbp_radii=(1,)),
        LinearSVM(epochs=3, batch_size=3, seed=19),
    ).fit(images, np.array([0, 1, 2] * 3))
    directory = tmp_path / "classical"
    model.save(directory)
    (directory / "selection.json").write_text(json.dumps({
        "method": "classical", "manifest_sha256": MANIFEST_SHA,
        "svm_sha256": sha256_file(directory / "svm.npz"),
        "feature_config": model.extractor.to_dict(),
        "selected_configuration": model.svm._configuration(),
    }), encoding="utf-8")
    return directory, model, images[0]


def test_classical_adapter_matches_direct_prediction_without_claiming_face_detection(classical_artifact) -> None:
    directory, model, image = classical_artifact
    predictor = AgeGroupPredictor("classical", directory, device="cpu")
    result = predictor.predict(image)
    expected = model.predict_image(image)
    assert result["class_id"] == expected and result["age_group"] == CLASS_NAMES[expected]
    assert result["manifest_sha256"] == MANIFEST_SHA and result["device"] == "cpu"
    assert result["inference_and_preprocessing_ms"] >= 0
    assert "exact_age" not in result and "face_detected" not in result
    # The classifier accepts an already-selected crop. A non-face image can
    # still receive a label; the adapter must not claim it validates faces.
    nonface = predictor.predict(Image.new("RGB", (20, 20), "white"))
    assert nonface["class_id"] in (0, 1, 2)
    assert "face_detected" not in nonface


def test_classical_adapter_rejects_changed_selected_weights(classical_artifact) -> None:
    directory, _, _ = classical_artifact
    with (directory / "svm.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        AgeGroupPredictor("classical", directory)


def test_classical_adapter_rejects_changed_feature_geometry_with_matching_dimension(classical_artifact) -> None:
    directory, _, _ = classical_artifact
    path = directory / "features.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    # Weighting changes the descriptor semantics without changing its length
    # or the separately protected SVM weight file.
    metadata["configuration"]["hog_weight"] = 2.0
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(feature|configuration|selected)"):
        AgeGroupPredictor("classical", directory)


def test_classical_adapter_rejects_changed_svm_metadata_configuration(classical_artifact) -> None:
    directory, _, _ = classical_artifact
    path = directory / "svm.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["configuration"]["standardize"] = False
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(configuration|selected)"):
        AgeGroupPredictor("classical", directory)


@pytest.fixture
def neural_artifact(tmp_path: Path):
    torch.set_num_threads(1)
    torch.manual_seed(64)
    config = {"widths": [8, 16, 32, 64], "blocks_per_stage": [1, 1, 1, 1], "dropout": 0.0}
    model = VisAgeNet(**config).eval()
    transform = {"image_size": 32, "mean": [0.5] * 3, "std": [0.5] * 3}
    checkpoint = {
        "artifact_format": 1, "method": "visagenet", "model_config": config,
        "model_state": model.state_dict(), "manifest_sha256": MANIFEST_SHA,
        "transform": transform,
    }
    directory = tmp_path / "neural"
    directory.mkdir()
    path = directory / "best.pt"
    torch.save(checkpoint, path)
    summary = {"artifact_sha256": sha256_file(path), "manifest_sha256": MANIFEST_SHA, "method": "visagenet"}
    (directory / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    image = Image.fromarray(np.random.default_rng(35).integers(0, 256, (23, 31, 3), dtype=np.uint8))
    return directory, model, image, checkpoint


def test_neural_adapter_matches_direct_cpu_prediction(neural_artifact) -> None:
    directory, model, image, checkpoint = neural_artifact
    transform = checkpoint["transform"]
    pixels = make_transform(image_size=transform["image_size"], mean=transform["mean"], std=transform["std"])(image).unsqueeze(0)
    with torch.inference_mode():
        expected = int(logits_to_probabilities(model(pixels), model.head_type).argmax(1))
    result = AgeGroupPredictor("visagenet", directory, device="cpu").predict(image)
    assert result["class_id"] == expected and result["age_group"] == CLASS_NAMES[expected]
    assert result["manifest_sha256"] == MANIFEST_SHA and result["device"] == "cpu"
    assert result["inference_and_preprocessing_ms"] >= 0


def test_neural_adapter_rejects_checksum_and_method_mismatch(neural_artifact) -> None:
    directory, _, _, _ = neural_artifact
    # Requesting a different method still loads the declared architecture
    # safely, then rejects the adapter/checkpoint mismatch.
    with pytest.raises(ValueError, match="method does not match"):
        AgeGroupPredictor("convnext_ordinal", directory, device="cpu")
    with (directory / "best.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        AgeGroupPredictor("visagenet", directory, device="cpu")


@pytest.mark.parametrize(("field", "changed"), [("manifest_sha256", "b" * 64), ("method", "convnext_ordinal")])
def test_neural_adapter_rejects_summary_checkpoint_provenance_mismatch(neural_artifact, field, changed) -> None:
    directory, _, _, _ = neural_artifact
    path = directory / "training_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary[field] = changed
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(manifest|method|summary|provenance)"):
        AgeGroupPredictor("visagenet", directory, device="cpu")

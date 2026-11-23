"""Mathematical, behavioral and artifact tests for the classical method."""

import json
import csv
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from visage.methods.classical import (
    ClassicalAgeClassifier,
    ClassicalFeatureExtractor,
    FeatureConfig,
    LinearSVM,
    feature_vector,
    hog_descriptor,
    multiclass_hinge_loss_and_gradient,
    uniform_lbp_codes,
)


def _toy_data(seed=7, counts=(24, 24, 24)):
    rng = np.random.default_rng(seed)
    centers = np.array([[-3, -2], [3, -2], [0, 4]], dtype=np.float32)
    X = np.concatenate([rng.normal(center, 0.18, size=(count, 2)) for center, count in zip(centers, counts)]).astype(np.float32)
    y = np.repeat(np.arange(3), counts)
    return X, y


def test_lbp_known_uniform_and_nonuniform_patterns():
    assert uniform_lbp_codes(np.ones((5, 5), dtype=np.float32)).shape == (3, 3)
    np.testing.assert_array_equal(uniform_lbp_codes(np.ones((5, 5))), 8)
    isolated = np.zeros((3, 3), dtype=np.float32)
    isolated[1, 1] = 1
    assert uniform_lbp_codes(isolated).item() == 0
    alternating = np.array([[2, 0, 2], [0, 1, 0], [2, 0, 2]], dtype=np.float32)
    assert uniform_lbp_codes(alternating).item() == 9


def test_lbp_monotone_intensity_and_rotation_invariance():
    rng = np.random.default_rng(11)
    gray = rng.uniform(0.1, 0.8, size=(15, 15)).astype(np.float32)
    codes = uniform_lbp_codes(gray, radius=2)
    np.testing.assert_array_equal(uniform_lbp_codes(gray * 2 + 0.5, radius=2), codes)
    np.testing.assert_array_equal(uniform_lbp_codes(np.rot90(gray), radius=2), np.rot90(codes))


def test_hog_zero_image_and_known_gradient_orientation():
    flat = hog_descriptor(np.zeros((16, 16)), cell_size=8, block_size=2)
    assert flat.shape == (36,)
    np.testing.assert_array_equal(flat, 0)
    horizontal = np.tile(np.arange(16, dtype=np.float32), (16, 1))
    horizontal_hog = hog_descriptor(horizontal).reshape(4, 9)
    vertical_hog = hog_descriptor(horizontal.T).reshape(4, 9)
    assert horizontal_hog.sum(axis=0).argmax() == 0
    assert vertical_hog.sum(axis=0).argmax() in (4, 5)
    np.testing.assert_allclose(np.linalg.norm(horizontal_hog), 1, atol=1e-6)
    # Multiplying every gradient magnitude leaves the normalized descriptor.
    np.testing.assert_allclose(hog_descriptor(horizontal * 5), horizontal_hog.ravel(), atol=1e-6)


@pytest.mark.parametrize("options", [{}, {"include_lbp": False}, {"include_hog": False}, {"image_size": 224}])
def test_feature_dimensions_and_reproducible_serialized_configuration(options):
    config = FeatureConfig(**options)
    image = Image.fromarray(np.random.default_rng(3).integers(0, 256, size=(43, 71, 3), dtype=np.uint8))
    restored = FeatureConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    first = feature_vector(image, config)
    second = feature_vector(image, restored)
    assert first.dtype == np.float32
    assert first.shape == (config.feature_dimension,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first), 1, atol=1e-6)
    if not options:
        assert len(first) == 8580


def test_extractor_supports_bounded_generator_to_memmap(tmp_path):
    extractor = ClassicalFeatureExtractor(FeatureConfig(image_size=32))
    images = [Image.new("RGB", (20, 30), color) for color in ("black", "white")]
    out = np.memmap(tmp_path / "features.dat", mode="w+", dtype=np.float32, shape=(2, extractor.feature_dimension))
    returned = extractor.transform((image for image in images), out=out)
    assert returned is out
    np.testing.assert_array_equal(out, np.stack([feature_vector(image, extractor.config) for image in images]))
    with pytest.raises(ValueError, match="generators"):
        extractor.transform(iter(images))
    with pytest.raises(ValueError, match="count"):
        extractor.transform(images[:1], out=out)


@pytest.mark.parametrize("options", [{"image_size": 127}, {"lbp_radii": []}, {"lbp_radii": [float("nan")]}, {"include_hog": False, "include_lbp": False}, {"hog_weight": 0}, {"image_size": 8}])
def test_invalid_feature_settings_fail_early(options):
    with pytest.raises(ValueError):
        FeatureConfig(**options)


def test_hinge_loss_matches_manual_weighted_objective():
    X = np.eye(3)
    y = np.arange(3)
    coef = np.array([[0.1, 0.7, -0.2], [0.4, 0.2, 0.3], [-0.1, -0.4, 0.6]])
    bias = np.array([0.2, -0.1, 0.05])
    sample_weights = np.array([0.5, 2.0, 1.0])
    scores = X @ coef.T + bias
    expected = 0.1 * np.sum(coef**2)
    expected += sum(sample_weights[i] * max(0, max(scores[i, j] for j in range(3) if j != y[i]) - scores[i, y[i]] + 1) for i in range(3)) / 3
    loss, _, _ = multiclass_hinge_loss_and_gradient(X, y, coef, bias, regularization=0.2, sample_weight=sample_weights)
    np.testing.assert_allclose(loss, expected)


def test_hinge_gradient_matches_finite_differences_away_from_kinks():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(7, 4))
    y = np.array([0, 1, 2, 0, 1, 2, 0])
    coef, bias = rng.normal(0, 0.15, size=(3, 4)), rng.normal(0, 0.05, size=3)
    weights = np.array([0.5, 1, 2, 0.5, 1, 2, 0.5])
    def objective():
        return multiclass_hinge_loss_and_gradient(X, y, coef, bias, regularization=0.07, sample_weight=weights)[0]
    _, grad_w, grad_b = multiclass_hinge_loss_and_gradient(X, y, coef, bias, regularization=0.07, sample_weight=weights)
    epsilon = 1e-6
    for values, gradient in ((coef, grad_w), (bias, grad_b)):
        numerical = np.empty_like(values)
        for index in np.ndindex(values.shape):
            original = values[index]
            values[index] = original + epsilon
            above = objective()
            values[index] = original - epsilon
            below = objective()
            values[index] = original
            numerical[index] = (above - below) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)


def test_inactive_hinge_has_only_weight_regularization_gradient():
    X, y = np.eye(3), np.arange(3)
    coef, bias = 3 * np.eye(3), np.zeros(3)
    loss, grad_w, grad_b = multiclass_hinge_loss_and_gradient(X, y, coef, bias, regularization=0.2)
    np.testing.assert_allclose(loss, 0.1 * np.sum(coef**2))
    np.testing.assert_allclose(grad_w, 0.2 * coef)
    np.testing.assert_array_equal(grad_b, 0)


def test_hinge_supports_integer_inputs_and_parameters():
    loss, gradient, _ = multiclass_hinge_loss_and_gradient(np.eye(3, dtype=int), np.arange(3), np.zeros((3, 3), dtype=int), np.zeros(3, dtype=int))
    assert loss == 1
    assert np.isfinite(gradient).all()


def test_svm_recovers_separable_classes_and_is_seed_reproducible():
    X, y = _toy_data()
    options = dict(epochs=35, batch_size=12, learning_rate=0.15, seed=9)
    model = LinearSVM(**options).fit(X, y)
    duplicate = LinearSVM(**options).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.coef_, duplicate.coef_)
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert model.decision_function(X).shape == (len(X), 3)


def test_balanced_class_weights_and_train_only_scaling(tmp_path):
    X, y = _toy_data(counts=(40, 10, 5))
    X = np.column_stack([X, np.full(len(X), 7)]).astype(np.float32)
    out = np.memmap(tmp_path / "training.dat", mode="w+", dtype="float32", shape=X.shape)
    out[:] = X
    model = LinearSVM(epochs=15, batch_size=7).fit(out, y)
    np.testing.assert_allclose(model.class_weight_, len(y) / (3 * np.array([40, 10, 5])), rtol=1e-6)
    np.testing.assert_allclose(model.feature_mean_, X.mean(axis=0), atol=1e-6)
    assert model.feature_scale_[-1] == 1
    before_mean, before_scale = model.feature_mean_.copy(), model.feature_scale_.copy()
    model.predict(np.array([[100, -200, 7]], dtype=np.float32))
    np.testing.assert_array_equal(model.feature_mean_, before_mean)
    np.testing.assert_array_equal(model.feature_scale_, before_scale)
    np.testing.assert_array_equal(out, X)


def test_svm_safe_artifact_roundtrip_and_invalid_array_rejection(tmp_path):
    X, y = _toy_data()
    model = LinearSVM(epochs=15).fit(X, y)
    array_path, json_path = model.save(tmp_path / "model")
    loaded = LinearSVM.load(array_path)
    np.testing.assert_array_equal(model.decision_function(X), loaded.decision_function(X))
    assert json.loads(json_path.read_text())["format_version"] == 1
    with np.load(array_path, allow_pickle=False) as arrays:
        values = {key: arrays[key] for key in arrays.files}
    values["feature_scale"][0] = 0
    np.savez_compressed(array_path, **values)
    with pytest.raises(ValueError, match="positive"):
        LinearSVM.load(array_path)


def test_image_classifier_bundle_keeps_feature_configuration(tmp_path):
    config = FeatureConfig(image_size=16, lbp_grid=2, lbp_radii=(1,))
    images = [Image.fromarray(np.random.default_rng(i).integers(0, 256, size=(20, 20), dtype=np.uint8)) for i in range(6)]
    model = ClassicalAgeClassifier(config, LinearSVM(epochs=3)).fit(images, np.array([0, 1, 2, 0, 1, 2]))
    model.save(tmp_path / "bundle")
    restored = ClassicalAgeClassifier.load(tmp_path / "bundle")
    assert restored.extractor.config == config
    np.testing.assert_array_equal(model.predict(images), restored.predict(images))
    assert restored.predict_image(images[0]) in (0, 1, 2)


@pytest.mark.parametrize("case", ["nan", "missing_class", "bad_label", "empty"])
def test_svm_rejects_invalid_training_data(case):
    X, y = _toy_data()
    if case == "nan":
        X[0, 0] = np.nan
    elif case == "missing_class":
        X, y = X[y != 2], y[y != 2]
    elif case == "bad_label":
        y[0] = 3
    else:
        X, y = X[:0], y[:0]
    with pytest.raises(ValueError):
        LinearSVM().fit(X, y)


def test_predict_requires_fitted_matching_dimensions_and_supports_empty_rows():
    X, y = _toy_data()
    with pytest.raises(ValueError, match="Fit"):
        LinearSVM().predict(X)
    model = LinearSVM(epochs=3).fit(X, y)
    with pytest.raises(ValueError, match="dimension"):
        model.predict(np.zeros((1, 3)))
    assert model.predict(np.empty((0, 2))).shape == (0,)


def test_training_cli_uses_train_for_fitting_validation_for_selection_and_reuses_cache(tmp_path, monkeypatch):
    # Tiny fixtures exercise the workflow without training on project data or
    # depending on a real dataset download/frozen manifest generation.
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_classical.py"
    spec = importlib.util.spec_from_file_location("train_classical_under_test", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for split in ("train", "validation", "test"):
        for label in range(3):
            filename = f"{split}_{label}.png"
            pixels = np.random.default_rng(len(rows)).integers(0, 256, (20, 20, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(raw / filename)
            rows.append({"sample_id": filename, "relative_path": filename, "split": split, "label": label})
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("controlled frozen manifest fixture", encoding="utf-8")
    verification_calls = []
    def verify(path, raw_dir, verify_files=False):
        verification_calls.append((path, raw_dir, verify_files))
        return {"ok": True}
    monkeypatch.setattr(trainer, "verify_manifest", verify)
    monkeypatch.setattr(trainer, "load_manifest", lambda path: rows)
    args = trainer.build_parser().parse_args(["--manifest", str(manifest), "--raw-dir", str(raw), "--output-dir", str(tmp_path / "artifacts"), "--cache-dir", str(tmp_path / "cache"), "--image-size", "16", "--epochs", "2", "--learning-rates", "0.01", "--regularizations", "0.001"])
    config = FeatureConfig(image_size=16)
    expected = {}
    for split in ("train", "validation"):
        expected[split] = np.stack([feature_vector(trainer._open_face(raw.resolve(), row["relative_path"]), config) for row in rows if row["split"] == split])
    original_fit, original_scores = LinearSVM.fit, LinearSVM.decision_function
    fit_calls, prediction_calls = [], []
    def fit_only_train(self, X, y):
        np.testing.assert_array_equal(X, expected["train"])
        np.testing.assert_array_equal(y, [0, 1, 2])
        fit_calls.append(len(X))
        return original_fit(self, X, y)
    def predict_only_validation(self, X):
        np.testing.assert_array_equal(X, expected["validation"])
        prediction_calls.append(len(X))
        return original_scores(self, X)
    monkeypatch.setattr(LinearSVM, "fit", fit_only_train)
    monkeypatch.setattr(LinearSVM, "decision_function", predict_only_validation)
    selected = trainer.train(args)
    assert selected["test_evaluated"] is False
    with (args.output_dir / "validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == 3
    assert {row["split"] for row in predictions} == {"validation"}
    assert {row["relative_path"] for row in predictions} == {f"validation_{label}.png" for label in range(3)}
    # A finite corruption must also be detected: validating only shape/dtype
    # would silently train on this changed cached row on the second run.
    split_path = next(args.cache_dir.glob("*/train.npy"))
    damaged = np.load(split_path, mmap_mode="r+", allow_pickle=False)
    damaged[0, 0] += 2
    damaged.flush()
    del damaged
    def forbid_reextraction(*args, **kwargs):
        raise AssertionError("A complete matching feature cache must be reused.")
    monkeypatch.setattr(trainer, "_open_face", forbid_reextraction)
    trainer.train(args)
    assert fit_calls == prediction_calls == [3, 3]
    assert len(verification_calls) == 2 and all(call[2] is True for call in verification_calls)

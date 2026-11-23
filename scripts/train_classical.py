"""Train the classical method using frozen train/validation partitions only.

Example (from an editable project installation)::

    python scripts/train_classical.py --manifest data/processed/manifest.csv \
        --raw-dir data/raw/UTKFace --output-dir artifacts/classical \
        --cache-dir data/cache/classical --seed 42

Fixed, label-independent descriptors are extracted once for the manifest.
Test rows are never passed to the optimizer, scaler, selection or prediction.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageOps

from visage.data import load_manifest, verify_manifest
from visage.methods import classical
from visage.methods.classical import ClassicalAgeClassifier, ClassicalFeatureExtractor, FeatureConfig, LinearSVM
from visage.metrics import classification_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _progress(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def _open_face(raw_dir: Path, relative_path: str) -> Image.Image:
    path = (raw_dir / relative_path).resolve()
    if not path.is_relative_to(raw_dir):
        raise ValueError("Manifest image path escapes the raw directory.")
    with Image.open(path) as original:
        # Copy the entire aligned crop; do not introduce a method-specific crop.
        return ImageOps.exif_transpose(original).convert("RGB")


def feature_cache(rows: list[dict], raw_dir: Path, cache_dir: Path, manifest_sha: str, config: FeatureConfig, *, workers: int = 2) -> tuple[np.ndarray, dict, Path]:
    """Create/resume a bounded, manifest/config/code-bound feature memmap."""
    source_sha = _sha256(Path(classical.__file__))
    identity = {"format_version": 1, "manifest_sha256": manifest_sha, "feature_config": config.to_dict(), "feature_source_sha256": source_sha, "rows": len(rows), "dimension": config.feature_dimension, "dtype": "float32"}
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = cache_dir / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    array_path, metadata_path = directory / "features.npy", directory / "features.json"
    completed = 0
    metadata = {}
    if metadata_path.exists() and array_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("Cache metadata differs from the requested frozen inputs.")
        completed = metadata.get("completed_rows", 0)
        if type(completed) is not int or not 0 <= completed <= len(rows):
            raise ValueError("Invalid feature-cache progress metadata.")
        existing = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if existing.shape != (len(rows), config.feature_dimension) or existing.dtype != np.float32:
            raise ValueError("Feature-cache shape/dtype differs from its metadata.")
        if metadata.get("complete") is True:
            if completed != len(rows) or metadata.get("features_sha256") != _sha256(array_path):
                raise ValueError("Completed feature-cache checksum does not match.")
            _progress("features_reused", rows=len(rows), cache=str(directory))
            return existing, metadata, directory
        del existing
        array = np.lib.format.open_memmap(array_path, mode="r+")
    else:
        array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32, shape=(len(rows), config.feature_dimension))
        metadata = {**identity, "completed_rows": 0, "complete": False}
        _write_json(metadata_path, metadata)
    extractor = ClassicalFeatureExtractor(config)
    started = time.perf_counter()
    previous_seconds = float(metadata.get("extraction_seconds", 0))
    def extract(row):
        face = _open_face(raw_dir, row["relative_path"])
        try:
            return extractor.transform_image(face)
        finally:
            face.close()
    # Map only one bounded chunk at a time. Workers never write the memmap,
    # and executor.map preserves manifest order. This changes scheduling only.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(completed, len(rows), 128):
            end = min(start + 128, len(rows))
            vectors = executor.map(extract, rows[start:end]) if workers > 1 else map(extract, rows[start:end])
            for index, vector in enumerate(vectors, start=start):
                array[index] = vector
            array.flush()
            metadata.update(completed_rows=end, complete=False, extraction_workers=workers, extraction_seconds=previous_seconds + time.perf_counter() - started)
            _write_json(metadata_path, metadata)
            _progress("features", completed=end, total=len(rows), workers=workers, seconds=round(time.perf_counter() - started, 2))
    array.flush()
    del array
    metadata.update(complete=True, completed_rows=len(rows), features_sha256=_sha256(array_path))
    metadata["extraction_seconds"] = previous_seconds + time.perf_counter() - started
    _write_json(metadata_path, metadata)
    return np.load(array_path, mmap_mode="r", allow_pickle=False), metadata, directory


def _split_features(features: np.ndarray, indices: np.ndarray, split: str, cache_directory: Path, source_sha: str) -> np.ndarray:
    """Copy split rows in bounded chunks, avoiding one large fancy-index copy."""
    array_path, metadata_path = cache_directory / f"{split}.npy", cache_directory / f"{split}.json"
    identity = {"source_features_sha256": source_sha, "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(), "shape": [len(indices), features.shape[1]], "dtype": "float32"}
    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in identity.items()) and metadata.get("features_sha256") == _sha256(array_path):
            existing = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if existing.shape == tuple(identity["shape"]) and existing.dtype == np.float32:
                return existing
            del existing
    array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32, shape=tuple(identity["shape"]))
    for start in range(0, len(indices), 128):
        array[start : start + 128] = features[indices[start : start + 128]]
    array.flush()
    del array
    _write_json(metadata_path, {**identity, "features_sha256": _sha256(array_path)})
    return np.load(array_path, mmap_mode="r", allow_pickle=False)


def train(args: argparse.Namespace) -> dict:
    manifest, raw_dir = args.manifest.resolve(), args.raw_dir.resolve()
    output_dir, cache_dir = args.output_dir.resolve(), args.cache_dir.resolve()
    if args.epochs < 1 or args.batch_size < 1 or args.feature_workers < 1:
        raise ValueError("epochs, batch_size and feature_workers must be positive.")
    if not args.learning_rates or any(not np.isfinite(v) or v <= 0 for v in args.learning_rates):
        raise ValueError("learning_rates must be positive and finite.")
    if not args.regularizations or any(not np.isfinite(v) or v < 0 for v in args.regularizations):
        raise ValueError("regularizations must be finite and nonnegative.")
    # Recheck actual bytes/normalized pixels, not just CSV identity, before
    # reusing descriptors or training against a supposedly frozen dataset.
    _progress("verification_started", manifest=str(manifest), verify_files=True)
    verified = verify_manifest(manifest, raw_dir, verify_files=True)
    rows = load_manifest(manifest)
    manifest_sha = _sha256(manifest)
    indices = {split: np.array([index for index, row in enumerate(rows) if row["split"] == split], dtype=np.int64) for split in ("train", "validation")}
    labels = {split: np.array([rows[index]["label"] for index in index_array], dtype=np.int64) for split, index_array in indices.items()}
    for split in indices:
        if not len(indices[split]) or set(labels[split].tolist()) != {0, 1, 2}:
            raise ValueError(f"{split} must contain all three classes before this comparison is trained.")
    config = FeatureConfig(image_size=args.image_size, include_hog=args.descriptor != "lbp", include_lbp=args.descriptor != "hog")
    _progress("manifest_verified", manifest_sha256=manifest_sha, train_rows=len(indices["train"]), validation_rows=len(indices["validation"]), feature_dimension=config.feature_dimension)
    features, cache_metadata, cache_directory = feature_cache(rows, raw_dir, cache_dir, manifest_sha, config, workers=args.feature_workers)
    X_train = _split_features(features, indices["train"], "train", cache_directory, cache_metadata["features_sha256"])
    X_validation = _split_features(features, indices["validation"], "validation", cache_directory, cache_metadata["features_sha256"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predeclared = {"manifest": str(manifest), "manifest_sha256": manifest_sha, "feature_config": config.to_dict(), "feature_source_sha256": cache_metadata["feature_source_sha256"], "cache_directory": str(cache_directory), "feature_workers": args.feature_workers, "feature_extraction_seconds": cache_metadata.get("extraction_seconds"), "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rates": args.learning_rates, "regularizations": args.regularizations, "selection_policy": "maximum validation macro_f1; then accuracy; then first declared grid configuration", "scaling_fit_split": "train", "class_weight_fit_split": "train", "selection_split": "validation", "test_evaluated": False, "hardware": {"device": "CPU", "platform": platform.platform(), "processor": platform.processor(), "numpy": np.__version__}, "verified_manifest": verified}
    _write_json(output_dir / "training_config.json", predeclared)
    best_key, best_model, best_prediction, best_scores, best_trial = None, None, None, None, None
    trials = []
    search_started = time.perf_counter()
    for rate in args.learning_rates:
        for regularization in args.regularizations:
            trial_id = len(trials)
            svm = LinearSVM(learning_rate=rate, regularization=regularization, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, class_weight="balanced", standardize=True)
            started = time.perf_counter()
            svm.fit(X_train, labels["train"])
            fit_seconds = time.perf_counter() - started
            scores = svm.decision_function(X_validation)
            prediction = scores.argmax(axis=1).astype(np.int64)
            metrics = classification_metrics(labels["validation"], prediction)
            trial = {"trial_id": trial_id, "learning_rate": rate, "regularization": regularization, "fit_seconds": fit_seconds, "epochs_completed": svm.n_iter_, "loss_history": svm.loss_history_, "validation_metrics": metrics}
            trials.append(trial)
            _write_json(output_dir / "search_results.json", trials)
            key = (metrics["macro_f1"], metrics["accuracy"], -trial_id)
            if best_key is None or key > best_key:
                best_key, best_trial = key, trial
                best_model = ClassicalAgeClassifier(config, svm)
                best_prediction, best_scores = prediction, scores
            _progress("trial_complete", trial_id=trial_id, validation_macro_f1=metrics["macro_f1"], validation_accuracy=metrics["accuracy"], fit_seconds=round(fit_seconds, 2))
    best_model.save(output_dir)
    _write_json(output_dir / "history.json", {"training_objective": "class-weighted multiclass hinge plus L2", "loss": best_model.svm.loss_history_})
    _write_json(output_dir / "validation_metrics.json", best_trial["validation_metrics"])
    with (output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "relative_path", "split", "label", "prediction", "margin_0", "margin_1", "margin_2"])
        writer.writeheader()
        for position, index in enumerate(indices["validation"]):
            row = rows[index]
            writer.writerow({"sample_id": row["sample_id"], "relative_path": row["relative_path"], "split": "validation", "label": row["label"], "prediction": int(best_prediction[position]), **{f"margin_{label}": float(best_scores[position, label]) for label in range(3)}})
    selection = {"method": "classical", "artifact_format": "visage.classical", "created_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest_sha, "feature_config": config.to_dict(), "feature_source_sha256": cache_metadata["feature_source_sha256"], "selected_trial_id": best_trial["trial_id"], "selected_configuration": best_model.svm._configuration(), "validation_metrics": best_trial["validation_metrics"], "selection_policy": predeclared["selection_policy"], "selected_fit_seconds": best_trial["fit_seconds"], "search_seconds": time.perf_counter() - search_started, "test_evaluated": False, "svm_sha256": _sha256(output_dir / "svm.npz")}
    _write_json(output_dir / "selection.json", selection)
    _progress("training_complete", artifact=str(output_dir), selected_trial=best_trial["trial_id"], validation_macro_f1=best_trial["validation_metrics"]["macro_f1"], test_evaluated=False)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/UTKFace"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/classical"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/classical"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--descriptor", choices=("combined", "hog", "lbp"), default="combined")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-workers", type=int, default=2)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--regularizations", type=float, nargs="+", default=[0.001, 0.01])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Classical training failed: {error}\n")


if __name__ == "__main__":
    main()

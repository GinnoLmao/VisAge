"""Classical component and nested-data experiments using validation only."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from visage.data import load_manifest, verify_manifest
from visage.methods.classical import ClassicalAgeClassifier, FeatureConfig, LinearSVM
from visage.metrics import classification_metrics
from visage.research import verify_protected_files
from visage.runtime import sha256_file, utc_now, write_json
from visage.training import select_training_subset


STUDY = ROOT / "reports/research"
ARTIFACTS = ROOT / "artifacts/research"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_helpers():
    spec = importlib.util.spec_from_file_location("visage_original_classical_trainer", ROOT / "scripts/train_classical.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(name, descriptor, fraction, rows, protocol, helpers):
    output = ARTIFACTS / name
    if (output / "selection.json").exists():
        selection = read(output / "selection.json")
        if (selection.get("research_protocol_sha256") != sha256_file(STUDY / "protocol.json")
                or sha256_file(output / "svm.npz") != selection["svm_sha256"]):
            raise ValueError("Completed classical research run changed.")
        print(json.dumps({"event": "classical_research_reused", "run": name}), flush=True)
        return
    run_started = time.perf_counter()
    selected = ([row for row in rows if row["split"] == "train"] if fraction == 100
                else select_training_subset(rows, STUDY / f"subsets/train_{fraction}.json", protocol["manifest_sha256"]))
    validation = [row for row in rows if row["split"] == "validation"]
    selected_ids = {row["sample_id"] for row in selected}
    baseline_config = read(ROOT / "artifacts/classical/training_config.json")
    baseline_selection = read(ROOT / "artifacts/classical/selection.json")
    feature_config = FeatureConfig.from_dict({**baseline_selection["feature_config"],
                                            "include_hog": descriptor != "lbp", "include_lbp": descriptor != "hog"})
    if descriptor == "combined":
        source_rows = rows
        directory = Path(baseline_config["cache_directory"])
        metadata = read(directory / "features.json")
        features_path = directory / "features.npy"
        if (metadata["manifest_sha256"] != protocol["manifest_sha256"]
                or metadata["features_sha256"] != sha256_file(features_path)
                or metadata["feature_config"] != feature_config.to_dict()):
            raise ValueError("Original fixed descriptor cache differs from frozen evidence.")
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
        if features.shape != (len(rows), feature_config.feature_dimension):
            raise ValueError("Original fixed descriptor cache has wrong shape.")
    else:
        source_rows = [row for row in rows if row["split"] in {"train", "validation"}]
        features, metadata, directory = helpers.feature_cache(
            source_rows, ROOT / "data/raw/UTKFace", ROOT / "data/cache/research/classical",
            protocol["manifest_sha256"], feature_config, workers=2,
        )
    train_indices = np.array([i for i, row in enumerate(source_rows) if row["sample_id"] in selected_ids], dtype=np.int64)
    val_indices = np.array([i for i, row in enumerate(source_rows) if row["split"] == "validation"], dtype=np.int64)
    train_features = helpers._split_features(features, train_indices, f"research_{name}_train", directory, metadata["features_sha256"])
    val_features = helpers._split_features(features, val_indices, f"research_{name}_validation", directory, metadata["features_sha256"])
    train_labels = np.array([source_rows[i]["label"] for i in train_indices], dtype=np.int64)
    val_labels = np.array([source_rows[i]["label"] for i in val_indices], dtype=np.int64)
    svm = LinearSVM(**baseline_selection["selected_configuration"])
    started = time.perf_counter()
    svm.fit(train_features, train_labels)
    training_seconds = time.perf_counter() - started
    scores = svm.decision_function(val_features)
    predictions = scores.argmax(1)
    metrics = classification_metrics(val_labels, predictions)
    model = ClassicalAgeClassifier(feature_config, svm)
    model.save(output)
    selection = {
        "method": "classical", "manifest_sha256": protocol["manifest_sha256"],
        "research_protocol_sha256": sha256_file(STUDY / "protocol.json"), "research_run_name": name,
        "descriptor": descriptor, "training_fraction_requested": fraction / 100,
        "training_rows": len(train_indices), "class_counts": np.bincount(train_labels, minlength=3).tolist(),
        "validation_rows": len(validation), "feature_config": feature_config.to_dict(),
        "selected_configuration": svm._configuration(), "validation_metrics": metrics,
        "selected_fit_seconds": training_seconds, "feature_extraction_seconds": metadata.get("extraction_seconds"),
        "feature_cache_source": "original verified fixed descriptors" if descriptor == "combined" else "new train/validation-only descriptor cache",
        "run_wall_seconds": time.perf_counter() - run_started,
        "svm_sha256": sha256_file(output / "svm.npz"), "created_utc": utc_now(),
        "selection_policy": "Fixed original selected hyperparameters; no new search. Validation-only component/data-efficiency experiment.",
        "test_evaluated": False,
    }
    write_json(output / "validation_metrics.json", metrics)
    write_json(output / "history.json", {"loss": svm.loss_history_})
    import csv
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "relative_path", "split", "label", "prediction", "margin_0", "margin_1", "margin_2"])
        for position, source_index in enumerate(val_indices):
            row = source_rows[source_index]
            writer.writerow([row["sample_id"], row["relative_path"], "validation", row["label"], int(predictions[position]), *scores[position].tolist()])
    write_json(output / "selection.json", selection)
    verify_protected_files(ROOT, protocol["protected_original_files"])
    print(json.dumps({"event": "classical_research_complete", "run": name, "validation_macro_f1": metrics["macro_f1"], "fit_seconds": training_seconds}), flush=True)


def main():
    protocol = read(STUDY / "protocol.json")
    verify_protected_files(ROOT, protocol["protected_original_files"])
    verify_protected_files(ROOT, protocol["implementation"]["source_files"])
    manifest = ROOT / "data/processed/manifest.csv"
    verify_manifest(manifest, ROOT / "data/raw/UTKFace", verify_files=False)
    if sha256_file(manifest) != protocol["manifest_sha256"]:
        raise ValueError("Research freeze changed.")
    rows = load_manifest(manifest)
    # Read only train/validation image bytes when creating new descriptors.
    for row in rows:
        if row["split"] in {"train", "validation"} and sha256_file(ROOT / "data/raw/UTKFace" / row["relative_path"]) != row["file_sha256"]:
            raise ValueError("A frozen training/validation image changed.")
    helpers = load_helpers()
    for percent in (25, 50):
        run(f"classical_data{percent}_seed42", "combined", percent, rows, protocol, helpers)
    for descriptor in ("hog", "lbp"):
        run(f"classical_{descriptor}_seed42", descriptor, 100, rows, protocol, helpers)


if __name__ == "__main__":
    main()

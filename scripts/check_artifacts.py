"""Independently reload completed artifacts and reproduce validation predictions.

This CPU check verifies artifact/freeze checksums and every saved validation
row. Inference is repeated on a fixed sample of validation images, never on
held-out test images. It does not train, tune, or change model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visage.runtime import configure_caches

configure_caches(ROOT)

import numpy as np
from PIL import Image, ImageOps
import torch

from visage.data import load_manifest, verify_manifest
from visage.datasets import ManifestDataset, make_transform
from visage.methods.classical import ClassicalAgeClassifier
from visage.metrics import classification_metrics
from visage.runtime import sha256_file, utc_now, write_json
from visage.training import load_neural_artifact, logits_to_probabilities


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_validation_csv(path: Path, rows: list[dict], expected_metrics: dict) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(rows):
        raise ValueError(f"{path}: saved validation row count differs from manifest.")
    for current, frozen in zip(saved, rows, strict=True):
        for key in ("sample_id", "relative_path"):
            if current[key] != frozen[key]:
                raise ValueError(f"{path}: validation {key} order differs from manifest.")
        if int(current["label"]) != frozen["label"]:
            raise ValueError(f"{path}: validation labels differ from manifest.")
    measured = classification_metrics(
        np.array([int(row["label"]) for row in saved], dtype=np.int64),
        np.array([int(row["prediction"]) for row in saved], dtype=np.int64),
    )
    if measured != expected_metrics:
        raise ValueError(f"{path}: recomputed saved-prediction metrics differ from selection.")
    return saved


def check_classical(directory: Path, raw: Path, rows: list[dict], positions: np.ndarray, manifest_sha: str) -> dict:
    selection = read_json(directory / "selection.json")
    if selection["manifest_sha256"] != manifest_sha:
        raise ValueError("Classical artifact belongs to another manifest.")
    actual_sha = sha256_file(directory / "svm.npz")
    if actual_sha != selection["svm_sha256"]:
        raise ValueError("Classical artifact SHA-256 differs from selection record.")
    model = ClassicalAgeClassifier.load(directory)
    if model.extractor.to_dict() != selection["feature_config"]:
        raise ValueError("Classical feature configuration differs from selection record.")
    saved = check_validation_csv(
        directory / "validation_predictions.csv", rows, selection["validation_metrics"]
    )
    maximum_error = 0.0
    for index in positions.tolist():
        with Image.open(raw / rows[index]["relative_path"]) as source:
            face = ImageOps.exif_transpose(source).convert("RGB")
            scores = model.decision_function([face])[0]
        expected = np.array([float(saved[index][f"margin_{label}"]) for label in range(3)])
        np.testing.assert_allclose(scores, expected, rtol=1e-4, atol=2e-5)
        if int(scores.argmax()) != int(saved[index]["prediction"]):
            raise ValueError("Reloaded classical prediction differs from saved validation output.")
        maximum_error = max(maximum_error, float(np.abs(scores - expected).max()))
    return {
        "artifact_sha256": actual_sha,
        "files_sha256": {path.name: sha256_file(path) for path in (
            directory / "svm.npz", directory / "svm.json", directory / "features.json"
        )},
        "csv_sha256": sha256_file(directory / "validation_predictions.csv"),
        "validation_rows_checked": len(saved),
        "reload_inference_samples": len(positions),
        "maximum_score_absolute_error": maximum_error,
        "ok": True,
    }


def check_neural(method: str, directory: Path, raw: Path, rows: list[dict], positions: np.ndarray, manifest_sha: str) -> dict:
    summary = read_json(directory / "training_summary.json")
    path = directory / "best.pt"
    actual_sha = sha256_file(path)
    if summary["manifest_sha256"] != manifest_sha:
        raise ValueError(f"{method}: training summary belongs to another manifest.")
    if summary["artifact_sha256"] != actual_sha:
        raise ValueError(f"{method}: artifact SHA-256 differs from completed training summary.")
    model, checkpoint = load_neural_artifact(path, device="cpu")
    if checkpoint["method"] != method or checkpoint["manifest_sha256"] != manifest_sha:
        raise ValueError(f"{method}: checkpoint identity differs from expected method/freeze.")
    if checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError(f"{method}: checkpoint epoch differs from selection summary.")
    if checkpoint["validation_metrics"]["macro_f1"] != summary["best_validation_macro_f1"]:
        raise ValueError(f"{method}: validation score differs from selection summary.")
    if sum(parameter.numel() for parameter in model.parameters()) != summary["parameters"]:
        raise ValueError(f"{method}: parameter count differs from training summary.")
    saved = check_validation_csv(
        directory / "validation_predictions.csv", rows, checkpoint["validation_metrics"]
    )
    transform = checkpoint["transform"]
    dataset = ManifestDataset(
        [rows[index] for index in positions.tolist()], raw,
        make_transform(image_size=transform["image_size"], mean=transform["mean"], std=transform["std"]),
    )
    maximum_error = 0.0
    with torch.inference_mode():
        for offset in range(0, len(dataset), 8):
            images = torch.stack([dataset[index][0] for index in range(offset, min(offset + 8, len(dataset)))])
            values = logits_to_probabilities(model(images), model.head_type).cpu().numpy()
            chosen = positions[offset:offset + len(values)].tolist()
            expected = np.array([
                [float(saved[index][f"score_{label}"]) for label in range(3)] for index in chosen
            ])
            np.testing.assert_allclose(values, expected, rtol=1e-4, atol=2e-5)
            if values.argmax(1).tolist() != [int(saved[index]["prediction"]) for index in chosen]:
                raise ValueError(f"{method}: reloaded predictions differ from saved validation output.")
            maximum_error = max(maximum_error, float(np.abs(values - expected).max()))
    return {
        "artifact_sha256": actual_sha,
        "csv_sha256": sha256_file(directory / "validation_predictions.csv"),
        "best_epoch": checkpoint["epoch"],
        "validation_rows_checked": len(saved),
        "reload_inference_samples": len(positions),
        "maximum_score_absolute_error": maximum_error,
        "ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/UTKFace")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--methods", nargs="+", choices=("classical", "visagenet", "convnext_ordinal"), default=["classical", "visagenet", "convnext_ordinal"])
    parser.add_argument("--samples", type=int, default=24, help="Fixed evenly spaced validation samples per method; 0 means all validation images.")
    parser.add_argument("--expected-manifest-sha", help="Optional SHA-256 pin from the approved dataset freeze.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/artifact_verification.json")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative.")
    torch.set_num_threads(2)
    verify_manifest(args.manifest, args.raw_dir, verify_files=False)
    manifest_sha = sha256_file(args.manifest)
    if args.expected_manifest_sha and manifest_sha != args.expected_manifest_sha:
        raise ValueError("Current manifest differs from the explicitly pinned SHA-256.")
    rows = [row for row in load_manifest(args.manifest) if row["split"] == "validation"]
    if not rows:
        raise ValueError("Validation partition is empty.")
    count = len(rows) if args.samples == 0 else min(args.samples, len(rows))
    positions = np.unique(np.linspace(0, len(rows) - 1, count, dtype=np.int64))
    # Check only the selected validation bytes; no held-out image is opened.
    for position in positions.tolist():
        row = rows[position]
        if sha256_file(args.raw_dir / row["relative_path"]) != row["file_sha256"]:
            raise ValueError("A validation image differs from the frozen dataset.")
    checks = {}
    for method in args.methods:
        directory = args.artifacts_dir / method
        checks[method] = (
            check_classical(directory, args.raw_dir, rows, positions, manifest_sha)
            if method == "classical"
            else check_neural(method, directory, args.raw_dir, rows, positions, manifest_sha)
        )
        print(json.dumps({"method": method, **checks[method]}), flush=True)
    report = {
        "ok": True, "checked_utc": utc_now(), "manifest_sha256": manifest_sha,
        "validation_positions": positions.tolist(), "methods": checks,
        "scope": "Artifact checksums; all saved validation rows and metrics; reloaded CPU predictions on fixed validation samples. No training or test-image inference.",
        "score_tolerance": {"rtol": 1e-4, "atol": 2e-5},
    }
    write_json(args.output, report)
    print(f"Verification report: {args.output}", flush=True)


if __name__ == "__main__":
    main()

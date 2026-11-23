"""Measure ordinary-photo face preprocessing on frozen validation crops only.

This is a crop-domain integration diagnostic, not an independent participant
pilot or latency benchmark. Every input remains in the attempted denominator.
Original age classifiers, research protocols and test evidence are never changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visage.data import load_manifest, verify_manifest
from visage.metrics import classification_metrics
from visage.runtime import configure_caches, seed_everything, sha256_file, utc_now, write_json

METHODS = ("classical", "visagenet", "convnext_ordinal")
FIELDS = ("sample_id", "label", "status", "error_code", "detected_count", "crop_sha256", "rejection_reasons",
          *METHODS, *(f"baseline_{method}" for method in METHODS))


def output_paths(output, predictions, root=ROOT):
    paths = tuple(Path(path).resolve() for path in (output, predictions))
    if paths[0] == paths[1]:
        raise ValueError("Report and predictions need distinct output paths.")
    for path in paths:
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Diagnostic outputs must stay inside the workspace.")
        if path.exists():
            raise FileExistsError("Choose fresh diagnostic paths; existing evidence is not overwritten.")
    return paths


def summarize_rows(outcomes):
    if not outcomes:
        raise ValueError("Nonempty attempted image outcomes are required.")
    accepted = [row for row in outcomes if row["status"] == "accepted"]
    if any(row["status"] not in {"accepted", "rejected"} for row in outcomes):
        raise ValueError("Unknown outcome status.")
    rejected = [row for row in outcomes if row["status"] == "rejected"]
    if any(row["label"] not in (0, 1, 2) for row in outcomes):
        raise ValueError("Unknown provided class label.")
    per_class = {}
    for label in range(3):
        rows = [row for row in outcomes if row["label"] == label]
        kept = sum(row["status"] == "accepted" for row in rows)
        per_class[str(label)] = {"attempted": len(rows), "accepted": kept, "rejected": len(rows) - kept,
                                 "coverage": kept / len(rows) if rows else None,
                                 "rejection_codes": dict(Counter(row["error_code"] for row in rows if row["status"] == "rejected"))}
    result = {"attempted": len(outcomes), "accepted": len(accepted), "rejected": len(rejected),
              "coverage": len(accepted) / len(outcomes), "by_class": per_class,
              "rejection_codes": dict(Counter(row["error_code"] for row in rejected)),
              "rejection_reasons": dict(Counter(reason for row in rejected for reason in json.loads(row["rejection_reasons"]))),
              "methods": {}}
    for method in METHODS:
        original_truth = [row["label"] for row in outcomes]
        original_predictions = [row[f"baseline_{method}"] for row in outcomes]
        if any(type(value) is not int or value not in (0, 1, 2) for value in original_predictions):
            raise ValueError("Invalid original prediction.")
        if accepted:
            truth = [row["label"] for row in accepted]
            predictions = [row[method] for row in accepted]
            baseline_accepted = [row[f"baseline_{method}"] for row in accepted]
            if any(type(value) is not int or value not in (0, 1, 2) for value in predictions):
                raise ValueError("Invalid accepted prediction.")
            conditional = classification_metrics(truth, predictions)
            paired_baseline = classification_metrics(truth, baseline_accepted)
            correct = sum(a == b for a, b in zip(truth, predictions, strict=True))
            changed = sum(a != b for a, b in zip(predictions, baseline_accepted, strict=True))
        else:
            conditional, paired_baseline, correct, changed = None, None, 0, 0
        result["methods"][method] = {
            "accepted_image_metrics": conditional, "original_aligned_metrics_same_accepted_images": paired_baseline,
            "original_aligned_metrics_all_images": classification_metrics(original_truth, original_predictions),
            "end_to_end_correct_count": correct, "end_to_end_correct_rate": correct / len(outcomes),
            "prediction_changes_on_accepted": changed,
            "prediction_change_rate_on_accepted": changed / len(accepted) if accepted else None,
            "macro_f1_delta_on_identical_accepted_subset": conditional["macro_f1"] - paired_baseline["macro_f1"] if accepted else None,
        }
    return result


class RecordingDetector:
    """Observe detection count without changing the production crop path."""
    def __init__(self, detector):
        self.detector = detector
        self.config = detector.config
        self.last_count = None

    def detect(self, image):
        self.last_count = None
        result = self.detector.detect(image)
        self.last_count = len(result)
        return result

    def prepare(self, image, face_index=None, candidates=None):
        return self.detector.prepare(image, face_index=face_index, candidates=candidates)

    def info(self):
        return self.detector.info()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/face_pipeline_validation.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports/face_pipeline_validation.csv")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    artifacts_dir = args.artifacts_dir.resolve(strict=True)
    if not artifacts_dir.is_relative_to(ROOT):
        raise ValueError("Diagnostic model evidence must stay inside the workspace.")
    if not 1 <= args.threads <= 4:
        raise ValueError("Use one to four CPU threads for this non-benchmark diagnostic.")
    args.output, args.predictions = output_paths(args.output, args.predictions)
    configure_caches(ROOT)
    seed_everything(42, threads=args.threads)
    import cv2
    cv2.setNumThreads(args.threads)
    from visage.faces import FaceDetector, FaceError
    from visage.service import InferenceService
    manifest, raw = ROOT / "data/processed/manifest.csv", ROOT / "data/raw/UTKFace"
    verify_manifest(manifest, raw, verify_files=False)
    rows = [row for row in load_manifest(manifest) if row["split"] == "validation"]
    if not rows or set(row["label"] for row in rows) != {0, 1, 2}:
        raise ValueError("All frozen validation classes are required.")
    # Preflight reads only the selected validation images, never test photos.
    for row in rows:
        path = (raw / row["relative_path"]).resolve()
        if not path.is_relative_to(raw.resolve()) or sha256_file(path) != row["file_sha256"]:
            raise ValueError("Frozen validation image bytes changed.")
    sources = ["src/visage/faces.py", "src/visage/alignment.py", "src/visage/service.py", "src/visage/inference.py", "scripts/evaluate_face_pipeline.py",
               "configs/face_geometry.json", "reports/training_alignment_comparison.json", "reports/face_quality_fit.json"]
    sources += [(artifacts_dir / method / filename).relative_to(ROOT).as_posix()
                for method in METHODS for filename in ("validation_predictions.csv", "validation_metrics.json")]
    source_hashes = {name: sha256_file(ROOT / name) for name in sources}
    baseline_predictions = {}
    for method in METHODS:
        with (artifacts_dir / method / "validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        if len(saved) != len(rows) or any(a["sample_id"] != b["sample_id"] or int(a["label"]) != b["label"]
                                         for a, b in zip(saved, rows, strict=True)):
            raise ValueError("Saved baseline validation order/labels differ from the manifest.")
        baseline_predictions[method] = [int(row["prediction"]) for row in saved]
    detector = RecordingDetector(FaceDetector())
    artifacts = {method: sha256_file(artifacts_dir / method / filename) for method, filename in (
        ("classical", "svm.npz"), ("visagenet", "best.pt"), ("convnext_ordinal", "best.pt"))}
    service = InferenceService(artifacts_dir, detector=detector, device="cpu")
    identity = {"manifest_sha256": sha256_file(manifest), "split": "validation", "device": "cpu",
                "torch_threads": args.threads, "opencv_threads": args.threads,
                "selection_policy": "No automatic index: single face only; multiple detections remain rejected attempts.",
                "source_sha256": source_hashes, "artifact_sha256": artifacts,
                "artifacts_dir": artifacts_dir.relative_to(ROOT).as_posix(),
                "face_config": asdict(detector.config), "detector": detector.info(),
                "new_test_evaluations": 0, "independent_local_pilot": False,
                "scope": "Full frozen validation-crop integration diagnostic. Existing aligned images are redetected/realigned; this is not ordinary-camera accuracy or an independent participant pilot.",
                "timing_note": "Elapsed bookkeeping only. Concurrent research training is allowed; no isolated inference-latency claim."}
    outcomes = []
    partial = args.predictions.with_suffix(args.predictions.suffix + ".partial")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_name(args.output.stem + "_progress.json")
    started = time.perf_counter()
    try:
        with partial.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for index, row in enumerate(rows):
                path = (raw / row["relative_path"]).resolve()
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != row["file_sha256"]:
                    raise ValueError("A validation photo changed after preflight.")
                outcome = {"sample_id": row["sample_id"], "label": row["label"],
                           "status": "rejected", "error_code": "", "detected_count": None,
                           "crop_sha256": "", "rejection_reasons": "[]",
                           **{method: None for method in METHODS},
                           **{f"baseline_{method}": baseline_predictions[method][index] for method in METHODS}}
                detector.last_count = None
                try:
                    result = service.predict(data, method="all")
                except FaceError as exc:
                    outcome.update(error_code=exc.code,
                                   detected_count=detector.last_count if detector.last_count is not None else exc.details.get("detected_count"),
                                   rejection_reasons=json.dumps(exc.details.get("reasons", [])))
                else:
                    values = {item["method"]: item["class_id"] for item in result["predictions"]}
                    if len(result["predictions"]) != 3 or set(values) != set(METHODS):
                        raise ValueError("Every accepted image must return all three methods.")
                    outcome.update(status="accepted", detected_count=result["faces_detected"],
                                   crop_sha256=result["face_processing"]["pixel_sha256"], **values)
                finally:
                    data = None
                writer.writerow(outcome)
                outcomes.append(outcome)
                if (index + 1) % 128 == 0 or index + 1 == len(rows):
                    stream.flush()
                    progress = {"status": "running", "completed": index + 1, "total": len(rows),
                                "accepted": sum(x["status"] == "accepted" for x in outcomes),
                                "rejection_codes": dict(Counter(x["error_code"] for x in outcomes if x["status"] == "rejected")),
                                "elapsed_seconds": time.perf_counter() - started, "recorded_utc": utc_now()}
                    write_json(progress_path, progress)
                    print(json.dumps(progress), flush=True)
        for name, expected in source_hashes.items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError("Production/source baseline bytes changed during diagnostic.")
        for method, expected in artifacts.items():
            artifact = artifacts_dir / method / ("svm.npz" if method == "classical" else "best.pt")
            if sha256_file(artifact) != expected:
                raise ValueError("An age classifier changed during diagnostic.")
        summary = summarize_rows(outcomes)
        for method in METHODS:
            previous = json.loads((artifacts_dir / method / "validation_metrics.json").read_text())
            if summary["methods"][method]["original_aligned_metrics_all_images"] != previous:
                raise ValueError("Recalculated original metrics differ from saved validation evidence.")
        partial.replace(args.predictions)
        record = {"ok": True, "completed_utc": utc_now(), **identity, **summary,
                  "predictions_csv": args.predictions.relative_to(ROOT).as_posix(),
                  "predictions_sha256": sha256_file(args.predictions),
                  "elapsed_seconds": time.perf_counter() - started,
                  "interpretation": "Compare new crop and original crop on the identical accepted subset. Coverage/rejections remain separate; end-to-end correct rate counts every attempted input. No settings or champions are changed automatically."}
        write_json(args.output, record)
        write_json(progress_path, {"status": "complete", "completed": len(rows), "total": len(rows),
                                   "recorded_utc": utc_now(), "output": args.output.relative_to(ROOT).as_posix()})
        print(json.dumps({"ok": True, "attempted": record["attempted"], "coverage": record["coverage"],
                          "methods": record["methods"]}), flush=True)
    finally:
        service.close()


if __name__ == "__main__":
    main()

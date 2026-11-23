"""Read-only aggregate evaluation for the currently served model artifacts."""
from __future__ import annotations

import json
import csv
import hashlib
from io import StringIO
import math
from dataclasses import asdict
from pathlib import Path
import threading
import numpy as np

from .runtime import sha256_file
from .metrics import classification_metrics
from . import CLASS_NAMES

_RECORD_CACHE = {}
_RECORD_LOCK = threading.RLock()


def compact_curve(curve, x_name, y_name, limit=501):
    x, y = curve[x_name], curve[y_name]
    if len(x) != len(y):
        raise ValueError("Invalid evaluation curve lengths.")
    positions = np.unique(np.linspace(0, max(0, len(x) - 1), min(len(x), limit), dtype=int))
    return {x_name: [x[index] for index in positions], y_name: [y[index] for index in positions],
            "full_points": curve.get("full_points", len(x)), "display_points": len(positions)}


def load_compact_record(path, kind):
    """Serialize first-load parsing and cache only small aggregate curve views.

    At most eight paths are cached. File mtime/size changes invalidate the view;
    exact served artifact hashes are still checked separately on every request.
    """
    path = Path(path).resolve()
    with _RECORD_LOCK:
        stat = path.stat()
        identity = (stat.st_mtime_ns, stat.st_size, kind)
        if path in _RECORD_CACHE and _RECORD_CACHE[path][0] == identity:
            return _RECORD_CACHE[path][1]
        record = json.loads(path.read_text(encoding="utf-8"))
        if kind == "classification":
            for value in record["methods"].values():
                value["roc"]["per_class"] = {label: {"auc": curve["auc"], **compact_curve(curve, "fpr", "tpr")}
                    for label, curve in value["roc"]["per_class"].items()}
                value["roc"].pop("macro_curve", None)
        elif kind == "detection":
            record["metrics"] = {name: {key: value[key] for key in ("ap_percent", "iou_threshold", "ground_truth_faces")}
                | compact_curve(value, "recall", "precision")
                | ({"froc": compact_curve(value, "false_positives_per_image", "recall")}
                   if "false_positives_per_image" in value else {})
                for name, value in record["metrics"].items()}
        else:
            raise ValueError("Unknown aggregate evaluation record kind.")
        if len(_RECORD_CACHE) >= 8:
            _RECORD_CACHE.pop(next(iter(_RECORD_CACHE)))
        _RECORD_CACHE[path] = identity, record
        return record


def classification_details(directory, expected_sha256, samples):
    """Derive the dashboard's F1/matrices from checksum-bound saved predictions.

    Cache only small aggregate metrics, never photos or individual predictions.
    Older/minimal evidence without a CSV exposes unavailable details explicitly.
    """
    path = (Path(directory) / "validation_predictions.csv").resolve()
    if not path.is_file():
        return None
    with _RECORD_LOCK:
        stat = path.stat()
        identity = (stat.st_mtime_ns, stat.st_size, "prediction_metrics", expected_sha256, samples)
        if path in _RECORD_CACHE and _RECORD_CACHE[path][0] == identity:
            return _RECORD_CACHE[path][1]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("Classification prediction evidence changed.")
        rows = list(csv.DictReader(StringIO(payload.decode("utf-8"))))
        if len(rows) != samples or len({row["sample_id"] for row in rows}) != samples:
            raise ValueError("Classification predictions do not cover the expected unique samples.")
        metrics = classification_metrics([int(row["label"]) for row in rows], [int(row["prediction"]) for row in rows])
        if len(_RECORD_CACHE) >= 8:
            _RECORD_CACHE.pop(next(iter(_RECORD_CACHE)))
        _RECORD_CACHE[path] = identity, metrics
        return metrics


def evaluation_snapshot(directory: Path, service):
    result = {"classification": {"status": "pending"}, "detection": {"status": "pending"}, "face_pipeline": {"status": "pending"},
              "identity_recognition": "Not performed. Classification accuracy refers to age groups.",
              "notice": "Offline aggregate evaluation; these values are not confidence percentages for an uploaded person."}
    classification = Path(directory) / "classification_roc.json"
    if classification.is_file():
        record = load_compact_record(classification, "classification")
        if not record.get("ok") or record["manifest_sha256"] != service.manifest_sha256:
            raise ValueError("Evaluation does not match the served dataset freeze.")
        methods = {}
        profile_path = Path(directory) / "application_profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else None
        if profile and (not profile.get("ok") or profile["manifest_sha256"] != service.manifest_sha256 or set(profile["methods"]) != set(service.predictors)):
            raise ValueError("Application performance profile does not match the served dataset/models.")
        for method, predictor in service.predictors.items():
            value = record["methods"][method]
            filename = "svm.npz" if method == "classical" else "best.pt"
            disk_sha256 = sha256_file(predictor.directory / filename)
            if getattr(predictor, "artifact_sha256", disk_sha256) != disk_sha256 or disk_sha256 != value["artifact_sha256"]:
                raise ValueError("Evaluation does not match a served model artifact.")
            details = classification_details(predictor.directory, value.get("predictions_sha256"), record["samples"])
            if details and not math.isclose(details["accuracy"] * 100, value["accuracy_percent"], abs_tol=1e-9):
                raise ValueError("Classification accuracy disagrees with saved predictions.")
            methods[method] = {"accuracy_percent": value["accuracy_percent"], "macro_auc": value["roc"]["macro_auc_mean"],
                "metric_details_status": "complete" if details else "pending",
                "macro_f1": details["macro_f1"] if details else None,
                "weighted_f1": sum(v["support"] * v["f1"] for v in details["per_class"]) / details["samples"] if details else None,
                "confusion_matrix": details["confusion_matrix"] if details else None,
                "row_normalized_confusion_matrix": details["row_normalized_confusion_matrix"] if details else None,
                "per_class": details["per_class"] if details else None,
                "under18_predicted_older_percent": details["under18_predicted_older_rate"] * 100 if details else None,
                "senior_predicted_below60_percent": details["senior_predicted_below60_rate"] * 100 if details else None,
                "score_type": value["score_type"], "one_vs_rest_roc": {label: {"auc": curve["auc"], **compact_curve(curve, "fpr", "tpr")}
                    for label, curve in value["roc"]["per_class"].items()}}
            if profile:
                cost = profile["methods"][method]
                if cost["artifact_sha256"] != disk_sha256:
                    raise ValueError("Application performance profile does not match the served model artifact.")
                if not all(math.isfinite(cost[key]) and cost[key] > 0 for key in ("parameters", "training_seconds", "runtime_median_ms")):
                    raise ValueError("Invalid application performance measurements.")
                if getattr(predictor, "metadata", {}).get("parameters", cost["parameters"]) != cost["parameters"]:
                    raise ValueError("Parameter count differs from the loaded model.")
                methods[method].update({"runtime_median_ms": cost["runtime_median_ms"], "parameters_millions": cost["parameters"] / 1e6,
                    "training_minutes": cost["training_seconds"] / 60, "training_epochs": cost["training_epochs"], "training_run_id": cost["training_run_id"]})
        result["classification"] = {"status": "complete", "task": record["task"], "split": record["split"],
            "input_scope": "Original pre-aligned validation crops; this is classifier evaluation, not full API or real-camera accuracy.",
            "samples": record["samples"], "class_names": list(CLASS_NAMES), "dataset": "UTKFace",
            "model_scope": record.get("scope", "Metrics for the currently served checkpoints"),
            "manifest_sha256": record["manifest_sha256"], "methods": methods,
            "runtime_scope": profile["runtime_scope"] if profile else None,
            "training_scope": profile["training_scope"] if profile else None}
    detection = Path(directory) / "detection_ap.json"
    if detection.is_file():
        record = load_compact_record(detection, "detection")
        if not record.get("ok") or record["detector_sha256"] != service.detector.info()["sha256"]:
            raise ValueError("Evaluation does not match the served face detector.")
        result["detection"] = {"status": "complete", "dataset": record["dataset"], "images": record["images"],
            "protocol_note": record["protocol_note"], "operating_point_note": record["operating_point_note"],
            "benchmark_configuration": {key: record.get("config", {}).get(key) for key in
                ("score_threshold", "nms_threshold", "max_detection_side", "max_faces")},
            "serving_confidence_threshold": service.detector.info().get("configuration", {}).get("score_threshold"),
            "metrics": {name: {"ap_percent": value["ap_percent"], "iou_threshold": value["iou_threshold"],
                              "ground_truth_faces": value["ground_truth_faces"], **compact_curve(value, "recall", "precision"),
                              **({"froc": value["froc"]} if "froc" in value else {})}
                        for name, value in record["metrics"].items()}}
    pipeline = Path(directory).parent / "face_pipeline_validation.json"
    if pipeline.is_file():
        record = json.loads(pipeline.read_text(encoding="utf-8"))
        if not record.get("ok") or record["manifest_sha256"] != service.manifest_sha256:
            raise ValueError("Face-pipeline evaluation belongs to another dataset freeze.")
        if record["face_config"] != asdict(service.config):
            raise ValueError("Face-pipeline evaluation uses other preprocessing settings.")
        for method, predictor in service.predictors.items():
            filename = "svm.npz" if method == "classical" else "best.pt"
            disk_sha256 = sha256_file(predictor.directory / filename)
            if getattr(predictor, "artifact_sha256", disk_sha256) != disk_sha256 or disk_sha256 != record["artifact_sha256"][method]:
                raise ValueError("Face-pipeline evaluation belongs to other served artifacts.")
        result["face_pipeline"] = {"status": "complete", "input_scope": record["scope"],
            "coverage": record["coverage"], "attempted": record["attempted"], "accepted": record["accepted"],
            "rejected": record["rejected"], "rejection_codes": record["rejection_codes"],
            "methods": {name: {"accepted_accuracy_percent": value["accepted_image_metrics"]["accuracy"] * 100 if value["accepted_image_metrics"] else None,
                "accepted_macro_f1": value["accepted_image_metrics"]["macro_f1"] if value["accepted_image_metrics"] else None,
                "end_to_end_correct_percent": value["end_to_end_correct_rate"] * 100}
                for name, value in record["methods"].items()}, "independent_local_pilot": False}
    return result

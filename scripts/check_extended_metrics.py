"""Independent rank-based ROC and scalar box-matching AP verification."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import sha256_file, utc_now, write_json


def rank_auc(labels, scores):
    ranked = sorted(zip(scores, labels, strict=True))
    positive_rank_sum, start = 0.0, 0
    while start < len(ranked):
        stop = start + 1
        while stop < len(ranked) and ranked[stop][0] == ranked[start][0]:
            stop += 1
        mean_rank = (start + 1 + stop) / 2
        positive_rank_sum += mean_rank * sum(label for _, label in ranked[start:stop])
        start = stop
    positives = sum(labels)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def iou(a, b):
    overlap = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return overlap / (a[2] * a[3] + b[2] * b[3] - overlap)


def scalar_ap(gt, predictions, threshold):
    ranked = sorted([(pred["score"], name, index, pred["bbox"]) for name, item in predictions.items()
                     for index, pred in enumerate(item["predictions"])], key=lambda value: (-value[0], value[1], value[2]))
    total = sum(len(item["boxes"]) for item in gt.values())
    used = {name: set() for name in gt}
    tp, fp, ignored, points = 0, 0, 0, []
    for _, name, _, box in ranked:
        hits = [iou(box, target) for target in gt[name]["boxes"]]
        best = max(range(len(hits)), key=hits.__getitem__) if hits else None
        if best is not None and hits[best] >= threshold:
            if best in used[name]:
                fp += 1
            else:
                tp += 1
                used[name].add(best)
        elif any(iou(box, target) >= threshold for target in gt[name]["ignore_boxes"]):
            ignored += 1
            continue
        else:
            fp += 1
        points.append((tp / total, tp / (tp + fp)))
    envelope, maximum = [], 0.0
    for _, precision in reversed(points):
        maximum = max(maximum, precision)
        envelope.append(maximum)
    envelope.reverse()
    area, last = 0.0, 0.0
    for (recall, _), precision in zip(points, envelope, strict=True):
        area += (recall - last) * precision
        last = recall
    return {"ap": area, "true_positives": tp, "false_positives": fp, "ignored_predictions": ignored}


def main():
    directory = ROOT / "reports/extended_evaluation"
    classification = json.loads((directory / "classification_roc.json").read_text())
    detection = json.loads((directory / "detection_ap.json").read_text())
    roc_checks = {}
    artifacts_dir = (ROOT / classification.get("artifacts_dir", "artifacts")).resolve(strict=True)
    if not artifacts_dir.is_relative_to(ROOT):
        raise ValueError("ROC artifact path must remain inside the workspace.")
    for method, value in classification["methods"].items():
        path = artifacts_dir / method / "validation_predictions.csv"
        if sha256_file(artifacts_dir / method / ("svm.npz" if method == "classical" else "best.pt")) != value["artifact_sha256"]:
            raise ValueError("ROC trained artifact changed.")
        if sha256_file(path) != value["predictions_sha256"]:
            raise ValueError("ROC prediction bytes changed.")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        prefix = "margin" if "margin_0" in rows[0] else "score"
        curves = {}
        for label in range(3):
            expected = rank_auc([int(row["label"]) == label for row in rows], [float(row[f"{prefix}_{label}"]) for row in rows])
            actual = value["roc"]["per_class"][str(label)]["auc"]
            if not math.isclose(expected, actual, abs_tol=1e-12):
                raise ValueError("Rank-derived AUC disagrees with trapezoidal ROC.")
            curves[str(label)] = expected
        correct = sum(int(row["label"]) == int(row["prediction"]) for row in rows)
        if not math.isclose(value["accuracy_percent"], 100 * correct / len(rows), abs_tol=1e-12):
            raise ValueError("Accuracy percentage differs from saved predictions.")
        roc_checks[method] = {"rank_auc": curves, "accuracy_percent": 100 * correct / len(rows), "rows": len(rows)}
    path = directory / "detection_predictions.json"
    if sha256_file(path) != detection["predictions_sha256"]:
        raise ValueError("Detection prediction bytes changed.")
    predictions = json.loads(path.read_text())["images"]
    # The independently implemented scalar matcher shares only the validated
    # annotation-file parser, not the vector IoU/matching/AP implementation.
    spec = importlib.util.spec_from_file_location("extended_metric_parser", ROOT / "scripts/evaluate_extended_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    annotation = ROOT / "data/detection_benchmark/extracted/wider_face_split/wider_face_val_bbx_gt.txt"
    if sha256_file(annotation) != detection["annotation_sha256"]:
        raise ValueError("Detection ground truth changed.")
    gt = module.parse_wider(annotation)
    if set(gt) != set(predictions):
        raise ValueError("All benchmark images must be evaluated.")
    ap_checks = {}
    for name, actual in detection["metrics"].items():
        expected = scalar_ap(gt, predictions, actual["iou_threshold"])
        if any(not math.isclose(actual[key], value, rel_tol=1e-10, abs_tol=1e-12) for key, value in expected.items()):
            raise ValueError("Independent scalar AP disagrees with reported metrics.")
        ap_checks[name] = expected
        print(json.dumps({name: expected}), flush=True)
    write_json(directory / "independent_review.json", {"ok": True, "checked_utc": utc_now(),
        "roc": roc_checks, "detection": ap_checks, "images": len(gt),
        "evidence_checksums": {name: sha256_file(directory / name) for name in ("classification_roc.json", "detection_ap.json", "detection_predictions.json")},
        "scope": "Rank-sum AUC, saved-label accuracy, independent scalar IoU/greedy matching/AP; parser shared. No identity recognition or extra age test inference."})


if __name__ == "__main__":
    main()

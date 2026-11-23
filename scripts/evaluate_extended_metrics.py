"""ROC/AUC and accuracy from saved classification scores; real detector AP.

Classification uses saved validation predictions without new image inference.
Detection uses independently annotated WIDER FACE validation boxes with a stated
general all-valid-box AP protocol, not the official difficulty-subset leaderboard.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import configure_caches, sha256_file, utc_now, write_json
configure_caches(ROOT)

import numpy as np
from PIL import Image
from visage.data import load_manifest
from visage.evaluation_metrics import detection_ap, multiclass_roc

OUTPUT = ROOT / "reports/extended_evaluation"
METHODS = ("classical", "visagenet", "convnext_ordinal")
NAMES = ("Under 18", "18-59", "60+")


def plots(report, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    if kind == "classification":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True)
        for axis, (method, result) in zip(axes, report["methods"].items(), strict=True):
            for label, curve in result["roc"]["per_class"].items():
                axis.plot(curve["fpr"], curve["tpr"], label=f"{NAMES[int(label)]}: {curve['auc']:.4f}")
            axis.plot([0, 1], [0, 1], "k--", linewidth=.8)
            axis.set(title=method, xlabel="False positive rate", xlim=(0, 1), ylim=(0, 1.01))
            axis.legend(fontsize=14)
            axis.tick_params(labelsize=14)
            axis.xaxis.label.set_size(14)
            axis.title.set_size(14)
        axes[0].set_ylabel("True positive rate", fontsize=14)
        fig.suptitle("Age-group one-vs-rest ROC: frozen validation scores")
        path = OUTPUT / "age_group_roc.png"
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for name, result in report["metrics"].items():
            axes[0].plot(result["recall"], result["precision"], label=f"{name}: AP {result['ap']:.4f}")
        result = report["metrics"]["AP50"]
        axes[1].plot(result["false_positives_per_image"], result["recall"])
        axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01), title="Box detection precision-recall")
        axes[0].legend()
        axes[1].set(xlabel="False positives per image", ylabel="Face recall", ylim=(0, 1.01), title="Detection FROC at IoU 0.50")
        path = OUTPUT / "detection_ap_pr_froc.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def classification(artifacts_dir=ROOT / "artifacts"):
    artifacts_dir = Path(artifacts_dir).resolve(strict=True)
    if not artifacts_dir.is_relative_to(ROOT):
        raise ValueError("Model evidence must be inside this workspace.")
    manifest = ROOT / "data/processed/manifest.csv"
    rows = [row for row in load_manifest(manifest) if row["split"] == "validation"]
    results = {}
    for method in METHODS:
        path = artifacts_dir / method / "validation_predictions.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        if len(saved) != len(rows) or any(a["sample_id"] != b["sample_id"] or int(a["label"]) != b["label"] for a, b in zip(saved, rows, strict=True)):
            raise ValueError("Saved scores differ from frozen validation identities or labels.")
        keys = ("margin_0", "margin_1", "margin_2") if "margin_0" in saved[0] else ("score_0", "score_1", "score_2")
        scores = np.array([[float(row[key]) for key in keys] for row in saved])
        if not np.isfinite(scores).all():
            raise ValueError("Classification scores must be finite.")
        truth = np.array([row["label"] for row in rows])
        prediction = np.array([int(row["prediction"]) for row in saved])
        if not np.array_equal(prediction, scores.argmax(1)):
            raise ValueError("Saved class IDs disagree with score argmax.")
        results[method] = {"accuracy_percent": float((prediction == truth).mean() * 100),
            "roc": multiclass_roc(truth, scores), "predictions_sha256": sha256_file(path),
            "artifact_sha256": sha256_file(artifacts_dir / method / ("svm.npz" if method == "classical" else "best.pt")),
            "score_type": "uncalibrated SVM margins" if method == "classical" else "uncalibrated derived neural class values"}
    report = {"ok": True, "task": "age_group_classification", "split": "validation", "samples": len(rows),
        "manifest_sha256": sha256_file(manifest), "methods": results, "completed_utc": utc_now(),
        "artifacts_dir": artifacts_dir.relative_to(ROOT).as_posix(),
        "identity_recognition": "Not performed; accuracy refers to the three age groups, not recognition of a person's identity.",
        "test_status": "Uses saved validation scores only; no new test-set inference or tuning."}
    write_json(OUTPUT / "classification_roc.json", report)
    plots(report, "classification")
    print(json.dumps({method: {"accuracy_percent": value["accuracy_percent"], "macro_auc": value["roc"]["macro_auc_mean"]} for method, value in results.items()}), flush=True)


def parse_wider(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    records, index = {}, 0
    while index < len(lines):
        name = lines[index]
        pure = Path(name)
        if pure.is_absolute() or ".." in pure.parts or ":" in name or name in records or not name.endswith(".jpg"):
            raise ValueError("Unexpected WIDER annotation image path.")
        count = int(lines[index + 1])
        if count < 0:
            raise ValueError("Negative annotation count.")
        index += 2
        boxes, ignored = [], []
        for _ in range(count):
            values = [int(value) for value in lines[index].split()]
            if len(values) != 10:
                raise ValueError("Expected ten WIDER face annotation fields.")
            if values[2] > 0 and values[3] > 0:
                (ignored if values[7] else boxes).append(values[:4])
            index += 1
        if count == 0 and index < len(lines) and not lines[index].endswith(".jpg"):
            if any(int(value) != 0 for value in lines[index].split()):
                raise ValueError("Unexpected zero-face placeholder.")
            index += 1
        records[name] = {"boxes": boxes, "ignore_boxes": ignored}
    return records


def detection():
    from visage.faces import FaceConfig, FaceDetector
    import cv2
    cv2.setNumThreads(2)
    data = ROOT / "data/detection_benchmark"
    provenance = json.loads((data / "provenance.json").read_text())
    for name, value in provenance["files"].items():
        if sha256_file(data / Path(name).name) != value["sha256"]:
            raise ValueError("Pinned benchmark archive changed.")
    annotation = data / "extracted/wider_face_split/wider_face_val_bbx_gt.txt"
    gt = parse_wider(annotation)
    if len(gt) != 3226:
        raise ValueError("The complete WIDER validation set must contain 3,226 images.")
    config = FaceConfig(score_threshold=0.05, max_faces=5000)
    detector = FaceDetector(config=config)
    result_path = OUTPUT / "detection_predictions.json"
    identity = {"annotation_sha256": sha256_file(annotation), "provenance_sha256": sha256_file(data / "provenance.json"),
                "detector_sha256": detector.info()["sha256"], "config": asdict(config)}
    saved = json.loads(result_path.read_text()) if result_path.exists() else {**identity, "images": {}}
    if any(saved[key] != value for key, value in identity.items()):
        raise ValueError("Detection evaluation configuration changed; choose a new study directory.")
    for i, name in enumerate(gt):
        path = data / "extracted/WIDER_val/images" / name
        digest = sha256_file(path)
        if name in saved["images"]:
            if digest != saved["images"][name]["file_sha256"]:
                raise ValueError("A detection validation photo changed.")
            continue
        with Image.open(path) as original:
            if original.width * original.height > config.max_pixels or original.getexif().get(274, 1) != 1:
                raise ValueError("Benchmark image dimensions/orientation require a separately declared evaluation policy.")
            image = original.convert("RGB")
            try:
                faces = detector.detect(image)
            finally:
                image.close()
        saved["images"][name] = {"file_sha256": digest, "predictions": [{"bbox": list(face.bbox), "score": face.score} for face in faces]}
        if (i + 1) % 128 == 0 or i + 1 == len(gt):
            write_json(result_path, saved)
            print(json.dumps({"detection_images": i + 1, "total": len(gt)}), flush=True)
    predictions = {name: record["predictions"] for name, record in saved["images"].items()}
    report = {"ok": True, "task": "face_detection", "dataset": "WIDER FACE validation", "images": len(gt),
        "completed_utc": utc_now(), **identity, "predictions_sha256": sha256_file(result_path),
        "detector": detector.info(), "metrics": {"AP50": detection_ap(gt, predictions, .5), "AP75": detection_ap(gt, predictions, .75)},
        "protocol_note": "All noninvalid positive-area boxes; invalid boxes ignored. Continuous xywh IoU and all-points AP. This is not the official WIDER easy/medium/hard evaluation and must not be compared directly with its leaderboard scores.",
        "operating_point_note": "Confidence floor0.05 bounds this AP sweep and its achievable recall; deployed API uses0.9. NMS0.3,maximum image side1280,top5000. Quality/age selection is not applied to detector AP.",
        "generalization_note": "YuNet is pretrained on face-detection data. WIDER validation is a development benchmark, not a new independent local participant pilot."}
    write_json(OUTPUT / "detection_ap.json", report)
    plots(report, "detection")
    print(json.dumps({name: metric["ap_percent"] for name, metric in report["metrics"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("classification", "detection"), required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    classification(args.artifacts_dir) if args.stage == "classification" else detection()


if __name__ == "__main__":
    main()

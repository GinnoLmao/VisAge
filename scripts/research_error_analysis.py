"""Descriptive validation error strata; no tuning, label edits or test images."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image, ImageOps
from visage.data import load_manifest, verify_manifest
from visage.runtime import sha256_file, utc_now, write_json


def main():
    manifest, raw = ROOT / "data/processed/manifest.csv", ROOT / "data/raw/UTKFace"
    verify_manifest(manifest, raw, verify_files=False)
    rows = [row for row in load_manifest(manifest) if row["split"] == "validation"]
    quality = []
    for row in rows:
        path = (raw / row["relative_path"]).resolve()
        if not path.is_relative_to(raw.resolve()) or sha256_file(path) != row["file_sha256"]:
            raise ValueError("Frozen validation file changed.")
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            grey = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            brightness = float(grey.mean())
            sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
            image.close()
        age = int(row["age"])
        age_bin = "below15" if age < 15 else "15_to20" if age <= 20 else "21_to54" if age < 55 else "55_to64" if age <= 64 else "65plus"
        quality.append({"sample_id": row["sample_id"], "label": row["label"], "age_band": age_bin,
                        "brightness": brightness, "laplacian_variance": sharpness,
                        "brightness_band": "dim_below60" if brightness < 60 else "bright_above180" if brightness > 180 else "middle60_to180",
                        "sharpness_band": "low_below100" if sharpness < 100 else "middle100_to500" if sharpness <= 500 else "high_above500"})
    results, prediction_hashes = {}, {}
    for method in ("classical", "visagenet", "convnext_ordinal"):
        path = ROOT / "artifacts" / method / "validation_predictions.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != len(rows) or any(pred["sample_id"] != row["sample_id"] or int(pred["label"]) != row["label"] for pred, row in zip(predictions, rows, strict=True)):
            raise ValueError("Saved predictions and validation manifest differ.")
        strata = {}
        for variable in ("age_band", "brightness_band", "sharpness_band"):
            strata[variable] = {}
            for band in sorted({item[variable] for item in quality}):
                selected = [i for i, item in enumerate(quality) if item[variable] == band]
                correct = sum(int(predictions[i]["prediction"]) == rows[i]["label"] for i in selected)
                strata[variable][band] = {"samples": len(selected), "correct": correct,
                    "errors": len(selected) - correct, "accuracy": correct / len(selected)}
        results[method] = strata
        prediction_hashes[method] = sha256_file(path)
    output = ROOT / "reports/research"
    write_json(output / "error_analysis.json", {
        "split": "validation", "manifest_sha256": sha256_file(manifest), "samples": len(rows),
        "completed_utc": utc_now(), "methods": results, "baseline_predictions_sha256": prediction_hashes,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "quality_definitions": {"brightness": "Mean decoded grayscale; bins <60,60..180,>180",
            "sharpness": "OpenCV Laplacian CV_64F variance at decoded source resolution; bins <100,100..500,>500"},
        "limitations": ["Descriptive exploratory bins are not validated image-quality thresholds or causal explanations.",
            "Provided ages are dataset annotations, not verified chronological ages.",
            "Age/quality strata can be correlated and have different class mixtures; compare supports alongside accuracy.",
            "No source labels were corrected, no test images were opened, and no photos were saved."],
    })
    print(json.dumps({"event": "validation_error_analysis_complete", "samples": len(rows)}))


if __name__ == "__main__":
    main()

"""Evaluate fixed original artifacts on validation-only aligned-crop perturbations.

This does not measure detector robustness on full photographs or use test images.
Stages are independently resumable and never change a trained artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import configure_caches, seed_everything, sha256_file, utc_now, write_json
configure_caches(ROOT)

from PIL import Image, ImageOps
import torch
from visage.data import load_manifest, verify_manifest
from visage.inference import AgeGroupPredictor
from visage.metrics import classification_metrics
from visage.robustness import CONDITIONS, perturb
from visage.training import logits_to_probabilities


def write_predictions(path, rows, predictions):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "label", "prediction"])
        writer.writerows((row["sample_id"], row["label"], pred) for row, pred in zip(rows, predictions, strict=True))


def error_analysis(output, rows):
    bands = {"below15": lambda age: age < 15, "15_to20": lambda age: 15 <= age <= 20,
             "21_to54": lambda age: 21 <= age <= 54, "55_to64": lambda age: 55 <= age <= 64,
             "65plus": lambda age: age >= 65}
    results = {}
    for method in ("classical", "visagenet", "convnext_ordinal"):
        with (ROOT / "artifacts" / method / "validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != len(rows) or any(a["sample_id"] != b["sample_id"] or int(a["label"]) != b["label"] for a, b in zip(predictions, rows, strict=True)):
            raise ValueError("Validation predictions do not match the manifest order/labels.")
        by_band = {}
        for name, predicate in bands.items():
            chosen = [(row, pred) for row, pred in zip(rows, predictions, strict=True) if predicate(int(row["age"]))]
            correct = sum(row["label"] == int(pred["prediction"]) for row, pred in chosen)
            by_band[name] = {"samples": len(chosen), "correct": correct, "errors": len(chosen) - correct,
                             "accuracy": correct / len(chosen) if chosen else None}
        results[method] = by_band
    write_json(output / "error_analysis.json", {"split": "validation", "methods": results,
               "age_note": "Bands use provided UTKFace age annotations, not verified chronological age.",
               "selection_note": "Descriptive error analysis; no automatic label correction or test-image access."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("classical", "neural", "merge"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/research")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data/processed/manifest.csv"
    raw = ROOT / "data/raw/UTKFace"
    verify_manifest(manifest, raw, verify_files=False)
    rows = [row for row in load_manifest(manifest) if row["split"] == "validation"]
    truth = [row["label"] for row in rows]
    if not rows or set(truth) != {0, 1, 2}:
        raise ValueError("All validation classes are required.")
    identity = {"manifest_sha256": sha256_file(manifest), "split": "validation", "samples": len(rows),
                "conditions": CONDITIONS, "perturbation_source_sha256": sha256_file(ROOT / "src/visage/robustness.py")}
    if args.stage == "merge":
        records = [json.loads((output / f"robustness_{stage}.json").read_text()) for stage in ("classical", "neural")]
        if any(any(record[key] != value for key, value in identity.items()) for record in records):
            raise ValueError("Robustness stages have mismatched provenance.")
        methods = {key: value for record in records for key, value in record["methods"].items()}
        if set(methods) != {"classical", "visagenet", "convnext_ordinal"}:
            raise ValueError("All three methods are required.")
        write_json(output / "robustness.json", {**identity, "methods": methods, "completed_utc": utc_now(),
            "scope": "Fixed original artifacts; aligned-crop corruption validation, not full-photo detector or local-pilot validation."})
        print("Merged all three robustness methods.")
        return
    seed_everything(42, threads=4)
    # Read and hash only validation images. The test partition is never decoded.
    for row in rows:
        path = (raw / row["relative_path"]).resolve()
        if not path.is_relative_to(raw.resolve()) or sha256_file(path) != row["file_sha256"]:
            raise ValueError("Frozen validation image changed.")
    methods = ("classical",) if args.stage == "classical" else ("visagenet", "convnext_ordinal")
    destination = output / f"robustness_{args.stage}.json"
    previous = json.loads(destination.read_text()) if destination.exists() else {**identity, "methods": {}}
    if any(previous[key] != value for key, value in identity.items()):
        raise ValueError("Refusing to reuse robustness results from a changed protocol.")
    for method in methods:
        device = "cpu" if method == "classical" else args.device
        adapter = AgeGroupPredictor(method, ROOT / "artifacts" / method, device=device)
        artifact = ROOT / "artifacts" / method / ("svm.npz" if method == "classical" else "best.pt")
        artifact_sha = sha256_file(artifact)
        record = previous["methods"].setdefault(method, {"artifact_sha256": artifact_sha, "conditions": {}})
        if record["artifact_sha256"] != artifact_sha:
            raise ValueError("Robustness artifact changed.")
        for condition in CONDITIONS:
            if condition in record["conditions"]:
                saved = record["conditions"][condition]
                if sha256_file(output / saved["predictions_csv"]) != saved["predictions_sha256"]:
                    raise ValueError("Saved robustness predictions changed.")
                continue
            started = time.perf_counter()
            predictions, batch = [], []

            def flush(model=adapter.model, run_device=device):
                if batch:
                    with torch.inference_mode():
                        values = logits_to_probabilities(model(torch.stack(batch).to(run_device)), model.head_type)
                        if not torch.isfinite(values).all():
                            raise RuntimeError("Nonfinite robustness predictions.")
                        predictions.extend(values.argmax(1).cpu().tolist())
                    batch.clear()

            for row in rows:
                with Image.open(raw / row["relative_path"]) as original:
                    source = ImageOps.exif_transpose(original).convert("RGB")
                    changed = perturb(source, condition)
                    try:
                        if method == "classical":
                            predictions.append(int(adapter.model.predict_image(changed)))
                        else:
                            batch.append(adapter.transform(changed))
                            if len(batch) == 32:
                                flush()
                    finally:
                        changed.close()
                        source.close()
            flush()
            csv_path = output / f"robustness_{method}_{condition}.csv"
            write_predictions(csv_path, rows, predictions)
            metrics = classification_metrics(truth, predictions)
            clean_f1 = record["conditions"].get("clean", {}).get("metrics", metrics)["macro_f1"]
            record["conditions"][condition] = {"metrics": metrics, "macro_f1_drop_from_clean": clean_f1 - metrics["macro_f1"],
                "predictions_csv": csv_path.name, "predictions_sha256": sha256_file(csv_path),
                "evaluation_seconds": time.perf_counter() - started}
            write_json(destination, previous)
            print(json.dumps({"method": method, "condition": condition, "macro_f1": metrics["macro_f1"]}), flush=True)
        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()
    previous["completed_utc"] = utc_now()
    write_json(destination, previous)
    error_analysis(output, rows)


if __name__ == "__main__":
    main()

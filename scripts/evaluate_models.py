"""Evaluate frozen models on one partition and benchmark reloaded artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import configure_caches
configure_caches(ROOT)

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader
from visage import CLASS_NAMES
from visage.data import load_manifest, verify_manifest
from visage.datasets import ManifestDataset
from visage.inference import AgeGroupPredictor
from visage.metrics import classification_metrics
from visage.training import evaluate_model, save_predictions, logits_to_probabilities
from visage.runtime import environment_info, seed_everything, sha256_file, utc_now, write_json


def distribution_ms(values):
    values = np.asarray(values) * 1000
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(values.mean()), "measurements": len(values)}


def benchmark_neural(model, tensor, device, repeats=30):
    model.to(device).eval()
    tensor = tensor.to(device)
    samples = []
    with torch.inference_mode():
        for iteration in range(repeats + 5):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            probabilities = logits_to_probabilities(model(tensor), model.head_type)
            if device.type == "cuda":
                torch.cuda.synchronize()
            duration = time.perf_counter() - start
            if iteration >= 5:
                samples.append(duration)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Invalid benchmark output.")
    return distribution_ms(samples)


def plot_outputs(output, summaries, artifacts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, summary in summaries.items():
        cm = np.array(summary["metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        image = ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel="Predicted age group", ylabel="Provided age group", title=f"{name}: held-out confusion counts")
        fig.colorbar(image, ax=ax, label="Images")
        fig.tight_layout()
        fig.savefig(output / f"{name}_confusion.png", dpi=170)
        plt.close(fig)
    for name in ("visagenet", "convnext_ordinal"):
        path = artifacts / name / "history.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(epochs, [x["train_loss"] for x in history], label="Augmented balanced training")
        axes[0].plot(epochs, [x["val_loss"] for x in history], label="Validation")
        axes[0].set(xlabel="Epoch", ylabel="Conditional BCE per eligible observation", title=f"{name} loss")
        axes[0].legend(fontsize=8)
        axes[1].plot(epochs, [x["val_accuracy"] for x in history], label="Validation accuracy")
        axes[1].plot(epochs, [x["val_macro_f1"] for x in history], label="Validation macro F1")
        axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation selection history", ylim=(0, 1))
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{name}_training_curves.png", dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/UTKFace"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    seed_everything(42)
    verify_manifest(args.manifest, args.raw_dir, verify_files=True)
    all_rows = load_manifest(args.manifest)
    rows = [r for r in all_rows if r["split"] == args.split]
    if set(int(r["label"]) for r in rows) != {0, 1, 2}:
        raise ValueError("Evaluation split must contain all three classes.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_file(args.manifest)
    truth = np.array([r["label"] for r in rows])
    with Image.open(args.raw_dir / rows[0]["relative_path"]) as image:
        example = ImageOps.exif_transpose(image).convert("RGB")
    results = {}
    classical_dir = args.artifacts_dir / "classical"
    selection = json.loads((classical_dir / "selection.json").read_text())
    if selection["manifest_sha256"] != manifest_sha:
        raise ValueError("Classical model belongs to another dataset freeze.")
    classical = AgeGroupPredictor("classical", classical_dir).model
    predictions = []
    with (args.output_dir / "classical_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "relative_path", "age", "label", "prediction"])
        for i, row in enumerate(rows):
            with Image.open(args.raw_dir / row["relative_path"]) as image:
                prediction = classical.predict_image(ImageOps.exif_transpose(image).convert("RGB"))
            predictions.append(int(prediction))
            writer.writerow([row["sample_id"], row["relative_path"], row["age"], row["label"], prediction])
            if (i + 1) % 512 == 0:
                print(json.dumps({"event": "evaluation_progress", "method": "classical", "images": i + 1}), flush=True)
    times = []
    for i in range(35):
        start = time.perf_counter()
        classical.predict_image(example)
        if i >= 5:
            times.append(time.perf_counter() - start)
    results["classical"] = {"metrics": classification_metrics(truth, predictions), "cpu_batch1": distribution_ms(times),
                            "parameters_rubric": 0, "artifact_bytes": sum(p.stat().st_size for p in classical_dir.glob("svm.*")),
                            "training_seconds": selection["selected_fit_seconds"], "search_seconds": selection["search_seconds"],
                            "validation_macro_f1": selection["validation_metrics"]["macro_f1"],
                            "timing_scope": "PIL aligned face -> HOG/LBP features -> scaler -> SVM prediction"}
    print(json.dumps({"method": "classical", **results["classical"]}), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for method in ("visagenet", "convnext_ordinal"):
        adapter = AgeGroupPredictor(method, args.artifacts_dir / method, device=device)
        model, checkpoint, transforms = adapter.model, adapter.metadata, adapter.transform
        del adapter
        if checkpoint["manifest_sha256"] != manifest_sha:
            raise ValueError(f"{method} belongs to another dataset freeze.")
        dataset = ManifestDataset(rows, args.raw_dir, transforms)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        evaluation = evaluate_model(model, loader, device, None)
        save_predictions(args.output_dir / f"{method}_predictions.csv", rows, evaluation)
        summary = json.loads((args.artifacts_dir / method / "training_summary.json").read_text())
        tensor = transforms(example).unsqueeze(0)
        cpu_benchmark = benchmark_neural(model, tensor, torch.device("cpu"))
        gpu_benchmark = benchmark_neural(model, tensor, torch.device("cuda")) if torch.cuda.is_available() else None
        # Include tensor preparation in a separate directly comparable CPU measurement.
        model.cpu().eval()
        complete_times = []
        with torch.inference_mode():
            for i in range(35):
                start = time.perf_counter()
                logits_to_probabilities(model(transforms(example).unsqueeze(0)), model.head_type)
                if i >= 5:
                    complete_times.append(time.perf_counter() - start)
        results[method] = {"metrics": evaluation["metrics"], "cpu_batch1": cpu_benchmark,
                           "cpu_preprocess_and_inference_batch1": distribution_ms(complete_times), "gpu_batch1": gpu_benchmark,
                           "parameters": summary["parameters"], "artifact_bytes": summary["artifact_bytes"],
                           "training_seconds": summary["training_seconds"],
                           "validation_macro_f1": checkpoint["validation_metrics"]["macro_f1"],
                           "best_epoch": checkpoint["epoch"], "timing_scope": "network and output conversion; preprocessing separately reported"}
        print(json.dumps({"method": method, **results[method]}), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # The ranking uses validation evidence only, not the test results above.
    recommended = max(results, key=lambda key: results[key]["validation_macro_f1"])
    write_json(args.output_dir / "comparison.json", {"split": args.split, "manifest_sha256": manifest_sha,
               "completed_utc": utc_now(), "environment": environment_info(), "methods": results,
               "recommendation": {"method": recommended, "basis": "highest validation macro F1; test results are reporting only"},
               "benchmark_note": "Single aligned image repeated after five warmups; 4 CPU threads; batch size one. Face detection, disk decode and network transport excluded."})
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "accuracy", "macro_f1", "under18_predicted_older_rate", "senior_predicted_below60_rate", "cpu_pipeline_median_ms", "cpu_core_median_ms", "parameters_m", "training_minutes", "artifact_mb"])
        for method, values in results.items():
            inclusive_timing = values.get("cpu_preprocess_and_inference_batch1", values["cpu_batch1"])
            writer.writerow([method, values["metrics"]["accuracy"], values["metrics"]["macro_f1"],
                             values["metrics"]["under18_predicted_older_rate"], values["metrics"]["senior_predicted_below60_rate"],
                             inclusive_timing["median_ms"], values["cpu_batch1"]["median_ms"], values.get("parameters", 0) / 1e6,
                             values["training_seconds"] / 60, values["artifact_bytes"] / 1e6])
    plot_outputs(args.output_dir, results, args.artifacts_dir)


if __name__ == "__main__":
    main()

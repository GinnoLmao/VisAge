"""Check actual CPU/CUDA service prediction parity on fixed validation inputs.

This is a functional GPU smoke check, not a latency benchmark or a claim that
floating-point outputs are bitwise equal across devices.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visage.runtime import configure_caches, seed_everything, sha256_file, utc_now, write_json


def signature(result):
    return {"manifest_sha256": result["manifest_sha256"],
            "crop_sha256": result["face_processing"]["pixel_sha256"],
            "selected_face": result["selected_face"],
            "predictions": [{"method": item["method"], "class_id": item["class_id"], "age_group": item["age_group"]}
                            for item in result["predictions"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/backend_cuda_smoke.json")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--training-idle-confirmed", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples <= 30:
        parser.error("Use between one and thirty fixed validation inputs.")
    if not args.training_idle_confirmed:
        parser.error("Run after research training finishes and pass --training-idle-confirmed.")
    if args.output.exists():
        raise FileExistsError("Choose a new smoke-report path; preserve earlier results.")
    if not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Smoke reports must stay inside the project workspace.")
    configure_caches(ROOT)
    seed_everything(42, threads=4)
    import torch
    from visage.service import InferenceService
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke check requires an actual CUDA device.")
    manifest = ROOT / "data/processed/manifest.csv"
    manifest_sha = sha256_file(manifest)
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "validation"][:args.samples]
    if len(rows) != args.samples:
        raise ValueError("Not enough frozen validation inputs are available.")
    artifact_paths = {method: args.artifacts_dir / method / filename for method, filename in (
        ("classical", "svm.npz"), ("visagenet", "best.pt"), ("convnext_ordinal", "best.pt"))}
    artifact_sha = {name: sha256_file(path) for name, path in artifact_paths.items()}
    raw = (ROOT / "data/raw/UTKFace").resolve()
    cpu = gpu = None
    records = []
    try:
        cpu = InferenceService(args.artifacts_dir, device="cpu")
        gpu = InferenceService(args.artifacts_dir, device="cuda")
        for row in rows:
            path = (raw / row["relative_path"]).resolve(strict=True)
            if not path.is_relative_to(raw) or path.stat().st_size > cpu.config.max_bytes or sha256_file(path) != row["file_sha256"]:
                raise ValueError("A frozen validation input changed or exceeds admission limits.")
            data = path.read_bytes()
            cpu_result = cpu.predict(data, method="all")
            gpu_result = gpu.predict(data, method="all")
            cpu_signature, gpu_signature = signature(cpu_result), signature(gpu_result)
            if cpu_signature["manifest_sha256"] != manifest_sha or gpu_signature["manifest_sha256"] != manifest_sha:
                raise ValueError("Loaded artifacts differ from the frozen validation manifest.")
            if [item["device"] for item in gpu_result["predictions"]] != ["cpu", "cuda", "cuda"]:
                raise ValueError("The GPU service did not actually execute neural methods on CUDA.")
            records.append({"sample_id": row["sample_id"], "input_sha256": row["file_sha256"],
                            "cpu": cpu_signature, "cuda": gpu_signature,
                            "matching_crop": cpu_signature["crop_sha256"] == gpu_signature["crop_sha256"],
                            "matching_predictions": cpu_signature["predictions"] == gpu_signature["predictions"],
                            "matching_signature": cpu_signature == gpu_signature})
        if any(sha256_file(path) != artifact_sha[name] for name, path in artifact_paths.items()):
            raise ValueError("An artifact changed during the functional GPU check.")
        result = {"ok": all(row["matching_signature"] for row in records), "recorded_utc": utc_now(),
                  "samples": len(records), "input_partition": "validation", "manifest_sha256": manifest_sha,
                  "artifact_sha256": artifact_sha, "detector": cpu.describe()["detector"],
                  "devices": ["cpu", "cuda"], "cuda_device": torch.cuda.get_device_name(0),
                  "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
                  "records": records, "all_crops_match": all(row["matching_crop"] for row in records),
                  "all_labels_match": all(row["matching_predictions"] for row in records),
                  "training_idle_confirmed": True,
                  "scope": "Actual CPU/CUDA shared-face service functional check. Five (by default) fixed validation images; no test images, no latency measurement, no numerical score-equality claim or photos saved."}
        write_json(args.output, result)
        print(json.dumps({"ok": result["ok"], "samples": len(records), "all_crops_match": result["all_crops_match"],
                          "all_labels_match": result["all_labels_match"], "output": str(args.output)}, indent=2))
        if not result["ok"]:
            raise RuntimeError("A CPU/CUDA crop or predicted label differs; inspect the recorded smoke evidence.")
    finally:
        if cpu is not None:
            cpu.close()
        if gpu is not None:
            gpu.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

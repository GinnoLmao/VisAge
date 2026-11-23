"""Validate consented local data, then evaluate a declared development/holdout split.

No images are collected by this command. No photo bytes are written to outputs.
Age-group labels must come from reliable permitted records, never model guesses.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.pilot import run_pilot_rows, validate_pilot_manifest
from visage.runtime import configure_caches, sha256_file, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--photo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new pilot output directory; prior results are immutable.")
    rows, validation = validate_pilot_manifest(args.manifest, args.photo_root)
    if args.validate_only:
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / "manifest_validation.json", {**validation, "status": "validated_only", "inference_run": False})
        print("Manifest validated. No model inference was run.")
        return
    configure_caches(ROOT)
    from visage.service import InferenceService
    from visage.model_registry import ARTIFACTS, METHODS
    args.artifacts_dir = args.artifacts_dir or ARTIFACTS
    service = InferenceService(artifacts_dir=args.artifacts_dir, device="cpu")
    try:
        result = run_pilot_rows(rows, args.photo_root,
            lambda data, face_index: service.predict(data, method="all", face_index=face_index), split=args.split)
        artifacts = {name: sha256_file(args.artifacts_dir / name / filename) for name, filename in ((method, "best.pt") for method in METHODS)}
        result.update(manifest_validation=validation, artifact_sha256=artifacts,
                      face_pipeline=service.describe(), completed_utc=utc_now(),
                      evaluation_note="Holdout results must not guide subsequent tuning; independent confirmation requires new participants.")
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / "results.json", result)
        print(f"Pilot complete: {result['accepted']}/{result['attempted']} accepted; results in {args.output_dir}")
    finally:
        service.close()


if __name__ == "__main__":
    main()

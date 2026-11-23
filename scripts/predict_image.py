"""Detect/select a face in an image and run the same path as the HTTP API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.model_registry import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--method", choices=("all", *METHODS), default=None)
    parser.add_argument("--face-index", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--detector-path", type=Path)
    parser.add_argument("--faces-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from visage.faces import FaceConfig, FaceError
    from visage.runtime import configure_caches, seed_everything, write_json
    from visage.service import InferenceService
    configure_caches(ROOT)
    seed_everything(42, threads=4)
    service = None
    try:
        if args.image.stat().st_size > FaceConfig().max_bytes:
            raise FaceError("IMAGE_TOO_LARGE", "Image exceeds the 8 MiB input limit.", status_code=413)
        data = args.image.read_bytes()
        service = InferenceService(args.artifacts_dir, device=args.device, detector_path=args.detector_path)
        result = service.faces(data) if args.faces_only else service.predict(data, method=args.method, face_index=args.face_index)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except FaceError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message, "details": exc.details}}, allow_nan=False), file=sys.stderr)
        return 2
    finally:
        if service is not None:
            service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

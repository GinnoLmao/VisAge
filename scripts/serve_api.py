"""Serve the local VisAge backend with one persistent model process."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--detector-path", type=Path)
    parser.add_argument("--allow-origin", action="append", default=[])
    parser.add_argument("--max-pending", default=2, type=int)
    args = parser.parse_args()
    import uvicorn
    from visage.api import APISettings, create_app
    from visage.model_registry import ARTIFACTS, METHODS
    settings = APISettings(artifacts_dir=args.artifacts_dir or ARTIFACTS, detector_path=args.detector_path,
                           device=args.device, allowed_origins=tuple(args.allow_origin), max_pending=args.max_pending)
    # A single process preserves one model set and the shared concurrency lock.
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1,
                access_log=False, timeout_keep_alive=5, limit_concurrency=16)


if __name__ == "__main__":
    main()

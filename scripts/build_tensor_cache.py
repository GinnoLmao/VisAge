"""Prepare the verified train/validation uint8 image cache without loading CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visage.tensor_cache import ensure_tensor_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/UTKFace")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/neural_faces")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    metadata = ensure_tensor_cache(args.manifest, args.raw_dir, args.cache_dir, args.image_size)
    print(json.dumps({"event": "tensor_cache_complete", "cache_dir": str(args.cache_dir), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()

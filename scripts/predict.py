"""Run trained methods on a selected/aligned face crop, without an app server."""
import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import configure_caches, seed_everything, write_json
configure_caches(ROOT)
from visage.model_registry import METHODS, ARTIFACTS, serving_default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--method", choices=("all", *METHODS), default=None)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from visage.inference import read_face_crop
    from visage.score_inference import ScoredPretrainedPredictor
    args.artifacts_dir = args.artifacts_dir or ARTIFACTS
    seed_everything(42, threads=4)
    image = read_face_crop(args.image)
    methods = METHODS if args.method == "all" else (args.method or serving_default(args.artifacts_dir),)
    predictions = []
    for method in methods:
        adapter = ScoredPretrainedPredictor(method, args.artifacts_dir / method, device=args.device)
        predictions.append(adapter.predict(image))
        del adapter
    image.close()
    result = {"input_assumption": "One selected/aligned face crop; no face detection or age verification is performed.",
              "notice": "Academic age-group estimates; never use alone to approve or refuse a transaction.",
              "predictions": predictions}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

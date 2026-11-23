"""Command line entry point for the UTKFace audit and data freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visage.data import prepare_dataset, verify_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                        metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Intentionally replace an existing data freeze.")
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing freeze without preparing it again.")
    parser.add_argument("--verify-files", action="store_true", help="Decode and rehash every retained image during verification.")
    arguments = parser.parse_args(argv)
    try:
        if not arguments.verify_only:
            audit = prepare_dataset(arguments.raw_dir, arguments.output_dir, seed=arguments.seed,
                                    ratios=tuple(arguments.ratios), phash_distance=arguments.phash_distance,
                                    overwrite=arguments.overwrite)
            print(json.dumps(audit, indent=2))
        verification = verify_manifest(arguments.output_dir / "manifest.csv", arguments.raw_dir,
                                       verify_files=arguments.verify_files)
        print(json.dumps({"verification": verification}, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Consent-manifest validation and honest end-to-end local pilot accounting.

Photos and identifying consent records stay with the operator. Only pseudonymous
sample/participant IDs, provided class labels and inference outcomes are reported.
"""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path, PurePosixPath
from typing import Callable

from .metrics import classification_metrics
from .runtime import sha256_file

PILOT_COLUMNS = ("sample_id", "relative_path", "participant_id", "label", "split", "consent", "face_index")


def validate_pilot_manifest(manifest: Path, photo_root: Path) -> tuple[list[dict], dict]:
    """Validate all rows before any inference; never silently drop a bad sample."""
    photo_root = Path(photo_root).resolve(strict=True)
    if not photo_root.is_dir():
        raise ValueError("Pilot photo root must be a directory.")
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or ()) != len(PILOT_COLUMNS) or set(reader.fieldnames or ()) != set(PILOT_COLUMNS):
            raise ValueError(f"Pilot CSV must have exactly these columns: {', '.join(PILOT_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Pilot manifest is empty.")
    seen_ids, seen_paths = set(), set()
    participants, hashes = {}, {}
    counts = Counter()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Pilot row has missing or extra CSV fields.")
        for key in PILOT_COLUMNS:
            row[key] = row[key].strip()
        if not row["sample_id"] or row["sample_id"] in seen_ids or not row["participant_id"]:
            raise ValueError("Sample IDs must be unique and participant IDs nonempty.")
        if row["consent"].lower() != "true":
            raise ValueError("Each sample requires recorded consent=true before processing.")
        if row["split"] not in {"development", "holdout"} or row["label"] not in {"0", "1", "2"}:
            raise ValueError("Use development/holdout splits and labels 0, 1, 2.")
        participant = row["participant_id"]
        group = (row["split"], row["label"])
        if participant in participants and participants[participant] != group:
            raise ValueError("A participant cannot cross splits or have conflicting labels.")
        participants[participant] = group
        relative = PurePosixPath(row["relative_path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise ValueError("Photo paths must stay inside the pilot photo root.")
        path = (photo_root / str(relative)).resolve(strict=True)
        if not path.is_relative_to(photo_root) or not path.is_file():
            raise ValueError("Photo paths must resolve to files inside the pilot photo root.")
        if path in seen_paths:
            raise ValueError("A photo path cannot occur more than once.")
        if not 0 < path.stat().st_size <= 8 * 1024**2:
            raise ValueError("Pilot photos must be nonempty and at most 8 MiB.")
        digest = sha256_file(path)
        if digest in hashes:
            raise ValueError("Duplicate photo bytes must be removed before the pilot.")
        hashes[digest] = row["sample_id"]
        index = row["face_index"]
        if index and (not index.isascii() or not index.isdigit()):
            raise ValueError("face_index must be blank or a nonnegative integer.")
        row.update(label=int(row["label"]), face_index=int(index) if index else None,
                   file_sha256=digest, relative_path=relative.as_posix())
        seen_ids.add(row["sample_id"])
        seen_paths.add(path)
        counts[(row["split"], row["label"])] += 1
    holdout_labels = {row["label"] for row in rows if row["split"] == "holdout"}
    if holdout_labels != {0, 1, 2}:
        raise ValueError("Pilot holdout must include all three age groups.")
    return rows, {
        "manifest_sha256": sha256_file(manifest), "samples": len(rows),
        "participants": len(participants),
        "counts": {split: {str(label): counts[(split, label)] for label in range(3)}
                   for split in ("development", "holdout")},
        "consent_recorded": True, "participant_disjoint": True,
        "identity_limit": "Pseudonymous IDs and byte checks do not independently prove identity or chronological age.",
    }


def run_pilot_rows(rows: list[dict], photo_root: Path, predict: Callable, *, split="holdout") -> dict:
    """Run one declared split; rejected images remain in the coverage denominator.

    ``predict(bytes, face_index=...)`` must return a predictions list with all
    three methods. Input/crop failures must expose a ``code`` attribute.
    Unexpected internal failures propagate instead of becoming omitted samples.
    """
    if split not in {"development", "holdout"}:
        raise ValueError("Unknown pilot split.")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError("The requested pilot split is empty.")
    from .model_registry import METHODS
    methods = METHODS
    outcomes, accepted_truth = [], []
    predicted = {method: [] for method in methods}
    for row in selected:
        path = (Path(photo_root).resolve() / row["relative_path"]).resolve(strict=True)
        if not path.is_relative_to(Path(photo_root).resolve()) or sha256_file(path) != row["file_sha256"]:
            raise ValueError("Pilot photo changed after validation.")
        outcome = {key: row[key] for key in ("sample_id", "participant_id", "label")}
        try:
            result = predict(path.read_bytes(), face_index=row["face_index"])
        except Exception as exc:
            if not hasattr(exc, "code"):
                raise
            outcome.update(status="rejected", error_code=exc.code)
        else:
            values = {item["method"]: item["class_id"] for item in result["predictions"]}
            if len(result["predictions"]) != len(methods) or set(values) != set(methods) or any(type(value) is not int or value not in (0, 1, 2) for value in values.values()):
                raise ValueError("Pilot inference must return one valid class for every method.")
            accepted_truth.append(row["label"])
            for method in methods:
                predicted[method].append(values[method])
            outcome.update(status="accepted", predictions=values)
        outcomes.append(outcome)
    accepted = len(accepted_truth)
    result = {"split": split, "attempted": len(selected), "accepted": accepted,
              "rejected": len(selected) - accepted, "coverage": accepted / len(selected),
              "outcomes": outcomes, "methods": {},
              "note": "Conditional metrics exclude rejected images; end-to-end correct rate includes all attempts. Multiple images of a participant are correlated."}
    for method in methods:
        correct = sum(a == b for a, b in zip(accepted_truth, predicted[method], strict=True))
        result["methods"][method] = {
            "end_to_end_correct_rate": correct / len(selected),
            "accepted_image_metrics": classification_metrics(accepted_truth, predicted[method]) if accepted else None,
        }
    return result

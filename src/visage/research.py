"""Validation-only research bookkeeping and whole-group training subsets."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path

from .runtime import sha256_file, write_json


def identity_sha(rows: list[dict]) -> str:
    values = [(row["sample_id"], int(row["label"])) for row in rows]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def nested_training_subsets(rows: list[dict], manifest_sha: str, fractions=(0.25, 0.5), seed=42) -> dict:
    """Choose nested complete groups, stratified by dominant group label.

    Ratios are approximate because indivisible perceptual groups stay intact.
    The original validation/test membership is never changed or sampled here.
    """
    fractions = tuple(float(value) for value in fractions)
    if not fractions or list(fractions) != sorted(set(fractions)) or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("fractions must be unique increasing values in (0,1].")
    train = [row for row in rows if row["split"] == "train"]
    if not train or {row["label"] for row in train} != {0, 1, 2}:
        raise ValueError("Complete three-class frozen training rows are required.")
    groups = defaultdict(list)
    for row in train:
        groups[row["group_id"]].append(row)
    buckets = defaultdict(list)
    rng = random.Random(seed)
    for group, members in sorted(groups.items()):
        counts = Counter(row["label"] for row in members)
        dominant = min(counts, key=lambda label: (-counts[label], label))
        buckets[dominant].append((rng.random(), group, len(members)))
    for bucket in buckets.values():
        bucket.sort()
    previous, results = set(), {}
    for fraction in fractions:
        selected = set(previous)
        for bucket in buckets.values():
            target = fraction * sum(count for _, _, count in bucket)
            count = 0
            for _, group, size in bucket:
                if count >= target:
                    break
                selected.add(group)
                count += size
        present = {row["label"] for group in selected for row in groups[group]}
        for missing in {0, 1, 2} - present:
            selected.add(next(group for group in sorted(groups) if any(row["label"] == missing for row in groups[group])))
        chosen = [row for row in train if row["group_id"] in selected]
        results[fraction] = {
            "format_version": 1, "split": "train", "manifest_sha256": manifest_sha,
            "selection_seed": seed, "fraction_requested": fraction,
            "fraction_actual": len(chosen) / len(train), "rows": len(chosen),
            "selection_policy": "Seeded per-dominant-label group prefixes; whole groups; nested; approximate ratios.",
            "sample_ids": [row["sample_id"] for row in chosen], "group_ids": sorted(selected),
            "class_counts": [sum(row["label"] == label for row in chosen) for label in range(3)],
        }
        previous = selected
    return results


def write_immutable_json(path: str | Path, value) -> None:
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing research record would change: {path}")
    else:
        write_json(path, value)


def verify_protected_files(root: Path, protected: dict[str, str]) -> None:
    for relative, expected in protected.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"Protected original evidence changed: {relative}")


def promotion_decision(baselines: list[dict], candidates: list[dict]) -> dict:
    """Apply predeclared paired-seed mean criteria, never held-out metrics."""
    if len(baselines) != 3 or len(candidates) != 3:
        return {"promote": False, "reason": "Three paired seeds are required for confirmation."}
    fields = {"macro_f1": "macro_f1", "minor_error": "under18_predicted_older_rate"}
    def averages(values):
        return {
            "macro_f1": sum(value[fields["macro_f1"]] for value in values) / len(values),
            "minor_error": sum(value[fields["minor_error"]] for value in values) / len(values),
            "senior_recall": sum(value["per_class"][2]["recall"] for value in values) / len(values),
        }
    old, new = averages(baselines), averages(candidates)
    checks = {
        "mean_macro_f1_improvement_at_least_0_005": new["macro_f1"] - old["macro_f1"] >= 0.005,
        "minor_error_increase_at_most_0_01": new["minor_error"] - old["minor_error"] <= 0.01,
        "senior_recall_drop_at_most_0_01": old["senior_recall"] - new["senior_recall"] <= 0.01,
    }
    return {"promote": all(checks.values()), "checks": checks, "baseline_mean": old, "candidate_mean": new,
            "reason": "Validation-only paired seeds; original test evidence is not a selection input."}

"""Audit UTKFace images and create reproducible, leakage-aware data partitions.

UTKFace labels are estimated annotations, not verified chronological ages.  A
perceptual-hash group helps keep visually similar images together; it does not
establish that images belong to the same person.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import re
import tempfile
from typing import Any
import warnings

import numpy as np
from PIL import Image, ImageOps


MAX_IMAGE_PIXELS = 20_000_000
CLASS_NAMES = ("under_18", "18_to_59", "60_plus")
SPLITS = ("train", "validation", "test")
MANIFEST_COLUMNS = (
    "sample_id", "relative_path", "age", "label", "width", "height",
    "file_sha256", "pixel_sha256", "phash", "group_id", "split",
)
_INTEGER_COLUMNS = ("age", "label", "width", "height")
_NAME = re.compile(
    r"^(?P<age>\d{1,3})_(?P<gender>[01])_(?P<race>[0-4])_"
    r"(?P<timestamp>\d{14,17})\.jpe?g(?:\.chip\.jpg)?$", re.IGNORECASE,
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PHASH = re.compile(r"^[0-9a-f]{16}$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_DCT_BASIS = np.cos(np.pi * (np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None] / 32)
_DCT_BASIS[0] *= np.sqrt(1 / 32)
_DCT_BASIS[1:] *= np.sqrt(2 / 32)


def age_to_label(age: int) -> int:
    """Convert a provided UTKFace age annotation to the common class ID."""
    if isinstance(age, bool) or not isinstance(age, (int, np.integer)):
        raise ValueError("Age must be an integer annotation.")
    if not 0 <= age <= 116:
        raise ValueError("UTKFace age annotation must be between 0 and 116.")
    return 0 if age < 18 else 1 if age < 60 else 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phash(image: Image.Image) -> str:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    # Only the first eight orthonormal DCT-II coefficients are needed.
    low = _DCT_BASIS @ np.asarray(gray, dtype=np.float64) @ _DCT_BASIS.T
    values = low.reshape(-1)
    bits = values > np.median(values[1:])
    bits[0] = False  # Exclude average brightness from the descriptor.
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _read_image(path: Path) -> dict[str, Any]:
    """Fully decode the normalized RGB image before accepting it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as original:
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise ValueError("Decoded image exceeds the maximum pixel count.")
            original.load()
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.load()
            width, height = image.size
            pixel_digest = hashlib.sha256(
                f"RGB:{width}:{height}:".encode("ascii") + image.tobytes()
            ).hexdigest()
            result = {
                "width": width, "height": height,
                "pixel_sha256": pixel_digest, "phash": _phash(image),
            }
            image.close()
    result["file_sha256"] = _sha256(path)
    return result


class _BKTree:
    """BK-tree for radius searches on 64-bit Hamming descriptors."""

    def __init__(self) -> None:
        self.root: tuple[int, int, dict] | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, index, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return  # A representative is sufficient for grouping.
            if distance not in node[2]:
                node[2][distance] = (value, index, {})
                return
            node = node[2][distance]

    def query(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []
        found, stack = [], [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                found.append(node[1])
            lower, upper = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if lower <= edge <= upper)
        return found


def _make_groups(records: list[dict], distance: int) -> dict[str, list[dict]]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parents[max(first, second)] = min(first, second)

    tree = _BKTree()
    for index, record in enumerate(records):
        value = int(record["phash"], 16)
        for matched in tree.query(value, distance):
            union(index, matched)
        tree.add(value, index)
    clusters: dict[int, list[dict]] = {}
    for index, record in enumerate(records):
        clusters.setdefault(find(index), []).append(record)
    groups = {}
    for members in clusters.values():
        identity = "\n".join(sorted(record["pixel_sha256"] for record in members))
        group_id = hashlib.sha256(f"phash-{distance}\n{identity}".encode("ascii")).hexdigest()
        for record in members:
            record["group_id"] = group_id
        groups[group_id] = members
    return groups


def _assign_splits(groups: dict[str, list[dict]], seed: int, ratios: tuple[float, ...]) -> None:
    totals = np.zeros(3, dtype=np.float64)
    for members in groups.values():
        for record in members:
            totals[record["label"]] += 1
    desired = np.asarray(ratios)[:, None] * totals[None, :]
    allocation = np.zeros((3, 3), dtype=np.float64)
    desired_size = np.asarray(ratios) * totals.sum()
    randomizer = random.Random(seed)
    ordered = []
    for group_id, members in sorted(groups.items()):
        counts = np.bincount([record["label"] for record in members], minlength=3)
        ordered.append((group_id, members, counts, randomizer.random()))
    ordered.sort(key=lambda item: (-len(item[1]), item[3], item[0]))
    assignments: dict[str, int] = {}

    def cost(candidate: np.ndarray) -> float:
        class_cost = np.sum((candidate - desired) ** 2 / (desired + 1.0))
        size_cost = np.sum((candidate.sum(axis=1) - desired_size) ** 2 / (desired_size + 1.0))
        return float(class_cost + 0.15 * size_cost)

    for group_id, members, counts, _ in ordered:
        choices = []
        for split_index in range(3):
            candidate = allocation.copy()
            candidate[split_index] += counts
            choices.append((cost(candidate), split_index))
        _, split_index = min(choices)
        allocation[split_index] += counts
        assignments[group_id] = split_index

    # Small or heavily grouped datasets may need an explicit nonempty-split pass.
    if len(groups) >= 3:
        for empty in [i for i in range(3) if i not in assignments.values()]:
            group_counts = [list(assignments.values()).count(i) for i in range(3)]
            candidates = []
            for group_id, members, counts, _ in ordered:
                previous = assignments[group_id]
                if group_counts[previous] <= 1:
                    continue
                candidate = allocation.copy()
                candidate[previous] -= counts
                candidate[empty] += counts
                candidates.append((cost(candidate), group_id, previous, counts))
            if candidates:
                _, group_id, previous, counts = min(candidates, key=lambda item: (item[0], item[1]))
                allocation[previous] -= counts
                allocation[empty] += counts
                assignments[group_id] = empty
    for group_id, members in groups.items():
        for record in members:
            record["split"] = SPLITS[assignments[group_id]]


def _csv_bytes(columns: tuple[str, ...], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _freeze_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.stem + ".freeze.json")


def prepare_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    phash_distance: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write manifest.csv, audit.json, excluded.csv and manifest.freeze.json.

    An existing freeze is immutable unless ``overwrite=True``. Repeating the
    same preparation is safe and leaves existing bytes unchanged. No images
    are moved or deleted: excluded.csv is the quarantine register.
    """
    raw = Path(raw_dir).resolve(strict=True)
    output = Path(output_dir).resolve()
    if not raw.is_dir():
        raise ValueError("raw_dir must be a directory.")
    if output == raw or output.is_relative_to(raw):
        raise ValueError("output_dir must be outside raw_dir to avoid auditing generated files.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != 3 or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("Provide three positive finite train/validation/test ratios.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to one.")
    if isinstance(phash_distance, bool) or not isinstance(phash_distance, int) or not 0 <= phash_distance <= 64:
        raise ValueError("phash_distance must be an integer between 0 and 64.")
    settings = {
        "schema_version": 1, "seed": seed, "ratios": list(ratios),
        "phash_distance": phash_distance, "max_image_pixels": MAX_IMAGE_PIXELS,
        "normalization": "EXIF transpose then RGB", "perceptual_hash": "32px DCT 8x8 DC excluded",
    }
    files = sorted((path for path in raw.rglob("*") if path.is_file()), key=lambda path: path.relative_to(raw).as_posix())
    records, excluded = [], []

    def exclude(path: Path, reason: str, detail: str = "", canonical: str = "") -> None:
        excluded.append({"relative_path": path.relative_to(raw).as_posix(), "reason": reason,
                         "detail": detail, "canonical_sample_id": canonical})

    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(raw):
            exclude(path, "unsafe_path", "Symbolic links and paths outside the dataset are not accepted.")
            continue
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            exclude(path, "unsupported_extension")
            continue
        match = _NAME.fullmatch(path.name)
        if match is None:
            exclude(path, "invalid_filename", "Expected age_gender_race_timestamp.jpg or .jpg.chip.jpg.")
            continue
        age = int(match["age"])
        try:
            label = age_to_label(age)
        except ValueError as error:
            exclude(path, "invalid_age", str(error))
            continue
        try:
            metadata = _read_image(path)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            exclude(path, "unreadable_image", f"{type(error).__name__}: {str(error).replace(str(raw), '<raw>')}")
            continue
        records.append({"sample_id": metadata["file_sha256"], "relative_path": path.relative_to(raw).as_posix(),
                        "age": age, "label": label, **metadata})

    valid_before_dedup = len(records)
    pixel_groups: dict[str, list[dict]] = {}
    for record in records:
        pixel_groups.setdefault(record["pixel_sha256"], []).append(record)
    accepted, duplicate_count, conflicts = [], 0, 0
    for members in pixel_groups.values():
        if len({record["label"] for record in members}) > 1:
            conflicts += 1
            for record in members:
                exclude(raw / record["relative_path"], "duplicate_label_conflict",
                        "Identical normalized pixels have conflicting age-group labels.")
            continue
        canonical = members[0]
        accepted.append(canonical)
        for record in members[1:]:
            duplicate_count += 1
            exclude(raw / record["relative_path"], "exact_pixel_duplicate",
                    "Canonical sample retained.", canonical["sample_id"])
    records = sorted(accepted, key=lambda record: record["relative_path"])
    groups = _make_groups(records, phash_distance)
    _assign_splits(groups, seed, ratios)
    excluded.sort(key=lambda record: (record["relative_path"], record["reason"]))
    split_counts = {split: sum(record["split"] == split for record in records) for split in SPLITS}
    class_counts = {str(label): sum(record["label"] == label for record in records) for label in range(3)}
    split_class_counts = {
        split: {str(label): sum(record["split"] == split and record["label"] == label for record in records)
                for label in range(3)} for split in SPLITS
    }
    notices = ["Perceptual groups are not verified subject identities; residual identity overlap is possible."]
    if not records:
        notices.append("No usable images remain; training must not start from this manifest.")
    for split in SPLITS:
        missing = [CLASS_NAMES[label] for label in range(3) if split_class_counts[split][str(label)] == 0]
        if missing:
            notices.append(f"{split} has no examples of: {', '.join(missing)}. Review data/group feasibility before training.")
    audit = {
        "settings": settings, "scanned_files": len(files), "valid_before_dedup": valid_before_dedup,
        "included_samples": len(records), "excluded_files": len(excluded),
        "exact_duplicates_removed": duplicate_count, "conflicting_duplicate_groups": conflicts,
        "perceptual_groups": len(groups), "multi_image_perceptual_groups": sum(len(group) > 1 for group in groups.values()),
        "largest_perceptual_group": max((len(group) for group in groups.values()), default=0),
        "split_counts": split_counts, "class_names": list(CLASS_NAMES), "class_counts": class_counts,
        "split_class_counts": split_class_counts, "identity_split_verified": False, "warnings": notices,
        "source": {
            "name": "UTKFace", "official_page": "https://susanqq.github.io/UTKFace/",
            "terms": "Non-commercial research only; original image owners retain copyright.",
            "annotation_origin": "DEX estimates double-checked by a human annotator, not verified chronological ages.",
            "subject_ids_available": False,
        },
        "quarantine_policy": "Excluded paths are registered, not copied, moved or deleted.",
    }
    payloads = {
        "manifest.csv": _csv_bytes(MANIFEST_COLUMNS, records),
        "excluded.csv": _csv_bytes(("relative_path", "reason", "detail", "canonical_sample_id"), excluded),
        "audit.json": _json_bytes(audit),
    }
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    freeze = {"schema_version": 1, "settings": settings, "checksums": hashes,
              "manifest_sha256": hashes["manifest.csv"],
              "dataset_fingerprint_sha256": hashlib.sha256(_json_bytes(hashes)).hexdigest()}
    manifest = output / "manifest.csv"
    if manifest.exists() and not overwrite:
        if manifest.read_bytes() != payloads["manifest.csv"]:
            raise ValueError("Frozen manifest would change. Use overwrite=True only for an intentional new data freeze.")
        frozen_path = _freeze_path(manifest)
        if not frozen_path.exists():
            raise ValueError("An existing manifest has no freeze metadata; inspect it before using overwrite=True.")
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        if existing != freeze:
            raise ValueError("Dataset audit or freeze settings changed. Use overwrite=True for an intentional new freeze.")
        verify_manifest(manifest, raw, verify_files=False)
        return json.loads((output / "audit.json").read_text(encoding="utf-8"))
    if not manifest.exists() and _freeze_path(manifest).exists() and not overwrite:
        raise ValueError("Freeze metadata exists without a manifest; inspect it before overwriting.")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        _atomic_write(output / name, content)
    _atomic_write(_freeze_path(manifest), _json_bytes(freeze))  # Commit marker written last.
    return audit


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a manifest with integer age, label and dimension fields."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError("Manifest columns do not match the supported schema.")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row at line {line}.")
            try:
                for column in _INTEGER_COLUMNS:
                    row[column] = int(row[column])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid integer at manifest line {line}.") from error
            rows.append(row)
    return rows


def verify_manifest(
    manifest_path: str | Path, raw_dir: str | Path, verify_files: bool = False,
) -> dict[str, Any]:
    """Check schema, safe paths, split/group separation, and frozen checksums.

    ``verify_files=True`` also fully decodes every retained image and compares
    its file/pixel hashes, dimensions and perceptual descriptor to the freeze.
    """
    manifest = Path(manifest_path).resolve(strict=True)
    raw = Path(raw_dir).resolve(strict=True)
    if not raw.is_dir():
        raise ValueError("raw_dir must be a directory.")
    records = load_manifest(manifest)
    seen_ids, seen_paths, seen_file_hashes, seen_pixel_hashes = set(), set(), {}, {}
    group_splits: dict[str, str] = {}
    for row in records:
        path_text = row["relative_path"]
        posix = PurePosixPath(path_text)
        windows = PureWindowsPath(path_text)
        if (not path_text or "\\" in path_text or ":" in path_text or posix.is_absolute()
                or windows.is_absolute() or windows.drive or any(part in {".", ".."} for part in path_text.split("/"))
                or posix.as_posix() != path_text):
            raise ValueError(f"Unsafe relative path in manifest: {path_text!r}")
        image_path = raw.joinpath(*posix.parts)
        if not image_path.resolve().is_relative_to(raw) or image_path.is_symlink():
            raise ValueError(f"Manifest path escapes raw_dir or is a symbolic link: {path_text!r}")
        if not image_path.is_file():
            raise ValueError(f"Manifest image is missing: {path_text!r}")
        if row["label"] not in (0, 1, 2) or age_to_label(row["age"]) != row["label"]:
            raise ValueError(f"Invalid age/class mapping for {path_text!r}")
        if row["width"] <= 0 or row["height"] <= 0 or row["width"] * row["height"] > MAX_IMAGE_PIXELS:
            raise ValueError(f"Invalid dimensions for {path_text!r}")
        if row["split"] not in SPLITS:
            raise ValueError(f"Invalid split for {path_text!r}")
        for column in ("sample_id", "file_sha256", "pixel_sha256", "group_id"):
            if not _HASH.fullmatch(row[column]):
                raise ValueError(f"Invalid {column} for {path_text!r}")
        if row["sample_id"] != row["file_sha256"] or not _PHASH.fullmatch(row["phash"]):
            raise ValueError(f"Invalid sample ID or perceptual hash for {path_text!r}")
        if row["sample_id"] in seen_ids or path_text in seen_paths:
            raise ValueError("Duplicate sample ID or path in manifest.")
        seen_ids.add(row["sample_id"])
        seen_paths.add(path_text)
        for column, hashes in (("file_sha256", seen_file_hashes), ("pixel_sha256", seen_pixel_hashes)):
            if row[column] in hashes:
                raise ValueError(f"Duplicate {column} remains in the canonical manifest.")
            hashes[row[column]] = row["split"]
        previous = group_splits.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("A perceptual group crosses dataset splits.")
        if verify_files:
            metadata = _read_image(image_path)
            for column, actual in metadata.items():
                if row[column] != actual:
                    raise ValueError(f"Frozen {column} changed for {path_text!r}")
            match = _NAME.fullmatch(image_path.name)
            if match is None or int(match["age"]) != row["age"]:
                raise ValueError(f"Filename age does not match manifest: {path_text!r}")
    frozen_path = _freeze_path(manifest)
    if not frozen_path.is_file():
        raise ValueError("Frozen checksum metadata is missing.")
    try:
        freeze = json.loads(frozen_path.read_text(encoding="utf-8"))
        checksums = freeze["checksums"]
        if freeze["schema_version"] != 1 or set(checksums) != {"manifest.csv", "excluded.csv", "audit.json"}:
            raise ValueError("Unsupported frozen checksum schema.")
        for name, expected in checksums.items():
            if not isinstance(expected, str) or not _HASH.fullmatch(expected) or _sha256(manifest.parent / name) != expected:
                raise ValueError(f"Frozen checksum mismatch for {name}.")
        actual_manifest_sha = _sha256(manifest)
        if freeze["manifest_sha256"] != actual_manifest_sha:
            raise ValueError("Frozen manifest checksum mismatch.")
        if freeze["dataset_fingerprint_sha256"] != hashlib.sha256(_json_bytes(checksums)).hexdigest():
            raise ValueError("Frozen dataset fingerprint mismatch.")
        audit = json.loads((manifest.parent / "audit.json").read_text(encoding="utf-8"))
        if freeze["settings"] != audit["settings"] or audit["included_samples"] != len(records):
            raise ValueError("Audit and freeze metadata disagree.")
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        raise ValueError("Frozen metadata is incomplete or unreadable.") from error
    return {"ok": True, "samples": len(records), "groups": len(group_splits),
            "split_counts": {split: sum(row["split"] == split for row in records) for split in SPLITS},
            "files_verified": bool(verify_files), "manifest_sha256": actual_manifest_sha,
            "identity_split_verified": False}

"""Frozen, disk-mapped uint8 face crops for train/validation input throughput.

Test images are never decoded or stored. Cache construction uses Pillow's
EXIF transpose, RGB conversion, and whole-face bicubic resize, matching the
deterministic geometry in ``make_transform``. Incomplete caches cannot be
used for training; verified chunk hashes permit construction to resume.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageOps

from .data import load_manifest, verify_manifest
from .runtime import sha256_file, utc_now, write_json


FORMAT_VERSION = 1
CHUNK_ROWS = 128
GEOMETRY = "EXIF transpose; RGB; whole aligned face resized square with Pillow bicubic; CHW"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_INDEX_FIELDS = {"sample_id", "relative_path", "file_sha256", "label", "split"}


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _chunk_sha(array: np.ndarray, start: int, end: int) -> str:
    return hashlib.sha256(memoryview(array[start:end]).cast("B")).hexdigest()


def _index_rows(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in sorted(_INDEX_FIELDS)} for row in rows]


def _read_metadata(directory: Path) -> dict:
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported tensor cache format.")
        size = metadata["image_size"]
        count = metadata["rows"]
        if type(size) is not int or size < 1 or type(count) is not int or count < 1:
            raise ValueError("Invalid tensor cache image dimensions/count.")
        if metadata["shape"] != [count, 3, size, size] or metadata["dtype"] != "uint8":
            raise ValueError("Invalid tensor cache shape or dtype.")
        if metadata["geometry"] != GEOMETRY or metadata["layout"] != "NCHW":
            raise ValueError("Unsupported tensor cache geometry/layout.")
        for key in ("manifest_sha256", "raw_source_sha256", "sample_ids_sha256", "preprocessing_source_sha256"):
            if not isinstance(metadata[key], str) or not _HASH.fullmatch(metadata[key]):
                raise ValueError(f"Invalid tensor cache {key}.")
        if type(metadata["complete"]) is not bool:
            raise ValueError("Invalid tensor cache completion flag.")
        completed = metadata["completed_rows"]
        if type(completed) is not int or not 0 <= completed <= count:
            raise ValueError("Invalid tensor cache progress.")
        previous = 0
        if not isinstance(metadata["chunks"], list):
            raise ValueError("Invalid tensor cache chunk records.")
        for chunk in metadata["chunks"]:
            if (set(chunk) != {"start", "end", "sha256"}
                    or type(chunk["start"]) is not int or type(chunk["end"]) is not int
                    or chunk["start"] != previous or not previous < chunk["end"] <= count
                    or not isinstance(chunk["sha256"], str) or not _HASH.fullmatch(chunk["sha256"])):
                raise ValueError("Invalid tensor cache chunk progress/hash.")
            previous = chunk["end"]
        if previous != completed:
            raise ValueError("Tensor cache chunk records disagree with progress.")
        if metadata["complete"]:
            if completed != count or not _HASH.fullmatch(metadata.get("images_sha256", "")):
                raise ValueError("Completed tensor cache has no valid array checksum.")
            if type(metadata["array_bytes"]) is not int or metadata["array_bytes"] <= 0:
                raise ValueError("Invalid completed tensor cache file size.")
        return metadata
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Tensor cache metadata is missing, malformed, or incomplete.") from error


def _read_index(directory: Path, metadata: dict) -> list[dict]:
    path = directory / "sample_ids.json"
    try:
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != metadata["sample_ids_sha256"]:
            raise ValueError("Tensor cache sample index checksum changed.")
        records = json.loads(content)
        if not isinstance(records, list) or len(records) != metadata["rows"]:
            raise ValueError("Tensor cache sample index count differs from metadata.")
        identities = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != _INDEX_FIELDS:
                raise ValueError("Malformed tensor cache sample index entry.")
            if (not isinstance(record["sample_id"], str) or not _HASH.fullmatch(record["sample_id"])
                    or record["file_sha256"] != record["sample_id"]
                    or record["sample_id"] in identities
                    or record["split"] not in {"train", "validation"}
                    or type(record["label"]) is not int or record["label"] not in (0, 1, 2)
                    or not isinstance(record["relative_path"], str) or not record["relative_path"]):
                raise ValueError("Invalid/duplicate tensor cache sample identity or split.")
            identities.add(record["sample_id"])
        return records
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Tensor cache sample index is missing or malformed.") from error


def _open_array(directory: Path, metadata: dict, mode: str = "r") -> np.memmap:
    try:
        array = np.load(directory / "images.npy", mmap_mode=mode, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("Tensor cache array is missing or unreadable.") from error
    if (not isinstance(array, np.memmap) or array.shape != tuple(metadata["shape"])
            or array.dtype != np.uint8 or not array.flags.c_contiguous):
        raise ValueError("Tensor cache array shape/dtype/layout differs from metadata.")
    return array


def verify_tensor_cache(cache_dir: str | Path) -> dict:
    """Verify a complete cache, including its entire on-disk array checksum."""
    directory = Path(cache_dir).resolve()
    metadata = _read_metadata(directory)
    if metadata["complete"] is not True:
        raise ValueError("Tensor cache is incomplete; finish construction before training.")
    _read_index(directory, metadata)
    path = directory / "images.npy"
    if path.stat().st_size != metadata["array_bytes"] or sha256_file(path) != metadata["images_sha256"]:
        raise ValueError("Tensor cache array checksum changed.")
    array = _open_array(directory, metadata)
    del array
    return metadata


def _resize_face(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        resized = rgb.resize((image_size, image_size), Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)


def ensure_tensor_cache(
    manifest: str | Path,
    raw_dir: str | Path,
    cache_dir: str | Path,
    image_size: int = 224,
) -> dict:
    """Build/resume a frozen train+validation cache, or verify/reuse it.

    Each source file's bytes are checked against the frozen manifest before
    decoding. Chunk hashes verify already-completed rows when resuming. A
    new freeze, preprocessing implementation, or image size requires a new
    cache directory; existing incompatible data is never silently replaced.
    """
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size must be a positive integer.")
    manifest = Path(manifest).resolve()
    raw = Path(raw_dir).resolve()
    directory = Path(cache_dir).resolve()
    verify_manifest(manifest, raw, verify_files=False)
    rows = [row for row in load_manifest(manifest) if row["split"] in {"train", "validation"}]
    if not rows:
        raise ValueError("Tensor cache requires nonempty train/validation image rows.")
    index = _index_rows(rows)
    index_bytes = _json_bytes(index)
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    expected = {
        "format_version": FORMAT_VERSION,
        "manifest_sha256": sha256_file(manifest),
        "raw_source_sha256": index_sha,
        "sample_ids_sha256": index_sha,
        "preprocessing_source_sha256": sha256_file(Path(__file__)),
        "image_size": image_size,
        "rows": len(rows),
        "shape": [len(rows), 3, image_size, image_size],
        "dtype": "uint8",
        "layout": "NCHW",
        "geometry": GEOMETRY,
    }
    for row in rows:
        path = (raw / row["relative_path"]).resolve()
        if not path.is_relative_to(raw) or sha256_file(path) != row["file_sha256"]:
            raise ValueError(f"Frozen source file hash changed: {row['relative_path']}")
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        metadata = _read_metadata(directory)
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Tensor cache identity/configuration differs; use a new cache directory.")
        if metadata["complete"]:
            return verify_tensor_cache(directory)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Tensor cache directory has files without metadata; inspect them first.")
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            **expected, "complete": False, "completed_rows": 0, "chunks": [],
            "images_sha256": None, "array_bytes": None, "created_utc": utc_now(),
            "build_seconds": 0.0,
        }
        write_json(metadata_path, metadata)
    index_path = directory / "sample_ids.json"
    if index_path.exists():
        if index_path.read_bytes() != index_bytes:
            raise ValueError("Tensor cache sample index changed.")
    elif metadata["completed_rows"] > 0:
        raise ValueError("Partial tensor cache is missing its completed sample index.")
    else:
        temporary = index_path.with_suffix(".json.partial")
        temporary.write_bytes(index_bytes)
        temporary.replace(index_path)
    _read_index(directory, metadata)
    array_path = directory / "images.npy"
    if array_path.exists():
        array = _open_array(directory, metadata, "r+")
    elif metadata["completed_rows"] > 0:
        raise ValueError("Partial tensor cache is missing its completed array.")
    else:
        array = np.lib.format.open_memmap(
            array_path, mode="w+", dtype=np.uint8, shape=tuple(metadata["shape"])
        )
    for chunk in metadata["chunks"]:
        if _chunk_sha(array, chunk["start"], chunk["end"]) != chunk["sha256"]:
            raise ValueError("Partial tensor cache completed chunk checksum changed.")
    started = time.perf_counter()
    previous_seconds = float(metadata["build_seconds"])
    for start in range(metadata["completed_rows"], len(rows), CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, len(rows))
        for position in range(start, end):
            row = rows[position]
            path = raw / row["relative_path"]
            if sha256_file(path) != row["file_sha256"]:
                raise ValueError(f"Frozen source file hash changed before decode: {row['relative_path']}")
            array[position] = _resize_face(path, image_size)
        array.flush()
        metadata["chunks"].append({"start": start, "end": end, "sha256": _chunk_sha(array, start, end)})
        metadata["completed_rows"] = end
        metadata["build_seconds"] = previous_seconds + time.perf_counter() - started
        write_json(metadata_path, metadata)
        print(json.dumps({"event": "tensor_cache_progress", "completed": end, "total": len(rows),
                          "build_seconds": metadata["build_seconds"]}), flush=True)
    array.flush()
    del array
    metadata["images_sha256"] = sha256_file(array_path)
    metadata["array_bytes"] = array_path.stat().st_size
    metadata["complete"] = True
    metadata["completed_utc"] = utc_now()
    metadata["build_seconds"] = previous_seconds + time.perf_counter() - started
    write_json(metadata_path, metadata)
    return metadata


class CachedFaceDataset:
    """Map frozen train/validation rows onto a read-only uint8 disk array.

    Only one image row is copied per access; the complete array remains
    memory-mapped. Returned indices refer to the caller's row order, matching
    ``ManifestDataset`` and validation prediction CSV conventions.
    """

    def __init__(self, rows: list[dict], cache_dir: str | Path) -> None:
        self.rows = list(rows)
        self.cache_dir = Path(cache_dir).resolve()
        self.metadata = verify_tensor_cache(self.cache_dir)
        records = _read_index(self.cache_dir, self.metadata)
        mapping = {record["sample_id"]: (index, record) for index, record in enumerate(records)}
        self._indices = []
        for row in self.rows:
            if row.get("split") not in {"train", "validation"} or row.get("sample_id") not in mapping:
                raise ValueError("Requested row is not in this train/validation tensor cache.")
            index, record = mapping[row["sample_id"]]
            if any(row.get(key) != record[key] for key in _INDEX_FIELDS):
                raise ValueError("Requested row metadata differs from the frozen tensor cache index.")
            self._indices.append(index)
        self._array = _open_array(self.cache_dir, self.metadata)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        pixels = np.array(self._array[self._indices[index]], copy=True, order="C")
        return torch.from_numpy(pixels), int(self.rows[index]["label"]), index

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._array = _open_array(self.cache_dir, self.metadata)

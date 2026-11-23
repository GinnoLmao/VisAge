"""Frozen source, held-out exclusion, pixel equivalence, and cache integrity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import pytest

from visage.data import load_manifest, prepare_dataset
import visage.tensor_cache as cache_module
from visage.tensor_cache import CachedFaceDataset, ensure_tensor_cache, verify_tensor_cache


@pytest.fixture
def cached_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(18):
        age = (12, 35, 70)[index % 3]
        pixels = np.random.default_rng(index + 500).integers(0, 256, (23, 31, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        exif = image.getexif()
        if index == 0:
            exif[274] = 6
        image.save(raw / f"{age}_0_0_{20260101000000000 + index}.jpg.chip.jpg", exif=exif)
    prepared = tmp_path / "prepared"
    prepare_dataset(raw, prepared, seed=19, phash_distance=0)
    manifest = prepared / "manifest.csv"
    return manifest, raw, tmp_path / "cache", load_manifest(manifest)


def test_cache_matches_resize_pixels_excludes_test_and_preserves_row_indices(cached_fixture, monkeypatch):
    torch = pytest.importorskip("torch")
    manifest, raw, directory, rows = cached_fixture
    test_paths = {(raw / row["relative_path"]).resolve() for row in rows if row["split"] == "test"}
    original_open = Image.open

    def forbid_held_out(path, *args, **kwargs):
        assert Path(path).resolve() not in test_paths
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Image, "open", forbid_held_out)
        metadata = ensure_tensor_cache(manifest, raw, directory, image_size=32)
    selected = [row for row in rows if row["split"] in {"train", "validation"}]
    assert metadata["complete"] is True and metadata["rows"] == len(selected)
    assert metadata["shape"] == [len(selected), 3, 32, 32]
    assert np.load(directory / "images.npy", mmap_mode="r").dtype == np.uint8
    dataset = CachedFaceDataset(selected[::-1], directory)
    for index in range(len(dataset)):
        pixels, label, local_index = dataset[index]
        row = selected[::-1][index]
        with Image.open(raw / row["relative_path"]) as source:
            expected = np.asarray(ImageOps.exif_transpose(source).convert("RGB").resize((32, 32), Image.Resampling.BICUBIC)).transpose(2, 0, 1)
        np.testing.assert_array_equal(pixels.numpy(), expected)
        assert pixels.dtype == torch.uint8 and pixels.is_contiguous()
        assert label == row["label"] and local_index == index
    # Each access copies a single row; changing one returned tensor cannot
    # modify cached data or affect a future access.
    before = dataset[0][0].clone()
    dataset[0][0].zero_()
    torch.testing.assert_close(dataset[0][0], before)
    with pytest.raises(ValueError, match="not in this train/validation"):
        CachedFaceDataset([next(row for row in rows if row["split"] == "test")], directory)
    assert ensure_tensor_cache(manifest, raw, directory, image_size=32) == metadata


def test_cache_rejects_frozen_source_tampering(cached_fixture):
    manifest, raw, directory, rows = cached_fixture
    row = next(row for row in rows if row["split"] == "train")
    Image.new("RGB", (31, 23), "red").save(raw / row["relative_path"])
    with pytest.raises(ValueError, match="Frozen source file hash changed"):
        ensure_tensor_cache(manifest, raw, directory, image_size=16)


def test_cache_array_and_sample_index_tampering_are_rejected(cached_fixture):
    manifest, raw, directory, _ = cached_fixture
    ensure_tensor_cache(manifest, raw, directory, image_size=16)
    array = np.load(directory / "images.npy", mmap_mode="r+")
    array[0, 0, 0, 0] ^= np.uint8(1)
    array.flush()
    del array
    with pytest.raises(ValueError, match="checksum changed"):
        verify_tensor_cache(directory)


def test_malformed_metadata_and_index_are_rejected(cached_fixture):
    manifest, raw, directory, _ = cached_fixture
    ensure_tensor_cache(manifest, raw, directory, image_size=16)
    path = directory / "sample_ids.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="index checksum changed"):
        verify_tensor_cache(directory)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["shape"] = [1, 1]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="shape"):
        verify_tensor_cache(directory)


def test_partial_cache_is_unusable_but_verified_prefix_resumes(cached_fixture, monkeypatch):
    manifest, raw, directory, _ = cached_fixture
    original_resize = cache_module._resize_face
    calls = []

    def fail_after_first_chunk(path, size):
        calls.append(path)
        if len(calls) == 5:
            raise InterruptedError("synthetic interrupted build")
        return original_resize(path, size)

    with monkeypatch.context() as context:
        context.setattr(cache_module, "CHUNK_ROWS", 4)
        context.setattr(cache_module, "_resize_face", fail_after_first_chunk)
        with pytest.raises(InterruptedError):
            ensure_tensor_cache(manifest, raw, directory, image_size=16)
    progress = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert progress["complete"] is False and progress["completed_rows"] == 4
    with pytest.raises(ValueError, match="incomplete"):
        CachedFaceDataset([], directory)
    resumed_calls = []

    def record_resize(path, size):
        resumed_calls.append(path)
        return original_resize(path, size)

    with monkeypatch.context() as context:
        context.setattr(cache_module, "_resize_face", record_resize)
        complete = ensure_tensor_cache(manifest, raw, directory, image_size=16)
    assert complete["complete"] and len(resumed_calls) == complete["rows"] - 4
    assert verify_tensor_cache(directory)["images_sha256"] == complete["images_sha256"]


def test_configuration_change_requires_new_cache_directory(cached_fixture):
    manifest, raw, directory, _ = cached_fixture
    ensure_tensor_cache(manifest, raw, directory, image_size=16)
    with pytest.raises(ValueError, match="identity/configuration differs"):
        ensure_tensor_cache(manifest, raw, directory, image_size=32)

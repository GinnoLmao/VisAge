"""Versioned canonical224 then256 Pillow geometry and bounded mapped batches."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageOps
import torch
from .data import load_manifest, verify_manifest
from .runtime import sha256_file, write_json, utc_now

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data/cache/pretrained_faces256'
GEOMETRY = 'EXIF transpose; RGB; whole face Pillow bicubic224; Pillow bicubic256; CHW uint8'


def pixels(image):
    # Prepared API crops are already224; resize to224 is an exact size no-op.
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    first = rgb.resize((224, 224), Image.Resampling.BICUBIC)
    resized = first.resize((256, 256), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1).copy()


def prepare_cache():
    manifest = ROOT / 'data/processed/manifest.csv'
    raw = ROOT / 'data/raw/UTKFace'
    verify_manifest(manifest, raw, verify_files=True)
    rows = [row for row in load_manifest(manifest) if row['split'] in ('train', 'validation')]
    identity = {'geometry': GEOMETRY, 'manifest_sha256': sha256_file(manifest), 'rows': len(rows),
        'source_sha256': sha256_file(Path(__file__)), 'ids': [row['sample_id'] for row in rows]}
    CACHE.mkdir(parents=True, exist_ok=True)
    record_path = CACHE / 'metadata.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {**identity, 'completed': 0, 'chunks': [], 'complete': False}
    if any(record[key] != value for key, value in identity.items()):
        raise ValueError('Cache identity changed; use a distinct cache version.')
    path = CACHE / 'images.npy'
    if record['complete']:
        if sha256_file(path) != record['sha256']:
            raise ValueError('Pretrained cache bytes changed.')
        return record
    array = np.load(path, mmap_mode='r+', allow_pickle=False) if path.exists() else np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(len(rows), 3, 256, 256))
    if array.shape != (len(rows), 3, 256, 256) or array.dtype != np.uint8:
        raise ValueError('Invalid cache shape.')
    for chunk in record['chunks']:
        if hashlib.sha256(memoryview(array[chunk['start']:chunk['end']]).cast('B')).hexdigest() != chunk['sha256']:
            raise ValueError('Partial cache chunk changed.')
    for start in range(record['completed'], len(rows), 128):
        end = min(start + 128, len(rows))
        for i in range(start, end):
            row = rows[i]
            path_source = raw / row['relative_path']
            if sha256_file(path_source) != row['file_sha256']:
                raise ValueError('Raw image differs from data freeze.')
            with Image.open(path_source) as image:
                array[i] = pixels(image)
        array.flush()
        record['chunks'].append({'start': start, 'end': end, 'sha256': hashlib.sha256(memoryview(array[start:end]).cast('B')).hexdigest()})
        record['completed'] = end
        write_json(record_path, record)
        if end % 2048 == 0:
            print(json.dumps({'event': 'cache_progress', 'rows': end, 'total': len(rows)}), flush=True)
    record.update(complete=True, sha256=sha256_file(path), created_utc=utc_now(), bytes=path.stat().st_size)
    write_json(record_path, record)
    write_json(ROOT / 'reports/pretrained/data_cache.json', {k: v for k, v in record.items() if k not in ('ids', 'chunks')})
    return record


class FaceBatches:
    def __init__(self, rows, cache=CACHE):
        self.rows = rows
        self.metadata = json.loads((cache / 'metadata.json').read_text())
        if not self.metadata['complete'] or self.metadata['geometry'] != GEOMETRY or self.metadata['source_sha256'] != sha256_file(Path(__file__)):
            raise ValueError('Unverified or incompatible input cache.')
        lookup = {sample_id: i for i, sample_id in enumerate(self.metadata['ids'])}
        self.indices = np.array([lookup[row['sample_id']] for row in rows], dtype=np.int64)
        self.labels = np.array([row['label'] for row in rows], dtype=np.int64)
        self.array = np.load(cache / 'images.npy', mmap_mode='r', allow_pickle=False)

    def batch(self, positions, device):
        data = torch.from_numpy(np.array(self.array[self.indices[positions]], copy=True)).to(device)
        targets = torch.from_numpy(self.labels[positions].copy()).to(device)
        return data, targets


def class_weights(labels):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    if (counts <= 0).any():
        raise ValueError('Every training subset must contain all three classes.')
    weights = 1 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)

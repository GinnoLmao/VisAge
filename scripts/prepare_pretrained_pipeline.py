"""Prepare every validation attempt through unchanged production YuNet geometry."""
import json
import sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.data import load_manifest
from visage.faces import FaceDetector, FaceError, decode_image
from visage.pretrained_data import pixels
from visage.runtime import configure_caches, sha256_file, write_json, utc_now
configure_caches(ROOT)

CACHE = ROOT / 'data/cache/pretrained_pipeline256'


def main():
    import cv2
    cv2.setNumThreads(2)
    detector = FaceDetector()
    rows = [r for r in load_manifest(ROOT / 'data/processed/manifest.csv') if r['split'] == 'validation']
    CACHE.mkdir(parents=True, exist_ok=True)
    record_path = CACHE / 'metadata.json'
    identity = {'manifest_sha256': sha256_file(ROOT / 'data/processed/manifest.csv'),
        'face_config': asdict(detector.config), 'detector': detector.info(),
        'source_sha256': {p: sha256_file(ROOT / p) for p in ('src/visage/faces.py', 'src/visage/alignment.py',
            'src/visage/pretrained_data.py', 'scripts/prepare_pretrained_pipeline.py', 'configs/face_geometry.json')},
        'sample_ids': [r['sample_id'] for r in rows],
        'scope': 'Full frozen validation-crop diagnostic; redetection/realignment of UTKFace, not real-camera accuracy or independent participant evidence.'}
    record = json.loads(record_path.read_text()) if record_path.exists() else {**identity, 'outcomes': [], 'complete': False}
    if any(record.get(k) != v for k, v in identity.items()):
        raise ValueError('Pipeline cache identity changed.')
    path = CACHE / 'images.npy'
    if record['complete']:
        if sha256_file(path) != record['sha256']:
            raise ValueError('Pipeline cache bytes changed.')
        print('Previously complete pipeline preparation verified.', flush=True)
        return
    array = np.load(path, mmap_mode='r+', allow_pickle=False) if path.exists() else np.lib.format.open_memmap(
        path, mode='w+', dtype=np.uint8, shape=(len(rows), 3, 256, 256))
    for index in range(len(record['outcomes']), len(rows)):
        row = rows[index]
        source = ROOT / 'data/raw/UTKFace' / row['relative_path']
        if sha256_file(source) != row['file_sha256']:
            raise ValueError('Frozen validation source changed.')
        outcome = {'sample_id': row['sample_id'], 'label': row['label'], 'status': 'rejected',
            'error_code': '', 'detected_count': None, 'crop_sha256': '', 'rejection_reasons': []}
        image, prepared = None, None
        try:
            image = decode_image(source.read_bytes(), detector.config)
            candidates = detector.detect(image)
            outcome['detected_count'] = len(candidates)
            prepared = detector.prepare(image, candidates=candidates)
            array[index] = pixels(prepared.crop)
            outcome.update(status='accepted', crop_sha256=prepared.metadata['pixel_sha256'])
        except FaceError as error:
            outcome.update(error_code=error.code, rejection_reasons=error.details.get('reasons', []))
        finally:
            if prepared:
                prepared.crop.close()
            if image:
                image.close()
        record['outcomes'].append(outcome)
        if (index + 1) % 64 == 0:
            array.flush(); write_json(record_path, record)
        if (index + 1) % 512 == 0:
            print(json.dumps({'prepared': index + 1, 'total': len(rows)}), flush=True)
    array.flush()
    record.update(complete=True, sha256=sha256_file(path), prepared_utc=utc_now())
    write_json(record_path, record)
    print(json.dumps({'complete': True, 'attempted': len(rows),
        'accepted': sum(r['status'] == 'accepted' for r in record['outcomes'])}), flush=True)


if __name__ == '__main__':
    main()

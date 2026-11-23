"""Reuse unchanged detector AP only after provenance and exact reload checks."""
import gc
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.faces import FaceConfig, FaceDetector, DETECTOR_SHA256, DETECTOR_REVISION
from visage.runtime import sha256_file, write_json, utc_now


def main():
    import cv2
    cv2.setNumThreads(2)
    directory = ROOT / 'reports/extended_evaluation'
    path = directory / 'detection_ap.json'
    record = json.loads(path.read_text())
    expected = {'detector_sha256': DETECTOR_SHA256,
        'annotation_sha256': sha256_file(ROOT / 'data/detection_benchmark/extracted/wider_face_split/wider_face_val_bbx_gt.txt'),
        'provenance_sha256': sha256_file(ROOT / 'data/detection_benchmark/provenance.json'),
        'predictions_sha256': sha256_file(directory / 'detection_predictions.json')}
    if not record['ok'] or any(record[k] != value for k, value in expected.items()) or record['detector']['revision'] != DETECTOR_REVISION:
        raise ValueError('Detector AP lineage does not match current immutable evidence.')
    saved = json.loads((directory / 'detection_predictions.json').read_text())
    if saved['config'] != record['config'] or any(saved[k] != expected[k] for k in ('detector_sha256', 'annotation_sha256', 'provenance_sha256')):
        raise ValueError('Detector predictions and AP records have different identities.')
    detector = FaceDetector(config=FaceConfig(**record['config']))
    # AP uses a low confidence sweep and no selected-face quality/alignment.
    # Other FaceConfig fields influence only crop preparation and do not enter
    # FaceDetector.detect. Re-run a fixed evenly spaced sample with exact boxes.
    names = list(saved['images'])
    reviews = []
    for position in np.linspace(0, len(names) - 1, 12, dtype=int):
        name = names[position]
        source = ROOT / 'data/detection_benchmark/extracted/WIDER_val/images' / name
        previous = saved['images'][name]
        if sha256_file(source) != previous['file_sha256']:
            raise ValueError('Detector reload fixture changed.')
        with Image.open(source) as image:
            faces = detector.detect(image.convert('RGB'))
        predictions = [{'bbox': list(face.bbox), 'score': face.score} for face in faces]
        if predictions != previous['predictions']:
            raise ValueError('Current detector does not exactly reproduce the AP benchmark predictions.')
        reviews.append({'image': name, 'input_sha256': previous['file_sha256'], 'detections': len(faces), 'exact_match': True})
    target = ROOT / 'reports/pretrained/evaluation'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target / 'detection_ap.json')
    lineage = {'ok': True, 'checked_utc': utc_now(), 'reused_evidence_sha256': sha256_file(path), **expected,
        'detector_revision': DETECTOR_REVISION, 'benchmark_configuration': record['config'],
        'current_detector_configuration': detector.info()['configuration'], 'exact_reload_samples': reviews,
        'note': 'Unchanged YuNet box detector AP reused with original benchmark scope/thresholds. New age-classifier weights do not determine detector AP. Twelve exact reload checks are not a new full AP run.'}
    write_json(target / 'detection_lineage.json', lineage)
    print(json.dumps({'ok': True, 'exact_reload_samples': len(reviews)}), flush=True)


if __name__ == '__main__':
    main()

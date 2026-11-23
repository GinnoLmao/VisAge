"""Reload the current detector and reproduce fixed WIDER validation predictions.

Crop/quality calibration changes do not enter box detection. This bounded check
confirms current inference against saved AP inputs; it does not recompute AP or
add age-test inference.
"""
from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.faces import FaceConfig, FaceDetector
from visage.runtime import sha256_file, utc_now, write_json


def main():
    import cv2
    from PIL import Image
    cv2.setNumThreads(2)
    directory = ROOT / 'reports/extended_evaluation'
    prediction_path = directory / 'detection_predictions.json'
    saved = json.loads(prediction_path.read_text())
    config = FaceConfig(score_threshold=.05, max_faces=5000)
    for key in ('max_detection_side', 'score_threshold', 'nms_threshold', 'max_faces'):
        if getattr(config, key) != saved['config'][key]:
            raise ValueError('The detection operating policy changed.')
    detector = FaceDetector(config=config)
    names = list(saved['images'])
    positions = sorted({round(index * (len(names)-1) / 11) for index in range(12)})
    checks = []
    for position in positions:
        name = names[position]
        path = ROOT / 'data/detection_benchmark/extracted/WIDER_val/images' / name
        expected = saved['images'][name]
        if sha256_file(path) != expected['file_sha256']:
            raise ValueError('A fixed detection-validation input changed.')
        with Image.open(path) as original:
            image = original.convert('RGB')
            try:
                actual = [{'bbox': list(face.bbox), 'score': face.score} for face in detector.detect(image)]
            finally:
                image.close()
        if actual != expected['predictions']:
            raise ValueError('Current detector boxes/scores differ from saved AP predictions.')
        checks.append({'image': name, 'position': position, 'detections': len(actual),
                       'input_sha256': expected['file_sha256'], 'exact_match': True})
    write_json(directory / 'detector_reload_review.json', {
        'ok': True, 'checked_utc': utc_now(), 'samples': len(checks), 'records': checks,
        'detector_sha256': detector.info()['sha256'], 'detection_config': asdict(config),
        'predictions_sha256': sha256_file(prediction_path),
        'source_sha256': {'src/visage/faces.py': sha256_file(ROOT / 'src/visage/faces.py')},
        'opencv_version': metadata.version('opencv-python-headless'),
        'scope': 'Twelve deterministic evenly spaced WIDER validation inputs; exact current CPU box/score reproduction at the AP floor. Crop/quality changes do not enter this detection check.'})
    print(json.dumps({'ok': True, 'samples': len(checks)}), flush=True)


if __name__ == '__main__':
    main()

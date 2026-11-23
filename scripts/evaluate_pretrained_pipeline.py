"""Fresh classifier results on every prepared validation-pipeline attempt."""
import csv
import gc
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.model_registry import METHODS, ARTIFACTS
from visage.pretrained_inference import PretrainedPredictor
from visage.pretrained_evaluation import checked_predictions
from visage.data import load_manifest
from visage.metrics import classification_metrics
from visage.runtime import configure_caches, seed_everything, sha256_file, write_json, utc_now
configure_caches(ROOT)


def main():
    seed_everything(42, threads=4)
    cache = ROOT / 'data/cache/pretrained_pipeline256'
    record = json.loads((cache / 'metadata.json').read_text())
    if not record['complete'] or sha256_file(cache / 'images.npy') != record['sha256']:
        raise ValueError('Complete verified shared-pipeline cache required.')
    for path, digest in record['source_sha256'].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError('Pipeline preparation implementation changed.')
    rows = [r for r in load_manifest(ROOT / 'data/processed/manifest.csv') if r['split'] == 'validation']
    if [r['sample_id'] for r in rows] != record['sample_ids']:
        raise ValueError('Pipeline attempts do not match frozen validation order.')
    outcomes = record['outcomes']
    if len(outcomes) != len(rows) or any(a['sample_id'] != b['sample_id'] or a['label'] != b['label'] for a, b in zip(outcomes, rows, strict=True)):
        raise ValueError('Pipeline outcome IDs/labels changed.')
    accepted = [i for i, row in enumerate(outcomes) if row['status'] == 'accepted']
    rejected = [r for r in outcomes if r['status'] == 'rejected']
    if len(accepted) + len(rejected) != len(rows):
        raise ValueError('Unknown pipeline status.')
    array = np.load(cache / 'images.npy', mmap_mode='r', allow_pickle=False)
    methods, artifacts = {}, {}
    output = ROOT / 'reports/pretrained/pipeline_predictions.csv'
    for method in METHODS:
        predictor = PretrainedPredictor(method, ARTIFACTS / method, 'cuda')
        scores = []
        for start in range(0, len(accepted), 32):
            indices = accepted[start:start + 32]
            scores.extend(predictor.predict_batch(array[indices]).tolist())
        scores = np.array(scores)
        truth = np.array([outcomes[i]['label'] for i in accepted], dtype=np.int64)
        predicted = scores.argmax(1)
        _, baseline = checked_predictions(ARTIFACTS / method / 'validation_predictions.csv', rows)
        for pos, index in enumerate(accepted):
            outcomes[index][method] = int(predicted[pos])
        correct = int((truth == predicted).sum())
        methods[method] = {'accepted_image_metrics': classification_metrics(truth, predicted),
            'end_to_end_correct_count': correct, 'end_to_end_correct_rate': correct / len(rows),
            'original_aligned_metrics_same_accepted_images': classification_metrics(truth, baseline[accepted].argmax(1)),
            'prediction_changes_on_accepted': int((predicted != baseline[accepted].argmax(1)).sum())}
        artifacts[method] = predictor.artifact_sha256
        del predictor; gc.collect(); torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'label', 'status', 'error_code', 'detected_count',
            'crop_sha256', 'rejection_reasons', *METHODS])
        writer.writeheader()
        for outcome in outcomes:
            writer.writerow({**outcome, 'rejection_reasons': json.dumps(outcome['rejection_reasons'])})
    # Independently recount all-attempted and accepted correct values from the CSV.
    with output.open(newline='', encoding='utf-8') as stream:
        saved = list(csv.DictReader(stream))
    for method in METHODS:
        count = sum(r['status'] == 'accepted' and r[method] == r['label'] for r in saved)
        if count != methods[method]['end_to_end_correct_count']:
            raise ValueError('Saved pipeline CSV disagrees with aggregate counts.')
    report = {'ok': True, 'completed_utc': utc_now(), 'scope': record['scope'], 'split': 'validation',
        'manifest_sha256': record['manifest_sha256'], 'face_config': record['face_config'], 'detector': record['detector'],
        'source_sha256': record['source_sha256'], 'artifact_sha256': artifacts,
        'attempted': len(rows), 'accepted': len(accepted), 'rejected': len(rejected), 'coverage': len(accepted) / len(rows),
        'rejection_codes': dict(Counter(r['error_code'] for r in rejected)), 'methods': methods,
        'predictions_sha256': sha256_file(output), 'shared_prepared_pixels_sha256': record['sha256'],
        'independent_local_pilot': False, 'selection_sha256': sha256_file(ROOT / 'reports/pretrained/selection.json')}
    write_json(ROOT / 'reports/pretrained/face_pipeline_validation.json', report)
    print(json.dumps({'ok': True, 'attempted': len(rows), 'accepted': len(accepted)}), flush=True)


if __name__ == '__main__':
    main()

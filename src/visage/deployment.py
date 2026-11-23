"""Explicit, reversible local model-bundle selection; no remote deployment."""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
METHODS = ('classical', 'visagenet', 'convnext_ordinal')


def default_artifacts_dir(root=ROOT):
    root=Path(root)
    bundle=root/'artifacts/serving'
    marker=bundle/'deployment.json'
    if not marker.exists():
        return root/'artifacts'
    record=json.loads(marker.read_text(encoding='utf-8'))
    if record.get('status') not in {'prepared','active'}:
        raise ValueError('Invalid local model-deployment status.')
    if record['status']=='active':
        if set(record.get('methods',{}))!={'classical','visagenet','convnext_ordinal'}:
            raise ValueError('The active bundle must contain all three methods.')
        return bundle
    return root/'artifacts'


def operational_decision(baseline, candidate, baseline_rows, candidate_rows,
                         baseline_benchmark, candidate_benchmark, protocol):
    """Apply the previously recorded gates to identical attempted validation inputs."""
    if not baseline.get('ok') or not candidate.get('ok'):
        raise ValueError('Completed pipeline records are required.')
    if (baseline['manifest_sha256'] != candidate['manifest_sha256']
            or baseline['face_config'] != candidate['face_config']
            or baseline['attempted'] != candidate['attempted']):
        raise ValueError('Pipeline comparisons require identical data and preprocessing.')
    keys = ('sample_id', 'label', 'status', 'error_code', 'crop_sha256')
    if (not baseline_rows or len(baseline_rows) != baseline['attempted']
            or len(candidate_rows) != len(baseline_rows)
            or any(any(a[key] != b[key] for key in keys)
                   for a, b in zip(baseline_rows, candidate_rows, strict=True))):
        raise ValueError('Attempted identities, labels, accepted faces and crop hashes must match.')
    for method in ('classical', 'convnext_ordinal'):
        if baseline['artifact_sha256'][method] != candidate['artifact_sha256'][method]:
            raise ValueError('Only the qualified VisAgeNet slot may change.')
    for benchmark, pipeline in ((baseline_benchmark, baseline), (candidate_benchmark, candidate)):
        if (not benchmark.get('ok') or not benchmark.get('no_concurrent_training')
                or not benchmark['cleanup']['server_process_stopped']
                or not benchmark['cleanup']['port_closed']
                or benchmark['artifact_sha256'] != pipeline['artifact_sha256']):
            raise ValueError('Isolated benchmarks must match their pipeline artifacts and close their servers.')
    if baseline_benchmark['sample_id'] != candidate_benchmark['sample_id']:
        raise ValueError('Latency comparisons require the same input.')
    old = baseline['methods']['visagenet']['accepted_image_metrics']
    new = candidate['methods']['visagenet']['accepted_image_metrics']
    if old is None or new is None:
        raise ValueError('Accepted-face metrics are required.')
    old_ms = baseline_benchmark['method_preprocessing_and_inference']['visagenet']['mean_ms']
    new_ms = candidate_benchmark['method_preprocessing_and_inference']['visagenet']['mean_ms']
    if not all(math.isfinite(v) and v >= 0 for v in (old_ms, new_ms)):
        raise ValueError('Finite nonnegative latency measurements are required.')
    deltas = {
        'macro_f1': new['macro_f1'] - old['macro_f1'],
        'minor_error': new['under18_predicted_older_rate'] - old['under18_predicted_older_rate'],
        'senior_recall': new['per_class'][2]['recall'] - old['per_class'][2]['recall'],
        'baseline_mean_ms': old_ms, 'candidate_mean_ms': new_ms,
        'latency_limit_ms': old_ms * 1.25 + 2.0,
    }
    checks = {
        'pipeline_macro_f1': deltas['macro_f1'] >= -protocol['pipeline_macro_f1_drop_max'],
        'pipeline_minor_error': deltas['minor_error'] <= protocol['pipeline_minor_error_increase_max'],
        'pipeline_senior_recall': deltas['senior_recall'] >= -protocol['pipeline_senior_recall_drop_max'],
        'latency': new_ms <= deltas['latency_limit_ms'],
    }
    return {'qualified': all(checks.values()), 'checks': checks, 'measurements': deltas,
            'scope': 'Frozen validation inputs and one-image sequential local CPU latency; no independent camera-domain claim.'}

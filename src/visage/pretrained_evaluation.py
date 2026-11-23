"""New-model evidence, independent CSV checks and validation-only selection."""
import csv
import gc
import json
import shutil
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import load_manifest
from .evaluation_metrics import multiclass_roc
from .metrics import classification_metrics
from .model_registry import ROOT, METHODS, ARTIFACTS, EVALUATION, REGISTRY
from .pretrained_data import pixels
from .pretrained_inference import PretrainedPredictor
from .pretrained_study import verify_protocol, PROTOCOL
from .pretrained_training import save_predictions
from .research import write_immutable_json
from .runtime import sha256_file, write_json, utc_now

REPORTS = ROOT / 'reports/pretrained'


def freeze_evaluation():
    """Bind evaluation implementation before inspecting any new test predictions."""
    sources = ['src/visage/pretrained_evaluation.py', 'src/visage/pretrained_inference.py',
        'src/visage/evaluation_metrics.py', 'src/visage/metrics.py', 'src/visage/robustness.py',
        'src/visage/faces.py', 'src/visage/alignment.py', 'scripts/evaluate_pretrained_study.py',
        'scripts/evaluate_pretrained_pipeline.py', 'scripts/prepare_pretrained_pipeline.py',
        'scripts/reuse_pretrained_detector_evidence.py']
    record = {'source_sha256': {p: sha256_file(ROOT / p) for p in sources},
        'training_protocol_sha256': sha256_file(PROTOCOL),
        'robustness': 'All3492 validation images; unchanged seven declared clean/dim/bright/blur/jpeg/low_resolution/crop_shift definitions.',
        'validation_selection': 'Three full-data seeds per family; frozen/subset jobs are supporting evidence, not additional tuning candidates.',
        'test': 'One durable resumable FP32 pass per seed42 serving family after validation-only selection; previously observed benchmark.',
        'pipeline': 'All3492 validation attempts through unchanged YuNet five-point geometry; accepted and all-attempted denominators.',
        'detector': 'Reuse original checksum-bound AP; exact12-image reload under its original low-threshold configuration; label benchmark scope.',
        'numeric_checks': 'CSV argmax/score sums, independent confusion counts and rank-sum AUC; CPU/CUDA checks on fixed validation fixtures.'}
    write_immutable_json(REPORTS / 'evaluation_protocol.json', record)
    return record


def independent_auc(truth, scores):
    """Mann–Whitney average ranks; independent of ROC threshold integration."""
    truth, scores = np.asarray(truth, dtype=bool), np.asarray(scores)
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=float)
    first = 0
    while first < len(order):
        end = first + 1
        while end < len(order) and scores[order[end]] == scores[order[first]]:
            end += 1
        ranks[order[first:end]] = (first + 1 + end) / 2
        first = end
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        raise ValueError('Independent AUC requires positive and negative samples.')
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def verified_roc(truth, scores):
    record = multiclass_roc(truth, scores)
    for label in range(3):
        if not np.isclose(independent_auc(np.asarray(truth) == label, scores[:, label]),
            record['per_class'][str(label)]['auc'], atol=1e-12, rtol=0):
            raise ValueError('Independent rank AUC disagrees with ROC integration.')
    record['independent_rank_auc_verified'] = True
    return record


def checked_predictions(path, expected_rows, expected_sha=None):
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise ValueError('Prediction CSV checksum differs from completed run.')
    with Path(path).open(newline='', encoding='utf-8') as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(expected_rows) or len({r['sample_id'] for r in saved}) != len(saved):
        raise ValueError('Prediction sample count/uniqueness mismatch.')
    for a, b in zip(saved, expected_rows, strict=True):
        if a['sample_id'] != b['sample_id'] or int(a['label']) != b['label'] or a['relative_path'] != b['relative_path']:
            raise ValueError('Prediction order/label/path differs from frozen manifest.')
    scores = np.array([[float(r[f'score_{k}']) for k in range(3)] for r in saved])
    truth, predicted = np.array([r['label'] for r in expected_rows]), np.array([int(r['prediction']) for r in saved])
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(1), 1, atol=2e-6, rtol=0):
        raise ValueError('Invalid softmax score vectors.')
    if not np.array_equal(scores.argmax(1), predicted):
        raise ValueError('Prediction labels disagree with softmax argmax.')
    # Independent confusion counts, checked against the shared metric helper.
    counts = [[sum(int(a == i and b == j) for a, b in zip(truth, predicted, strict=True)) for j in range(3)] for i in range(3)]
    metrics = classification_metrics(truth, predicted)
    if counts != metrics['confusion_matrix']:
        raise ValueError('Independent confusion counts disagree.')
    metrics['weighted_f1'] = sum(v['support'] * v['f1'] for v in metrics['per_class']) / len(truth)
    return metrics, scores


def validate_runs():
    protocol = verify_protocol()
    validation = [r for r in load_manifest(ROOT / 'data/processed/manifest.csv') if r['split'] == 'validation']
    records = []
    for config in protocol['runs']:
        directory = ROOT / config['output']
        record = json.loads((directory / 'result.json').read_text())
        if record['status'] != 'complete' or record['protocol_sha256'] != sha256_file(PROTOCOL):
            raise ValueError('All18 declared runs must complete under the frozen protocol.')
        if sha256_file(directory / 'best.pt') != record['artifact_sha256']:
            raise ValueError('Selected research artifact changed.')
        metrics, _ = checked_predictions(directory / 'validation_predictions.csv', validation, record['predictions_sha256'])
        for key in ('macro_f1', 'accuracy', 'confusion_matrix'):
            if metrics[key] != record['best']['validation_metrics'][key]:
                raise ValueError('Reported research metrics disagree with the actual predictions.')
        records.append({**record, 'experiment': config['experiment'], 'verified_metrics': metrics})
    write_json(REPORTS / 'verified_runs.json', {'ok': True, 'runs': records, 'count': len(records),
        'verified_utc': utc_now(), 'no_test_results_used': True})
    return records, validation


def prepare_serving(records):
    for method in METHODS:
        record = next(r for r in records if r['method'] == method and r['seed'] == 42 and r['experiment'] == 'full')
        source = ROOT / f'artifacts/pretrained/runs/{record["run_id"]}'
        target = ARTIFACTS / method
        target.mkdir(parents=True, exist_ok=True)
        for name in ('best.pt', 'validation_predictions.csv', 'history.json', 'config.json'):
            if (target / name).exists() and sha256_file(target / name) != sha256_file(source / name):
                raise ValueError('Serving evidence already exists with different bytes.')
            if not (target / name).exists():
                shutil.copy2(source / name, target / name)
        summary = {**record['best'], 'artifact_sha256': record['artifact_sha256'],
            'predictions_sha256': record['predictions_sha256'], 'completed_run': record['run_id'],
            'epochs': record['epochs'], 'selection_rule': 'Seed42 best validation epoch; family default chosen from three seed means.'}
        write_immutable_json(target / 'training_summary.json', summary)
        write_immutable_json(target / 'validation_metrics.json', record['verified_metrics'])


def cpu_latency(validation):
    path = REPORTS / 'selection_cpu_latency.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['artifact_sha256'] != {m: sha256_file(ARTIFACTS / m / 'best.pt') for m in METHODS}:
            raise ValueError('Latency belongs to other models.')
        return saved
    results = {}
    for method in METHODS:
        predictor = PretrainedPredictor(method, ARTIFACTS / method, 'cpu')
        images = []
        for row in validation[:5]:
            with Image.open(ROOT / 'data/raw/UTKFace' / row['relative_path']) as image:
                images.append(image.convert('RGB').copy())
        for index in range(5):
            predictor.predict(images[index])
        timings = []
        for index in range(30):
            timings.append(predictor.predict(images[index % 5])['inference_and_preprocessing_ms'])
        results[method] = {'median_ms': statistics.median(timings), 'p95_ms': float(np.percentile(timings, 95)),
            'samples': 30, 'fixture_images': 5, 'timings_ms': timings}
        for image in images:
            image.close()
        del predictor; gc.collect()
    record = {'scope': 'Isolated warm CPU batch-one crop inference plus preprocessing, four Torch threads; no concurrent training.',
        'torch_threads': torch.get_num_threads(), 'artifact_sha256': {m: sha256_file(ARTIFACTS / m / 'best.pt') for m in METHODS},
        'methods': results, 'measured_utc': utc_now()}
    write_immutable_json(path, record)
    return record


def choose_default(records, latency):
    means = {}
    for method in METHODS:
        values = [r['verified_metrics'] for r in records if r['method'] == method and r['experiment'] == 'full']
        if len(values) != 3:
            raise ValueError('Three primary seeds per family are required.')
        means[method] = {'macro_f1_mean': float(np.mean([v['macro_f1'] for v in values])),
            'macro_f1_std_sample': float(np.std([v['macro_f1'] for v in values], ddof=1)),
            'under18_older_mean': float(np.mean([v['under18_predicted_older_rate'] for v in values])),
            'senior_recall_mean': float(np.mean([v['per_class'][2]['recall'] for v in values])),
            'cpu_median_ms': latency['methods'][method]['median_ms']}
    highest = max(v['macro_f1_mean'] for v in means.values())
    candidates = [m for m in METHODS if highest - means[m]['macro_f1_mean'] <= .005]
    chosen = min(candidates, key=lambda m: (means[m]['under18_older_mean'], -means[m]['senior_recall_mean'],
        means[m]['cpu_median_ms'], METHODS.index(m)))
    record = {'default_method': chosen, 'validation_means': means, 'near_tied_candidates': candidates,
        'protocol_sha256': sha256_file(PROTOCOL), 'test_results_used': False,
        'primary_evidence': {r['run_id']: {'artifact_sha256': r['artifact_sha256'], 'predictions_sha256': r['predictions_sha256']}
            for r in records if r['experiment'] == 'full'},
        'latency_record_sha256': sha256_file(REPORTS / 'selection_cpu_latency.json')}
    write_immutable_json(REPORTS / 'selection.json', record)
    bundle = {'format_version': 2, 'default_method': chosen, 'selection_sha256': sha256_file(REPORTS / 'selection.json'),
        'protocol_sha256': sha256_file(PROTOCOL), 'methods': {m: {'artifact_sha256': sha256_file(ARTIFACTS / m / 'best.pt'),
            'seed': 42, 'checkpoint': REGISTRY[m].checkpoint} for m in METHODS}}
    write_immutable_json(ARTIFACTS / 'bundle.json', bundle)
    return record


def validation_roc(validation):
    record = {'ok': True, 'task': 'age_group_classification', 'split': 'validation', 'samples': len(validation),
        'manifest_sha256': sha256_file(ROOT / 'data/processed/manifest.csv'), 'methods': {},
        'protocol_sha256': sha256_file(PROTOCOL), 'scope': 'Seed42 serving candidates; full FP32 aligned-crop validation.'}
    for method in METHODS:
        directory = ARTIFACTS / method
        metrics, scores = checked_predictions(directory / 'validation_predictions.csv', validation)
        record['methods'][method] = {'accuracy_percent': metrics['accuracy'] * 100,
            'roc': verified_roc(np.array([r['label'] for r in validation]), scores),
            'score_type': 'Three-class softmax scores; not calibrated certainty',
            'artifact_sha256': sha256_file(directory / 'best.pt'),
            'predictions_sha256': sha256_file(directory / 'validation_predictions.csv')}
    write_json(EVALUATION / 'classification_roc.json', record)
    return record


def score_image_rows(predictor, rows, *, condition='clean', progress_path, scope):
    """Bounded image batches with durable per-batch continuation for interrupted evaluation."""
    from .robustness import perturb
    progress_path = Path(progress_path)
    identity = {'method': predictor.method, 'artifact_sha256': predictor.artifact_sha256,
        'sample_ids': [r['sample_id'] for r in rows], 'condition': condition, 'scope': scope,
        'selection_sha256': sha256_file(REPORTS / 'selection.json')}
    saved = json.loads(progress_path.read_text()) if progress_path.exists() else {**identity, 'scores': []}
    if any(saved.get(k) != v for k, v in identity.items()):
        raise ValueError('Evaluation continuation identity changed.')
    scores = list(saved['scores'])
    for start in range(len(scores), len(rows), 32):
        arrays = []
        for row in rows[start:start + 32]:
            source = ROOT / 'data/raw/UTKFace' / row['relative_path']
            if sha256_file(source) != row['file_sha256']:
                raise ValueError('Evaluation photo differs from frozen dataset.')
            with Image.open(source) as image:
                image.load()
                modified = perturb(image, condition)
                arrays.append(pixels(modified))
                modified.close()
        scores.extend(predictor.predict_batch(arrays).tolist())
        saved['scores'] = scores
        write_json(progress_path, saved)
    return np.array(scores)


def test_once():
    rows = [r for r in load_manifest(ROOT / 'data/processed/manifest.csv') if r['split'] == 'test']
    # Reading only the freeze/decision does not open any test pixels.
    if not (REPORTS / 'selection.json').exists():
        raise ValueError('Freeze default selection before evaluating the existing test benchmark.')
    methods = {}
    for method in METHODS:
        target = REPORTS / 'test' / method
        target.mkdir(parents=True, exist_ok=True)
        report = target / 'metrics.json'
        if report.exists():
            record = json.loads(report.read_text())
            metrics, _ = checked_predictions(target / 'predictions.csv', rows, record['predictions_sha256'])
            if metrics != record['metrics'] or record['artifact_sha256'] != sha256_file(ARTIFACTS / method / 'best.pt'):
                raise ValueError('Existing test evidence changed.')
        else:
            predictor = PretrainedPredictor(method, ARTIFACTS / method, 'cuda')
            scores = score_image_rows(predictor, rows, progress_path=target / 'progress.json',
                scope='One post-selection pass on existing previously observed test benchmark')
            save_predictions(target / 'predictions.csv', rows, scores)
            metrics, checked = checked_predictions(target / 'predictions.csv', rows)
            record = {'metrics': metrics, 'roc': verified_roc(np.array([r['label'] for r in rows]), checked),
                'artifact_sha256': predictor.artifact_sha256,
                'predictions_sha256': sha256_file(target / 'predictions.csv'),
                'selection_sha256': sha256_file(REPORTS / 'selection.json'),
                'note': 'Previously observed existing benchmark. No new tuning or default selection uses these test results.'}
            write_json(report, record)
            del predictor; gc.collect(); torch.cuda.empty_cache()
        methods[method] = record
        print(json.dumps({'event': 'test_complete', 'method': method, 'accuracy': record['metrics']['accuracy']}), flush=True)
    write_json(REPORTS / 'test_summary.json', {'ok': True, 'samples': len(rows), 'methods': methods,
        'independent_confirmation': False, 'selection_sha256': sha256_file(REPORTS / 'selection.json')})


def robustness(validation):
    from .robustness import CONDITIONS
    methods = {}
    for method in METHODS:
        predictor = PretrainedPredictor(method, ARTIFACTS / method, 'cuda')
        conditions = {}
        for condition, definition in CONDITIONS.items():
            target = REPORTS / 'robustness' / method
            target.mkdir(parents=True, exist_ok=True)
            scores = score_image_rows(predictor, validation, condition=condition,
                progress_path=target / f'{condition}_progress.json', scope='Full aligned validation perturbations')
            save_predictions(target / f'{condition}.csv', validation, scores)
            metrics, checked = checked_predictions(target / f'{condition}.csv', validation)
            if condition == 'clean':
                _, original = checked_predictions(ARTIFACTS / method / 'validation_predictions.csv', validation)
                if not np.array_equal(checked.argmax(1), original.argmax(1)) or not np.allclose(checked, original, atol=2e-5, rtol=2e-4):
                    raise ValueError('Clean inference does not reproduce the selected validation predictions.')
            conditions[condition] = {'definition': definition, 'metrics': metrics,
                'predictions_sha256': sha256_file(target / f'{condition}.csv')}
            write_json(target / 'summary.json', conditions)
            print(json.dumps({'event': 'robustness', 'method': method, 'condition': condition,
                'macro_f1': metrics['macro_f1']}), flush=True)
        methods[method] = {'artifact_sha256': predictor.artifact_sha256, 'conditions': conditions}
        del predictor; gc.collect(); torch.cuda.empty_cache()
    write_json(REPORTS / 'robustness.json', {'ok': True, 'samples': len(validation), 'split': 'validation',
        'scope': 'Bounded perturbations of aligned crops; not real-camera or demographic fairness evidence.', 'methods': methods})

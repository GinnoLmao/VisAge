"""User-approved two-seed amendment and inference-only model finalization.

The original training protocol and completed run evidence remain immutable.
This module never imports or calls a training/optimizer entry point.
"""
import gc
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .data import load_manifest
from .model_registry import ROOT, METHODS, REGISTRY, ARTIFACTS
from .pretrained_data import FaceBatches, class_weights, GEOMETRY
from .pretrained_evaluation import checked_predictions, verified_roc, cpu_latency
from .pretrained_models import PretrainedAgeModel
from .pretrained_study import verify_protocol, PROTOCOL
from .pretrained_training import evaluate
from .research import write_immutable_json, verify_protected_files
from .runtime import sha256_file, write_json, utc_now

REPORTS = ROOT / 'reports/pretrained'
AMENDMENT = ROOT / 'configs/pretrained/two_seed_amendment.json'
SEEDS = (42, 43)


def freeze_amendment():
    if AMENDMENT.exists():
        record = json.loads(AMENDMENT.read_text())
        if record['training_protocol_sha256'] != sha256_file(PROTOCOL):
            raise ValueError('Original training protocol changed.')
        verify_protocol()
        verify_protected_files(ROOT, record['completed_evidence_sha256'])
        return record
    protocol = verify_protocol()
    included = [r for r in protocol['runs'] if r['experiment'] == 'full' and r['seed'] in SEEDS]
    if len(included) != 6 or {(r['method'], r['seed']) for r in included} != {(m, s) for m in METHODS for s in SEEDS}:
        raise ValueError('The amendment requires exactly two shared seeds for every family.')
    hashes, excluded, deferred = {}, [], []
    for config in protocol['runs']:
        directory = ROOT / config['output']
        if config in included:
            result = json.loads((directory / 'result.json').read_text())
            if result['status'] != 'complete':
                raise ValueError('Only completed runs can be included.')
            for name in ('best.pt', 'last.pt', 'config.json', 'history.json', 'best.json', 'validation_predictions.csv', 'result.json'):
                relative = f'{config["output"]}/{name}'
                hashes[relative] = sha256_file(ROOT / relative)
        elif config['experiment'] == 'full':
            history = json.loads((directory / 'history.json').read_text()) if (directory / 'history.json').exists() else []
            excluded.append({'run_id': config['run_id'], 'reason': 'Seed44 excluded by the user-approved two-seed scope',
                'status': 'unfinished' if history else 'not_started', 'last_completed_epoch': len(history),
                'retained_files_sha256': {p.name: sha256_file(p) for p in directory.glob('*')
                    if p.is_file() and p.name in ('best.pt', 'last.pt', 'history.json', 'config.json', 'best.json', 'validation_predictions.csv')}})
        else:
            deferred.append({'run_id': config['run_id'], 'experiment': config['experiment'],
                'status': 'deferred_not_run', 'reason': 'User stopped training; no supporting-study results are claimed.'})
    record = {'format_version': 1, 'approved_scope': 'User explicitly approved two seeds per model and requested steps1–5 without restarting training.',
        'recorded_utc': utc_now(), 'training_protocol_sha256': sha256_file(PROTOCOL), 'seeds': list(SEEDS),
        'included_runs': included, 'completed_runs': 6, 'completed_evidence_sha256': hashes,
        'excluded_runs': excluded, 'deferred_runs': deferred, 'training_restart_authorized': False,
        'selection': {**protocol['default_selection'], 'primary': 'Mean validation macro F1 across seeds42 and43 only', 'serving_seed': 42},
        'test_policy': protocol['test_policy'],
        'limitation': 'Two seeds provide limited replication; report both scores, mean and sample standard deviation. This amendment was made after validation results had been observed, at the user request, before any new-model test evaluation.',
        'unchanged': ['original protocol and run bytes', 'data membership', 'weights and preprocessing',
            'within-family seed42 serving choice', 'near-tie threshold0.005 and boundary/latency tie-break order']}
    write_immutable_json(AMENDMENT, record)
    write_json(REPORTS / 'active_study.json', {'status': 'six_completed_runs_frozen', 'completed': 6,
        'seeds': list(SEEDS), 'amendment_sha256': sha256_file(AMENDMENT), 'training_stopped': True,
        'excluded_seed44_runs': 3, 'deferred_supporting_runs': 9})
    return record


def freeze_evaluation_amendment():
    old = REPORTS / 'evaluation_protocol.json'
    original = json.loads(old.read_text())
    verify_protected_files(ROOT, original['source_sha256'])
    sources = [*original['source_sha256'], 'src/visage/pretrained_two_seed.py',
        'scripts/finalize_pretrained_two_seed.py', 'tests/test_pretrained_two_seed.py']
    record = {'original_evaluation_protocol_sha256': sha256_file(old),
        'training_protocol_sha256': sha256_file(PROTOCOL), 'amendment_sha256': sha256_file(AMENDMENT),
        'source_sha256': {p: sha256_file(ROOT / p) for p in sources}, 'seeds': list(SEEDS),
        'selection': 'Exactly six completed full-data runs; shared seed42 serving checkpoints; original validation-only tie rules.',
        'test_policy': original['test'], 'robustness': original['robustness'], 'pipeline': original['pipeline'],
        'detector': original['detector'], 'numeric_checks': original['numeric_checks'],
        'training_steps_performed': 0, 'reason': 'User-approved reduction from three seeds to two; incomplete/deferred jobs cannot gate or contaminate evaluation.'}
    write_immutable_json(REPORTS / 'evaluation_protocol_two_seed.json', record)
    return record


def verify_six_runs(*, reload_check=True):
    amendment = freeze_amendment()
    rows = load_manifest(ROOT / 'data/processed/manifest.csv')
    validation = [r for r in rows if r['split'] == 'validation']
    train_labels = [r['label'] for r in rows if r['split'] == 'train']
    expected_weights = class_weights(train_labels).tolist()
    records, reloaded = [], []
    for config in amendment['included_runs']:
        directory = ROOT / config['output']
        result = json.loads((directory / 'result.json').read_text())
        history = json.loads((directory / 'history.json').read_text())
        saved_config = json.loads((directory / 'config.json').read_text())
        best = json.loads((directory / 'best.json').read_text())
        if saved_config != config or result['protocol_sha256'] != sha256_file(PROTOCOL):
            raise ValueError('Run configuration/protocol mismatch.')
        if result['best'] != best or result['epochs'] != len(history) or [r['epoch'] for r in history] != list(range(1, len(history) + 1)):
            raise ValueError('Run history or saved best metadata mismatch.')
        selected = max(history, key=lambda v: v['validation']['macro_f1'])
        if selected['epoch'] != best['epoch'] or selected['validation'] != best['validation_metrics']:
            raise ValueError('Selected artifact is not the earliest highest validation epoch.')
        if any(r['stage'] != ('head_only' if r['epoch'] <= config['warmup_epochs'] else 'fine_tune') for r in history):
            raise ValueError('Recorded freeze/unfreeze transition disagrees with the protocol.')
        if best['geometry'] != GEOMETRY or not np.allclose(best['class_weights'], expected_weights, atol=0, rtol=0):
            raise ValueError('Preprocessing/class weights differ from the frozen training policy.')
        if sha256_file(directory / 'best.pt') != result['artifact_sha256']:
            raise ValueError('Completed selected artifact changed.')
        metrics, scores = checked_predictions(directory / 'validation_predictions.csv', validation, result['predictions_sha256'])
        for key, value in best['validation_metrics'].items():
            if metrics[key] != value:
                raise ValueError(f'Saved predictions disagree with reported metric: {key}')
        roc = verified_roc(np.array([r['label'] for r in validation]), scores)
        saved = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        if json.loads(json.dumps(saved['metadata'])) != best:
            raise ValueError('Checkpoint metadata differs from the selected run.')
        model = PretrainedAgeModel(config['method'])
        model.load_state_dict(saved['state_dict'], strict=True)
        if sum(p.numel() for p in model.parameters()) != best['parameters']:
            raise ValueError('Checkpoint parameter count mismatch.')
        del saved
        reload_record = {'run_id': config['run_id'], 'selected_epoch': best['epoch'], 'selected_stage': selected['stage'],
            'strict_full_checkpoint_reload': True, 'artifact_sha256': result['artifact_sha256']}
        if reload_check:
            model.to('cuda').to(memory_format=torch.channels_last)
            actual_metrics, actual_scores = evaluate(model, FaceBatches(validation), REGISTRY[config['method']],
                device=torch.device('cuda'), batch_size=32, memory_format='channels_last')
            if not np.array_equal(actual_scores.argmax(1), scores.argmax(1)) or not np.allclose(actual_scores, scores, atol=2e-5, rtol=2e-4):
                raise ValueError('Full validation predictions changed after reloading the selected checkpoint.')
            if actual_metrics != best['validation_metrics']:
                raise ValueError('Reloaded full-validation metrics differ.')
            reload_record.update(all_validation_labels_reproduced=True,
                max_absolute_score_difference=float(np.abs(actual_scores - scores).max()), samples=len(validation))
        records.append({**result, 'experiment': 'full', 'verified_metrics': metrics, 'roc': roc,
            'selected_stage': selected['stage'], 'amendment_sha256': sha256_file(AMENDMENT)})
        reloaded.append(reload_record)
        write_json(REPORTS / 'reload_verification_progress.json', reloaded)
        print(json.dumps({'event': 'artifact_verified', **reload_record}), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    write_json(REPORTS / 'verified_runs.json', {'ok': True, 'runs': records, 'count': len(records),
        'seeds': list(SEEDS), 'verified_utc': utc_now(), 'no_test_results_used': True,
        'amendment_sha256': sha256_file(AMENDMENT), 'deferred_supporting_runs': 9, 'excluded_seed44_runs': 3})
    write_json(REPORTS / 'reload_verification.json', {'ok': True, 'runs': reloaded, 'inference_only': True,
        'validation_samples_per_run': len(validation), 'atol': 2e-5, 'rtol': 2e-4})
    return records, validation


def prepare_two_seed_serving(records):
    for method in METHODS:
        record = next(r for r in records if r['method'] == method and r['seed'] == 42)
        source = ROOT / f'artifacts/pretrained/runs/{record["run_id"]}'
        target = ARTIFACTS / method
        target.mkdir(parents=True, exist_ok=True)
        for name in ('best.pt', 'validation_predictions.csv', 'history.json', 'config.json'):
            if (target / name).exists() and sha256_file(target / name) != sha256_file(source / name):
                raise ValueError('Serving evidence exists with different bytes.')
            if not (target / name).exists():
                shutil.copy2(source / name, target / name)
        summary = {**record['best'], 'artifact_sha256': record['artifact_sha256'],
            'predictions_sha256': record['predictions_sha256'], 'completed_run': record['run_id'],
            'epochs': record['epochs'], 'selected_stage': record['selected_stage'],
            'amendment_sha256': sha256_file(AMENDMENT),
            'selection_rule': 'Seed42 best validation epoch; default family chosen from the shared seeds42/43 means.'}
        write_immutable_json(target / 'training_summary.json', summary)
        write_immutable_json(target / 'validation_metrics.json', record['verified_metrics'])


def rank_families(records, latency):
    means = {}
    for method in METHODS:
        values = [r for r in records if r['method'] == method and r['experiment'] == 'full']
        if len(values) != 2 or {r['seed'] for r in values} != set(SEEDS):
            raise ValueError('Exactly seeds42 and43 are required per family; duplicates/seed44 are excluded.')
        metrics = [r['verified_metrics'] for r in values]
        means[method] = {'macro_f1_mean': float(np.mean([v['macro_f1'] for v in metrics])),
            'macro_f1_std_sample': float(np.std([v['macro_f1'] for v in metrics], ddof=1)),
            'accuracy_mean': float(np.mean([v['accuracy'] for v in metrics])),
            'under18_older_mean': float(np.mean([v['under18_predicted_older_rate'] for v in metrics])),
            'senior_recall_mean': float(np.mean([v['per_class'][2]['recall'] for v in metrics])),
            'cpu_median_ms': latency['methods'][method]['median_ms']}
    highest = max(v['macro_f1_mean'] for v in means.values())
    candidates = [m for m in METHODS if highest - means[m]['macro_f1_mean'] <= .005]
    chosen = min(candidates, key=lambda m: (means[m]['under18_older_mean'], -means[m]['senior_recall_mean'],
        means[m]['cpu_median_ms'], METHODS.index(m)))
    return chosen, means, candidates


def select_two_seed_default(records, latency):
    chosen, means, candidates = rank_families(records, latency)
    decision = {'default_method': chosen, 'seeds': list(SEEDS), 'validation_means': means,
        'near_tied_candidates': candidates, 'protocol_sha256': sha256_file(PROTOCOL),
        'amendment_sha256': sha256_file(AMENDMENT), 'test_results_used': False,
        'primary_evidence': {r['run_id']: {'artifact_sha256': r['artifact_sha256'], 'predictions_sha256': r['predictions_sha256']}
            for r in records}, 'latency_record_sha256': sha256_file(REPORTS / 'selection_cpu_latency.json'),
        'rule': 'Mean validation macroF1; within0.005 use lower mean under18-to-older error, higher senior recall, lower CPU median, registry order.',
        'limitation': 'Two seeds; limited replication; no independent-confirmation claim.'}
    write_immutable_json(REPORTS / 'selection.json', decision)
    bundle = {'format_version': 2, 'default_method': chosen, 'selection_sha256': sha256_file(REPORTS / 'selection.json'),
        'protocol_sha256': sha256_file(PROTOCOL), 'amendment_sha256': sha256_file(AMENDMENT),
        'seeds': list(SEEDS), 'methods': {m: {'artifact_sha256': sha256_file(ARTIFACTS / m / 'best.pt'),
            'seed': 42, 'selected_stage': next(r['selected_stage'] for r in records if r['method'] == m and r['seed'] == 42),
            'checkpoint': REGISTRY[m].checkpoint} for m in METHODS}}
    write_immutable_json(ARTIFACTS / 'bundle.json', bundle)
    return decision

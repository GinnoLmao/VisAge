"""Immutable study setup and sequential job execution."""
import json
from pathlib import Path
from .data import load_manifest
from .model_registry import ROOT, METHODS, REGISTRY
from .research import nested_training_subsets, verify_protected_files, write_immutable_json
from .runtime import sha256_file, utc_now, write_json, environment_info

PROTOCOL = ROOT / 'configs/pretrained/protocol.json'
REPORTS = ROOT / 'reports/pretrained'
SOURCES = ['src/visage/model_registry.py', 'src/visage/pretrained_models.py',
    'src/visage/pretrained_data.py', 'src/visage/pretrained_training.py',
    'src/visage/pretrained_study.py', 'src/visage/augmentation.py', 'src/visage/metrics.py',
    'src/visage/runtime.py', 'src/visage/research.py', 'scripts/run_pretrained_study.py',
    'scripts/pilot_pretrained_models.py', 'scripts/pilot_pretrained_layout.py',
    'scripts/download_pretrained_models.py', 'scripts/prepare_pretrained_data.py']


def verify_originals():
    old = json.loads((ROOT / 'reports/research/protocol.json').read_text())
    verify_protected_files(ROOT, old['protected_original_files'])
    verify_protected_files(ROOT, old['implementation']['source_files'])


def verify_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    verify_protected_files(ROOT, protocol['source_files'])
    verify_protected_files(ROOT, protocol['input_files'])
    verify_originals()
    return protocol


def freeze_protocol():
    if PROTOCOL.exists():
        return verify_protocol()
    verify_originals()
    pilot = json.loads((REPORTS / 'gpu_pilot.json').read_text())
    layout = json.loads((REPORTS / 'layout_pilot.json').read_text())
    if not pilot['ok'] or pilot['microbatch'] != 32 or {r['method'] for r in layout} != set(METHODS):
        raise ValueError('The common batch/layout pilot must pass before freezing.')
    if any(r['peak_reserved_mib'] > 5300 for r in layout):
        raise ValueError('Layout exceeds the predeclared GPU budget.')
    manifest = ROOT / 'data/processed/manifest.csv'
    rows = load_manifest(manifest)
    subsets = nested_training_subsets(rows, sha256_file(manifest))
    for fraction, record in subsets.items():
        write_immutable_json(ROOT / f'configs/pretrained/subset{int(fraction * 100)}.json', record)
    common = {'device': 'cuda', 'microbatch': 32, 'effective_batch': 32,
        'memory_format': 'channels_last', 'amp': 'bfloat16', 'validation_precision': 'float32',
        'max_epochs': 24, 'min_epochs': 8, 'patience': 5, 'minimum_improvement': 0.0,
        'warmup_epochs': 2, 'backbone_lr': .0001, 'head_lr': .0005,
        'weight_decay': .01, 'gradient_clip': 5.0}
    runs = []
    settings = [('full', seed, False, None) for seed in (42, 43, 44)]
    settings += [('frozen', 42, True, None)]
    settings += [(f'subset{n}', 42, False, f'configs/pretrained/subset{n}.json') for n in (25, 50)]
    for label, seed, frozen, subset in settings:
        for method in METHODS:
            run_id = f'{method}_{label}_seed{seed}'
            runs.append({**common, 'run_id': run_id, 'method': method, 'seed': seed,
                'frozen_backbone': frozen, 'subset': subset, 'experiment': label,
                'output': f'artifacts/pretrained/runs/{run_id}'})
    inputs = ['data/processed/manifest.csv', 'data/cache/pretrained_faces256/images.npy',
        'data/cache/pretrained_faces256/metadata.json', 'reports/pretrained/gpu_pilot.json',
        'reports/pretrained/layout_pilot.json', 'configs/pretrained/subset25.json',
        'configs/pretrained/subset50.json', 'reports/pretrained_migration/plan.md']
    for method in METHODS:
        inputs += [f'.cache/pretrained_weights/{method}/{name}' for name in
            ('model.safetensors', 'config.json', 'README.md', 'provenance.json')]
    record = {'format_version': 2, 'frozen_utc': utc_now(), 'models': {k: v.to_dict() for k, v in REGISTRY.items()},
        'source_files': {p: sha256_file(ROOT / p) for p in SOURCES},
        'input_files': {p: sha256_file(ROOT / p) for p in inputs}, 'runs': runs,
        'loss': 'three-logit class-weighted cross entropy; normalized inverse-square-root training counts; no resampling',
        'augmentation': 'unchanged visage_tensor_batch_v1; independent per-image device RNG',
        'stop_rule': 'Strictly higher validation macro F1 saves best; equal keeps earlier; patience5; minimum8; maximum24',
        'scheduler': 'cosine per epoch: .5*(1+cos(pi*(epoch-1)/23)); no separate scaler for BF16',
        'default_selection': {'primary': 'mean validation macro F1 across seeds42,43,44',
            'near_tie': 'highest mean minus candidate mean <=0.005 inclusive; full precision, no rounding',
            'ties': ['lowest mean under18_predicted_older_rate', 'highest mean senior recall',
                'lowest isolated CPU batch-one median latency', 'registry order for exact final tie'],
            'boundary_tie_tolerance': 0.0, 'serving_seed': 42},
        'test_policy': 'Existing previously observed benchmark; one final evaluation after selection freeze; no test-guided tuning',
        'environment': environment_info()}
    write_immutable_json(PROTOCOL, record)
    write_json(REPORTS / 'protocol_identity.json', {'path': str(PROTOCOL.relative_to(ROOT)),
        'sha256': sha256_file(PROTOCOL), 'runs': len(runs), 'frozen_utc': record['frozen_utc']})
    return record

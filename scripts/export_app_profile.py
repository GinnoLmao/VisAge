"""Publish checksum-bound recorded serving cost and seed-42 training duration."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import sha256_file, write_json, utc_now
from visage.model_registry import METHODS, ARTIFACTS


def main():
    reports = ROOT / 'reports/pretrained'
    runs = json.loads((reports / 'verified_runs.json').read_text())
    benchmark = json.loads((reports / 'benchmark/backend_benchmark.json').read_text())
    bundle = json.loads((ARTIFACTS / 'bundle.json').read_text())
    if not runs['ok'] or runs['count'] != 6 or not benchmark['ok'] or not benchmark['no_concurrent_training']:
        raise ValueError('Verified two-seed runs and isolated timing evidence are required.')
    methods = {}
    for method in METHODS:
        run = next(r for r in runs['runs'] if r['method'] == method and r['seed'] == 42)
        digest = sha256_file(ARTIFACTS / method / 'best.pt')
        if run['artifact_sha256'] != digest or benchmark['artifact_sha256'][method] != digest or bundle['methods'][method]['artifact_sha256'] != digest:
            raise ValueError('Profile evidence must refer to the currently served checkpoint.')
        methods[method] = {'artifact_sha256': digest, 'parameters': run['best']['parameters'],
            'training_seconds': run['elapsed_seconds'], 'training_run_id': run['run_id'], 'training_epochs': run['epochs'],
            'runtime_median_ms': benchmark['method_preprocessing_and_inference'][method]['median_ms'],
            'runtime_p95_ms': benchmark['method_preprocessing_and_inference'][method]['p95_ms']}
    write_json(reports / 'evaluation/application_profile.json', {'ok': True, 'created_utc': utc_now(),
        'manifest_sha256': runs['runs'][0]['best']['manifest_sha256'], 'methods': methods,
        'runtime_scope': 'Recorded warm CPU median per classifier, including its resizing/normalization and inference; excludes shared decode/detection/alignment and HTTP overhead. Thirty Compare All requests after five warm-ups on one validation image.',
        'training_scope': 'Elapsed wall time for each complete seed-42 run, including warm-up and subsequent fine-tuning through stopping; not just the selected epoch or the combined two-seed total.',
        'source_sha256': {str(p.relative_to(ROOT)).replace(chr(92), '/'): sha256_file(p) for p in
            (reports / 'verified_runs.json', reports / 'benchmark/backend_benchmark.json')}})
    print('Exported recorded runtime, parameters and training durations for three serving checkpoints.')


if __name__ == '__main__':
    main()

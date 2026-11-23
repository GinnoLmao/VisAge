"""Prepare, assess and activate a separately verified local model bundle.

Original artifacts are never overwritten. This changes local defaults only;
there is no network publishing, public server or automatic training selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.deployment import METHODS, operational_decision, default_artifacts_dir
from visage.runtime import sha256_file, utc_now, write_json

BUNDLE = ROOT / 'artifacts/serving'
MARKER = BUNDLE / 'deployment.json'
ASSESSMENT = ROOT / 'reports/deployment/assessment.json'
BASELINE = ROOT / 'reports/diagnostics/deployment_baseline'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def eligible():
    study = read(ROOT / 'reports/research/study_summary.json')
    review = read(ROOT / 'reports/research/independent_review.json')
    if (not review.get('ok') or not review.get('study_complete')
            or len(study['runs']) != 15 or not study['promotion']['promote']
            or review['promotion']['promote'] != study['promotion']['promote']):
        raise ValueError('Completed, independently qualified paired-seed research is required.')
    if sha256_file(ROOT / 'reports/research/study_summary.json') != review['evidence_checksums']['reports/research/study_summary.json']:
        raise ValueError('The research summary changed after review.')
    return study['proposed_champion']


def bundle_hashes():
    return {method: sha256_file(BUNDLE / method / ('svm.npz' if method == 'classical' else 'best.pt'))
            for method in METHODS}


def verify_prepared():
    marker = read(MARKER)
    if marker['candidate'] != eligible() or marker['status'] not in {'prepared', 'active'}:
        raise ValueError('Prepared bundle is not the qualified candidate.')
    for relative, digest in marker['files'].items():
        path = (ROOT / relative).resolve(strict=True)
        if not path.is_relative_to(BUNDLE.resolve()) or sha256_file(path) != digest:
            raise ValueError('A prepared model-bundle file changed.')
    if marker['methods'] != bundle_hashes():
        raise ValueError('Prepared model weights changed.')
    return marker


def prepare():
    candidate = eligible()
    if BUNDLE.exists():
        raise FileExistsError('Preserve the existing bundle; preparation requires a fresh directory.')
    sources = {method: ROOT / 'artifacts' / method for method in METHODS}
    sources['visagenet'] = ROOT / 'artifacts/research' / candidate
    for method, source in sources.items():
        target = BUNDLE / method
        target.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file() and path.name != 'last.pt':
                shutil.copy2(path, target / path.name)
    files = {path.relative_to(ROOT).as_posix(): sha256_file(path)
             for path in sorted(BUNDLE.rglob('*')) if path.is_file()}
    write_json(MARKER, {'status': 'prepared', 'prepared_utc': utc_now(), 'candidate': candidate,
        'serving_seed': 42, 'methods': bundle_hashes(), 'files': files,
        'scope': 'Separate local inference bundle; original models preserved; activation pending operational checks.'})
    print('Prepared a separate bundle. Original defaults remain active.', flush=True)


def assess():
    marker = verify_prepared()
    protocol = read(ROOT / 'configs/deployment_protocol.json')
    paths = {'baseline_pipeline': BASELINE / 'face_pipeline_validation.json',
        'candidate_pipeline': ROOT / 'reports/face_pipeline_validation.json',
        'baseline_benchmark': BASELINE / 'backend_benchmark.json',
        'candidate_benchmark': ROOT / 'reports/backend_benchmark.json'}
    records = {key: read(path) for key, path in paths.items()}
    rows = []
    for name in ('baseline_pipeline', 'candidate_pipeline'):
        path = ROOT / records[name]['predictions_csv']
        if sha256_file(path) != records[name]['predictions_sha256']:
            raise ValueError('Pipeline predictions changed.')
        paths[name + '_predictions'] = path
        with path.open(newline='', encoding='utf-8') as stream:
            rows.append(list(csv.DictReader(stream)))
    if records['candidate_pipeline']['artifact_sha256'] != marker['methods']:
        raise ValueError('Candidate pipeline did not evaluate the prepared bundle.')
    result = operational_decision(records['baseline_pipeline'], records['candidate_pipeline'], *rows,
        records['baseline_benchmark'], records['candidate_benchmark'], protocol)
    paths.update({'protocol': ROOT / 'configs/deployment_protocol.json',
                  'study': ROOT / 'reports/research/study_summary.json'})
    result.update(candidate=marker['candidate'], recorded_utc=utc_now(), artifact_sha256=marker['methods'],
        evidence_checksums={path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths.values()})
    write_json(ASSESSMENT, result)
    print(json.dumps(result), flush=True)


def activate():
    marker = verify_prepared()
    result = read(ASSESSMENT)
    if not result['qualified'] or result['artifact_sha256'] != marker['methods']:
        raise ValueError('All operational gates must pass for this exact bundle.')
    for relative, digest in result['evidence_checksums'].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError('Operational evidence changed after assessment.')
    checks = ('reports/backend_cuda_smoke.json', 'reports/api_http_smoke.json',
              'reports/face_pipeline_independent_review.json', 'reports/extended_evaluation/independent_review.json')
    for relative in checks:
        record = read(ROOT / relative)
        if not record.get('ok'):
            raise ValueError('Every independent and functional check must pass.')
        if 'artifact_sha256' in record and record['artifact_sha256'] != marker['methods']:
            raise ValueError('Functional checks evaluated another artifact bundle.')
    pipeline_review = read(ROOT / checks[2])
    if pipeline_review['report_sha256'] != sha256_file(ROOT / 'reports/face_pipeline_validation.json'):
        raise ValueError('Independent pipeline review is stale.')
    review = read(ROOT / checks[3])
    for filename, digest in review['evidence_checksums'].items():
        if sha256_file(ROOT / 'reports/extended_evaluation' / filename) != digest:
            raise ValueError('AP/ROC review is stale.')
    marker.update(status='active', activated_utc=utc_now(), assessment_sha256=sha256_file(ASSESSMENT),
                  final_checks={relative: sha256_file(ROOT / relative) for relative in checks},
                  scope='Verified local default bundle. Original and research artifacts preserved; no public deployment.')
    write_json(MARKER, marker)
    print('Activated the verified local bundle for new API/CLI processes.', flush=True)


def finalize():
    study = read(ROOT / 'reports/research/study_summary.json')
    review = read(ROOT / 'reports/research/independent_review.json')
    if not review.get('ok') or not review.get('study_complete'):
        raise ValueError('The complete study must pass independent verification.')
    active = default_artifacts_dir()
    hashes = {method: sha256_file(active / method / ('svm.npz' if method == 'classical' else 'best.pt')) for method in METHODS}
    paths = ['reports/research/study_summary.json', 'reports/research/independent_review.json',
             'configs/deployment_protocol.json', 'reports/backend_benchmark.json',
             'reports/backend_cuda_smoke.json', 'reports/api_http_smoke.json',
             'reports/face_pipeline_validation.json', 'reports/extended_evaluation/classification_roc.json']
    for relative in ('reports/backend_benchmark.json', 'reports/backend_cuda_smoke.json',
                     'reports/api_http_smoke.json', 'reports/face_pipeline_validation.json'):
        record = read(ROOT / relative)
        if not record.get('ok') or record['artifact_sha256'] != hashes:
            raise ValueError('Final system evidence must match the active artifacts.')
    roc = read(ROOT / 'reports/extended_evaluation/classification_roc.json')
    if {method: value['artifact_sha256'] for method, value in roc['methods'].items()} != hashes:
        raise ValueError('Final ROC/accuracy evidence must match active artifacts.')
    if active == BUNDLE:
        marker = verify_prepared()
        assessment = read(ASSESSMENT)
        if marker['status'] != 'active' or not assessment['qualified']:
            raise ValueError('An active candidate must pass every operational gate.')
        status, reason = 'active_candidate', 'Paired-seed research and all operational gates passed; seed42 candidate activated locally.'
        paths += ['artifacts/serving/deployment.json', 'reports/deployment/assessment.json']
    else:
        status = 'baseline_retained'
        reason = 'Original verified models retained. '
        reason += ('Research promotion criteria were not all satisfied.' if not study['promotion']['promote']
                   else 'Research candidate qualified, but operational activation was not completed or did not pass every gate.')
        if not study['promotion']['checks']['senior_recall_drop_at_most_0_01']:
            drop = (study['promotion']['baseline_mean']['senior_recall']
                    - study['promotion']['candidate_mean']['senior_recall']) * 100
            reason += f' Mean senior recall fell by {drop:.2f} percentage points, exceeding the 1-point limit.'
    write_json(ROOT / 'reports/deployment/decision.json', {
        'ok': True, 'status': status, 'reason': reason, 'recorded_utc': utc_now(),
        'research_qualified': study['promotion']['promote'], 'proposed_champion': study['proposed_champion'],
        'research_promotion': study['promotion'],
        'active_artifacts_dir': active.relative_to(ROOT).as_posix(), 'artifact_sha256': hashes,
        'public_deployment': False, 'originals_preserved': True,
        'evidence_checksums': {relative: sha256_file(ROOT / relative) for relative in paths}})
    print(reason, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'assess', 'activate', 'finalize'), required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'assess': assess, 'activate': activate, 'finalize': finalize}[args.stage]()


if __name__ == '__main__':
    main()

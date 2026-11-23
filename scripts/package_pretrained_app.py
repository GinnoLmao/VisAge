"""Build a new inference package only from completed verified v2 evidence."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.faces import DETECTOR_FILENAME, DETECTOR_SHA256, DETECTOR_LICENSE_SHA256
from visage.model_registry import METHODS, ARTIFACTS
from visage.pretrained_study import verify_protocol
from visage.runtime import sha256_file, utc_now, write_json


def main():
    verify_protocol()
    reports = ROOT / 'reports/pretrained'
    def read(relative):
        return json.loads((reports / relative).read_text(encoding='utf-8'))
    for relative in ('verified_runs.json', 'frozen_head_verification.json', 'test_summary.json', 'robustness.json', 'face_pipeline_validation.json',
        'device_verification.json', 'evaluation/detection_lineage.json', 'web_app/automated_acceptance.json',
        'web_app/visual_review.json', 'web_app/http_verification.json', 'web_app/wheel_assets.json'):
        if not read(relative)['ok']:
            raise ValueError(f'Completion gate is missing: {relative}')
    if read('verified_runs.json')['count'] != 18:
        raise ValueError('All18 declared runs are required.')
    if read('study_status.json')['status'] != 'complete':
        raise ValueError('Training is still incomplete.')
    for filename in ('tests_full.xml', 'tests_javascript.xml'):
        cases = ET.parse(reports / 'web_app' / filename).findall('.//testcase')
        if not cases or any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped')):
            raise ValueError('Project regression tests must all pass without skips before delivery.')
    for name in ('production_cleanup.json', 'headless_cleanup.json'):
        record = read(f'web_app/{name}')
        if name.startswith('headless'):
            if not record['browser_process_stopped'] or record['remaining_browser_processes']:
                raise ValueError('Owned browser processes remain.')
            record = record['server']
        if not record['server_process_stopped'] or not record['port_closed']:
            raise ValueError('Owned preview server remains.')
    benchmark = read('benchmark/backend_benchmark.json')
    if not benchmark['ok'] or not benchmark['no_concurrent_training'] or not benchmark['cleanup']['port_closed']:
        raise ValueError('Isolated serving benchmark is incomplete.')
    bundle = json.loads((ARTIFACTS / 'bundle.json').read_text())
    notices = json.loads((ROOT / 'licenses/pretrained/provenance.json').read_text())
    for relative, digest in notices['files'].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError('Published license/card evidence changed.')
    original_notices = json.loads((ROOT / 'licenses/pretrained/original_sources/provenance.json').read_text())
    for record in original_notices['repositories'].values():
        if sha256_file(ROOT / record['path']) != record['sha256']:
            raise ValueError('Original software copyright notice changed.')
    for method in METHODS:
        if sha256_file(ARTIFACTS / method / 'best.pt') != bundle['methods'][method]['artifact_sha256']:
            raise ValueError('Serving artifacts changed.')
        summary = json.loads((ARTIFACTS / method / 'training_summary.json').read_text())
        if sha256_file(ROOT / f'licenses/pretrained/{method}/README.md') != summary['pretrained']['model_card_sha256']:
            raise ValueError('Packaged model card differs from the selected checkpoint provenance.')
    detector = ROOT / '.cache/face_detector'
    if sha256_file(detector / DETECTOR_FILENAME) != DETECTOR_SHA256 or sha256_file(detector / 'LICENSE') != DETECTOR_LICENSE_SHA256:
        raise ValueError('Detector/license identity mismatch.')
    # The previous delivery is a separately preserved rollback; never overwrite it.
    old = json.loads((reports / 'start.json').read_text())['rollback_archive']
    if sha256_file(Path(old['path'])) != old['zip_sha256']:
        raise ValueError('Historical rollback archive changed.')
    candidates = []
    for folder in ('src', 'scripts', 'tests', 'configs', 'licenses'):
        candidates += [p for p in (ROOT / folder).rglob('*') if p.is_file()]
    candidates += [p for p in reports.rglob('*') if p.is_file() and 'v1_sources' not in p.parts]
    candidates += [p for p in ARTIFACTS.rglob('*') if p.is_file()]
    candidates += [ROOT / p for p in ('README.md', 'pyproject.toml', 'package.json', '.gitignore',
        'requirements-lock-windows-cu128.txt', 'requirements-lock-backend-windows-cu128.txt',
        'data/provenance.json', 'data/processed/manifest.csv', 'data/detection_benchmark/provenance.json')]
    candidates += [detector / p for p in (DETECTOR_FILENAME, 'LICENSE', 'provenance.json')]
    # The final supervision report is generated after this archive's isolated
    # verification and delivered alongside it, avoiding a self-hash dependency.
    candidates += [ROOT / 'docs/VisAge_Project_Documentation.md']
    candidates += [p for p in (ROOT / 'output/wheels').glob('visage_cv-0.4.0-*.whl')]
    candidates = sorted(set(p.resolve() for p in candidates if p.is_file() and '__pycache__' not in p.parts
        and p.suffix not in ('.log', '.pid', '.err', '.partial') and not p.name.endswith('_progress.json')
        and not (p.is_relative_to(reports / 'web_app') and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.pdf'))))
    for path in candidates:
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(('data/raw/', '.venv/', '.cache/web_qa/', 'artifacts/pretrained/runs/')) or path.name == 'last.pt' or path.suffix.lower() in ('.jpg', '.jpeg', '.webp'):
            raise ValueError(f'Forbidden package entry: {relative}')
    manifest = {'created_utc': utc_now(), 'scope': 'VisAge local app with exactly three pretrained age-group classifiers',
        'default_method': bundle['default_method'], 'artifact_bundle': bundle,
        'excluded': ['research face photos', 'visitor images', 'image caches', 'environment', 'resume optimizer states', 'historical delivery ZIPs'],
        'remaining_scope': ['physical webcam and real-phone acceptance', 'independent consented participant pilot', 'public hosting'],
        'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in candidates]}
    destination = ROOT / 'output/VisAge_Pretrained_Web_App.zip'
    if destination.exists():
        raise FileExistsError('Choose a versioned package; do not overwrite an existing delivered package.')
    write_json(ROOT / 'output/pretrained_package_manifest.json', manifest)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in candidates:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('pretrained_package_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise ValueError('Package CRC validation failed.')
        for item in manifest['files']:
            if hashlib.sha256(archive.read(item['path'])).hexdigest() != item['sha256']:
                raise ValueError('Packaged entry checksum mismatch.')
    result = {'ok': True, 'path': str(destination), 'bytes': destination.stat().st_size,
        'sha256': sha256_file(destination), 'files': len(manifest['files']), 'created_utc': utc_now(),
        'fresh_extraction_verification': 'Pending separate real startup and prediction check'}
    write_json(reports / 'package.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

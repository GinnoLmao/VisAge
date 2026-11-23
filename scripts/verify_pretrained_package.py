"""Verify a fresh extraction using its own source, models, detector and reports."""
import hashlib
import json
import os
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
import httpx
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from benchmark_backend import wait_for_server, stop_owned_server
from visage.runtime import write_json, sha256_file, utc_now
from visage.model_registry import METHODS


def main():
    package = ROOT / 'output/VisAge_Pretrained_Web_App.zip'
    destination = ROOT / 'tmp' / f'pretrained_delivery_{uuid.uuid4().hex[:10]}'
    destination.mkdir(parents=True)
    with zipfile.ZipFile(package) as archive:
        for name in archive.namelist():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive extraction path.')
        archive.extractall(destination)
    manifest = json.loads((destination / 'pretrained_package_manifest.json').read_text())
    for item in manifest['files']:
        if sha256_file(destination / item['path']) != item['sha256']:
            raise ValueError('Fresh extraction differs from package manifest.')
    # -I prevents the editable project installation from supplying local source;
    # insert only this extraction's src before importing the server entry point.
    code = 'import runpy,sys;from pathlib import Path;root=Path.cwd();sys.path.insert(0,str(root/"src"));sys.argv=["serve_api.py","--port","8767"];runpy.run_path(str(root/"scripts/serve_api.py"),run_name="__main__")'
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2')
    report_dir = ROOT / 'reports/pretrained'
    log_path = report_dir / 'package_server.log'
    cleanup = {}
    with log_path.open('w', encoding='utf-8') as log:
        server = subprocess.Popen([sys.executable, '-I', '-u', '-c', code], cwd=destination, env=environment,
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            with httpx.Client(base_url='http://127.0.0.1:8767', timeout=60, trust_env=False) as client:
                wait_for_server(server, client, log_path, 8767, 120)
                page = client.get('/'); page.raise_for_status()
                metadata = client.get('/v1/models').json()
                if [m['id'] for m in metadata['methods']] != list(METHODS):
                    raise ValueError('Extracted package exposes incorrect methods.')
                evaluation = client.get('/v1/evaluation'); evaluation.raise_for_status()
                if set(evaluation.json()['classification']['methods']) != set(METHODS):
                    raise ValueError('Extracted package evaluation is incomplete.')
                fixture = ROOT / '.cache/web_qa/single.jpg'
                result = client.post('/v1/predict?method=all&face_index=0', content=fixture.read_bytes(), headers={'Content-Type': 'image/jpeg'})
                result.raise_for_status()
                predicted = result.json()
                if [p['method'] for p in predicted['predictions']] != list(METHODS):
                    raise ValueError('Extracted app did not run all three classifiers.')
                expected = json.loads((ROOT / 'reports/pretrained/web_app/http_verification.json').read_text())['checks']['fixtures']['single.jpg']['labels']
                if [p['class_id'] for p in predicted['predictions']] != expected:
                    raise ValueError('Extracted app labels differ from the verified workspace app.')
        finally:
            cleanup = stop_owned_server(server, 8767)
    record = {'ok': True, 'verified_utc': utc_now(), 'package_sha256': sha256_file(package),
        'extraction': str(destination), 'manifest_files_verified': len(manifest['files']),
        'isolated_python_mode': True, 'source_root': str(destination / 'src'),
        'external_fixture_note': 'Existing validation fixture supplied as HTTP bytes; not part of the delivered ZIP.',
        'fixture_sha256': sha256_file(ROOT / '.cache/web_qa/single.jpg'),
        'methods': list(METHODS), 'default_method': metadata['default_method'],
        'labels': {p['method']: p['class_id'] for p in predicted['predictions']},
        'shared_crop_sha256': predicted['face_processing']['pixel_sha256'], 'cleanup': cleanup}
    write_json(report_dir / 'package_verification.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

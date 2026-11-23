"""Execute acceptance against real v2 artifacts, preserving all historical logs."""
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from visage.runtime import write_json, sha256_file, utc_now
from benchmark_backend import wait_for_server, stop_owned_server

REPORTS = ROOT / 'reports/pretrained/web_app'


def run(arguments, name):
    log = REPORTS / f'{name}.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect {log.relative_to(ROOT)}')


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    status = json.loads((ROOT / 'reports/pretrained/active_study.json').read_text())
    verified = json.loads((ROOT / 'reports/pretrained/verified_runs.json').read_text())
    if status['completed'] != 6 or not status['training_stopped'] or not verified['ok'] or verified['count'] != 6:
        raise ValueError('The six approved completed runs must be verified before acceptance.')
    run([sys.executable, '-m', 'pytest', '-q', '--junitxml=reports/pretrained/web_app/tests_full.xml'], 'python_tests')
    run(['node', '--test', '--test-reporter=junit', 'tests/web/core.test.mjs'], 'javascript_tests')
    # Keep machine-readable test output in a conventional XML filename.
    (REPORTS / 'tests_javascript.xml').write_bytes((REPORTS / 'javascript_tests.log').read_bytes())
    for filename in ('tests_full.xml', 'tests_javascript.xml'):
        cases = ET.parse(REPORTS / filename).findall('.//testcase')
        if not cases or any(case.find(tag) is not None for case in cases for tag in ('error', 'failure', 'skipped')):
            raise ValueError(f'Regression suite is incomplete: {filename}')
    run([sys.executable, '-m', 'pip', 'check'], 'dependencies')
    run([sys.executable, '-m', 'ruff', 'check', 'src', 'scripts', 'tests'], 'lint')
    run([sys.executable, 'scripts/check_pretrained_devices.py'], 'devices')
    run([sys.executable, 'scripts/benchmark_backend.py', '--training-idle-confirmed', '--iterations', '30',
        '--warmups', '5', '--port', '8766', '--output', 'reports/pretrained/benchmark/backend_benchmark.json'], 'http_benchmark')
    import httpx
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 8000))
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2')
    log_path = REPORTS / 'production_server.log'
    with log_path.open('w', encoding='utf-8') as log:
        server = subprocess.Popen([sys.executable, '-u', 'scripts/serve_api.py', '--port', '8000'], cwd=ROOT,
            env=environment, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            with httpx.Client(base_url='http://127.0.0.1:8000', trust_env=False, timeout=30) as client:
                wait_for_server(server, client, log_path, 8000, 120)
            run([sys.executable, 'scripts/verify_web_app.py', '--fixtures', '.cache/web_qa'], 'production_http')
        finally:
            write_json(REPORTS / 'production_cleanup.json', stop_owned_server(server, 8000))
    run([sys.executable, 'scripts/run_browser_qa.py'], 'browser_integration')
    counts = {filename: len(ET.parse(REPORTS / filename).findall('.//testcase'))
        for filename in ('tests_full.xml', 'tests_javascript.xml')}
    write_json(REPORTS / 'automated_acceptance.json', {'ok': True, 'completed_utc': utc_now(),
        'counts': counts, 'checks': ['full Python regression', 'JavaScript regression', 'dependency consistency',
            'lint', 'CPU CUDA consistency', 'isolated HTTP latency', 'real HTTP errors and shared crop', 'browser upload camera evaluation'],
        'visual_review': 'Pending separate inspection of desktop/mobile screenshots',
        'physical_camera_and_phone': 'Pending hardware acceptance; simulated browser camera was exercised',
        'local_participant_pilot': 'Pending consented data'})
    print(json.dumps({'ok': True, 'counts': counts}), flush=True)


if __name__ == '__main__':
    main()

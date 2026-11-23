"""Record actual lint/dependency checks and extract real integration test evidence.

Run with the project environment after pytest writes the complete JUnit report.
No tests are invented or re-executed by the extraction step.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import sha256_file, utc_now, write_json


def main():
    cases = ET.parse(ROOT / 'reports/tests_backend_research.xml').findall('.//testcase')
    if not cases or any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped')):
        raise ValueError('A complete successful JUnit report is required.')
    actual = [case for case in cases if case.get('classname', '').split('.')[-1] == 'test_integration']
    if len(actual) != 5:
        raise ValueError('Expected all five real-model/API/CLI integration checks.')
    suite = ET.Element('testsuite', name='actual_real_artifact_integration', tests=str(len(actual)),
                       failures='0', errors='0', skipped='0', time=str(sum(float(case.get('time', '0')) for case in actual)))
    suite.extend(deepcopy(actual))
    ET.ElementTree(suite).write(ROOT / 'reports/tests_integration.xml', encoding='utf-8', xml_declaration=True)
    commands = []
    for arguments in (['-m', 'ruff', 'check', 'src', 'scripts', 'tests'], ['-m', 'pip', 'check']):
        command = [sys.executable, *arguments]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        commands.append({'argv': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            raise RuntimeError('A final software check failed: ' + result.stdout + result.stderr)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--exclude', 'visage-cv'],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    lock = ROOT / 'requirements-lock-backend-windows-cu128.txt'
    lock.write_text(freeze.stdout, encoding='utf-8')
    write_json(ROOT / 'reports/final_software_checks.json', {
        'ok': True, 'checked_utc': utc_now(), 'commands': commands,
        'full_test_report_sha256': sha256_file(ROOT / 'reports/tests_backend_research.xml'),
        'full_suite_tests': len(cases), 'integration_tests': len(actual),
        'integration_report_sha256': sha256_file(ROOT / 'reports/tests_integration.xml'),
        'integration_note': 'Exact testcase records extracted from the final full-suite execution; no additional test runs or fabricated results.',
        'lock_sha256': sha256_file(lock), 'python': sys.version,
        'packages': {name: metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'fastapi', 'uvicorn', 'opencv-python-headless', 'httpx', 'psutil')}})
    print(f'Verified lint/dependencies; {len(cases)} successful tests, including {len(actual)} real integration checks.')


if __name__ == '__main__':
    main()

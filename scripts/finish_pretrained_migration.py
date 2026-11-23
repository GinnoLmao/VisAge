"""Continue completed training through real evaluation and app acceptance."""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import write_json, sha256_file, utc_now

REPORTS = ROOT / 'reports/pretrained'


def record(status, phase, **extra):
    write_json(REPORTS / 'migration_status.json', {'status': status, 'phase': phase,
        'pid': os.getpid(), 'updated_utc': utc_now(), **extra})


def run(arguments, name):
    record('running', name)
    print(json.dumps({'event': 'phase_start', 'phase': name}), flush=True)
    with (REPORTS / f'{name}.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect reports/pretrained/{name}.log')
    print(json.dumps({'event': 'phase_complete', 'phase': name}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-training', action='store_true')
    parser.add_argument('--wait-for-final-review', action='store_true')
    args = parser.parse_args()
    keep_awake = getattr(getattr(ctypes, 'windll', None), 'kernel32', None)
    if keep_awake:
        keep_awake.SetThreadExecutionState(0x80000001)
    try:
        while True:
            status = json.loads((REPORTS / 'study_status.json').read_text())
            if status['status'] == 'complete' and status['completed'] == 18:
                break
            if status['status'] == 'interrupted':
                raise RuntimeError('Training interrupted; inspect study_status before resuming.')
            if not args.wait_for_training:
                raise ValueError('Training is incomplete; use --wait-for-training for continuation.')
            record('waiting', 'training', completed=status['completed'], total=18, current=status.get('current'))
            time.sleep(15)
        # Allow the completed training process to release its final CUDA context.
        time.sleep(2)
        run([sys.executable, 'scripts/check_pretrained_frozen_heads.py'], 'frozen_heads')
        run([sys.executable, '-u', 'scripts/evaluate_pretrained_study.py'], 'evaluation')
        run([sys.executable, '-u', 'scripts/evaluate_pretrained_pipeline.py'], 'pipeline_evaluation')
        run([sys.executable, '-u', 'scripts/reuse_pretrained_detector_evidence.py'], 'detector_lineage')
        run([sys.executable, 'scripts/plot_pretrained_study.py'], 'figures')
        run([sys.executable, 'scripts/stage_pretrained_app.py', '--apply'], 'app_switch')
        run([sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-deps', '--no-build-isolation'], 'editable_install')
        run([sys.executable, '-m', 'pip', 'wheel', '.', '--no-deps', '--no-build-isolation', '-w', 'output/wheels'], 'wheel_build')
        wheels = list((ROOT / 'output/wheels').glob('visage_cv-0.4.0-*.whl'))
        if len(wheels) != 1:
            raise ValueError('Exactly one v0.4 wheel is required.')
        assets = [p.name for p in (ROOT / 'src/visage/web').glob('*') if p.is_file()]
        with zipfile.ZipFile(wheels[0]) as archive:
            for name in assets:
                if archive.read(f'visage/web/{name}') != (ROOT / 'src/visage/web' / name).read_bytes():
                    raise ValueError(f'Wheel web asset mismatch: {name}')
        write_json(REPORTS / 'web_app/wheel_assets.json', {'ok': True, 'wheel': wheels[0].name,
            'sha256': sha256_file(wheels[0]), 'assets': assets})
        run([sys.executable, '-u', 'scripts/accept_pretrained_app.py'], 'acceptance')
        run([sys.executable, 'scripts/write_pretrained_readme.py'], 'readme')
        packages = json.loads(subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'], cwd=ROOT, text=True))
        locked = sorted(f'{p["name"]}=={p["version"]}' for p in packages if p['name'].lower().replace('_', '-') != 'visage-cv')
        (REPORTS / 'requirements-lock-pretrained-windows-cu128.txt').write_text('\n'.join(locked) + '\n', encoding='utf-8')
        record('awaiting_visual_review', 'automated_acceptance_complete',
            next='Inspect desktop/mobile app screenshots, review actual results, then package, extract/verify and generate final report.')
        if args.wait_for_final_review:
            while not (REPORTS / 'final_complete.json').is_file():
                time.sleep(5)
            record('complete', 'delivered')
    except BaseException as error:
        record('interrupted', 'inspect_failure', error=repr(error))
        raise
    finally:
        if keep_awake:
            keep_awake.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

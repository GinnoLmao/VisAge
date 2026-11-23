"""Freeze and execute exactly the declared18 jobs, one CUDA model at a time."""
import argparse
import ctypes
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import configure_caches, sha256_file, write_json, utc_now
configure_caches(ROOT)
from visage.pretrained_study import freeze_protocol, verify_protocol, PROTOCOL
from visage.pretrained_training import train_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    protocol = freeze_protocol()
    if args.freeze_only:
        print(json.dumps({'protocol_sha256': sha256_file(PROTOCOL), 'runs': len(protocol['runs'])}), flush=True)
        return
    # Prevent this owned long calculation from putting the machine to sleep.
    # This thread-scoped request is restored on exit; no OS power plan is edited.
    keep_awake = getattr(getattr(ctypes, 'windll', None), 'kernel32', None)
    if keep_awake:
        keep_awake.SetThreadExecutionState(0x80000001)
    results = []
    current = None
    try:
        for config in protocol['runs']:
            verify_protocol()
            current = config['run_id']
            write_json(ROOT / 'reports/pretrained/study_status.json', {'status': 'running',
                'current': current, 'completed': len(results), 'total': len(protocol['runs']), 'updated_utc': utc_now()})
            print(json.dumps({'event': 'run_start', 'run': current, 'completed': len(results)}), flush=True)
            results.append(train_run(config, sha256_file(PROTOCOL)))
            write_json(ROOT / 'reports/pretrained/run_results.json', results)
        verify_protocol()
        write_json(ROOT / 'reports/pretrained/study_status.json', {'status': 'complete',
            'completed': len(results), 'total': len(protocol['runs']), 'updated_utc': utc_now()})
    except BaseException as error:
        write_json(ROOT / 'reports/pretrained/study_status.json', {'status': 'interrupted',
            'current': current, 'completed': len(results), 'total': len(protocol['runs']),
            'error': repr(error), 'updated_utc': utc_now(), 'resume': 'Run the same script; completed jobs are verified and skipped.'})
        raise
    finally:
        if keep_awake:
            keep_awake.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

"""Inference-only steps1–4 using the six completed runs; never restart training."""
import argparse
import ctypes
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import configure_caches, seed_everything
configure_caches(ROOT)
from visage.pretrained_two_seed import (freeze_amendment, freeze_evaluation_amendment,
    verify_six_runs, prepare_two_seed_serving, cpu_latency, select_two_seed_default)
from visage.pretrained_evaluation import validation_roc, test_once, robustness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    seed_everything(42, threads=4)
    freeze_amendment()
    freeze_evaluation_amendment()
    if args.freeze_only:
        print('Two-seed training-scope and evaluation amendments frozen. No training started.', flush=True)
        return
    awake = getattr(getattr(ctypes, 'windll', None), 'kernel32', None)
    if awake:
        awake.SetThreadExecutionState(0x80000001)
    try:
        records, validation = verify_six_runs()
        prepare_two_seed_serving(records)
        latency = cpu_latency(validation)
        decision = select_two_seed_default(records, latency)
        print(json.dumps({'event': 'selection_frozen', 'default': decision['default_method'],
            'means': decision['validation_means']}), flush=True)
        validation_roc(validation)
        test_once()
        robustness(validation)
    finally:
        if awake:
            awake.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

"""Run only after all18 jobs; select on validation, then evaluate frozen models."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import configure_caches, seed_everything
configure_caches(ROOT)
from visage.pretrained_evaluation import freeze_evaluation, validate_runs, prepare_serving, cpu_latency, choose_default, validation_roc, test_once, robustness

if __name__ == '__main__':
    seed_everything(42, threads=4)
    freeze_evaluation()
    records, validation = validate_runs()
    prepare_serving(records)
    latency = cpu_latency(validation)
    choice = choose_default(records, latency)
    print('Validation-selected default:', choice['default_method'], flush=True)
    validation_roc(validation)
    test_once()
    robustness(validation)

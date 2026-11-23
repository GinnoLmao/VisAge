"""Build the separate canonical224-to256 training/validation cache."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.pretrained_data import prepare_cache

if __name__ == '__main__':
    record = prepare_cache()
    print({k: record[k] for k in ('complete', 'rows', 'sha256', 'bytes')})

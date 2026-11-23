"""Vercel ASGI entry point; selected checkpoints are restored at build time."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for key in ('HF_HOME', 'TORCH_HOME', 'MPLCONFIGDIR'):
    os.environ.setdefault(key, '/tmp/visage/' + key.lower())
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

from visage.api import APISettings, create_app
from visage.runtime import seed_everything
from visage.service import InferenceService


class HostedService(InferenceService):
    def describe(self):
        return {**super().describe(), 'input_limits': {'max_bytes': 4 * 1024**2, 'max_pixels': self.config.max_pixels}}


def load_service():
    seed_everything(42, threads=2)
    return HostedService(device='cpu')


app = create_app(APISettings(max_upload_bytes=4 * 1024**2), service_factory=load_service)

"""Verify entire frozen-baseline backbones stayed published while heads learned."""
import gc
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.model_registry import METHODS
from visage.pretrained_models import PretrainedAgeModel
from visage.runtime import configure_caches, seed_everything, sha256_file, write_json, utc_now
configure_caches(ROOT)


def main():
    import timm  # Import before seeding, as in the later frozen-baseline jobs.
    checks = {}
    for method in METHODS:
        seed_everything(42, threads=4)
        directory = ROOT / '.cache/pretrained_weights' / method
        provenance = json.loads((directory / 'provenance.json').read_text())
        model = PretrainedAgeModel(method, weights=directory / 'model.safetensors', provenance=provenance)
        run = ROOT / f'artifacts/pretrained/runs/{method}_frozen_seed42'
        result = json.loads((run / 'result.json').read_text())
        path = run / 'best.pt'
        if sha256_file(path) != result['artifact_sha256']:
            raise ValueError('Completed frozen-baseline artifact changed.')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['metadata']['method'] != method or saved['metadata']['seed'] != 42:
            raise ValueError('Frozen baseline metadata is inconsistent.')
        for key, value in model.backbone.state_dict().items():
            if not torch.equal(value, saved['state_dict']['backbone.' + key]):
                raise AssertionError(f'Frozen backbone tensor changed during training: {method} {key}')
        bias_updated = bool(torch.count_nonzero(saved['state_dict']['head.bias']).item())
        if torch.equal(model.head.weight, saved['state_dict']['head.weight']) or not bias_updated:
            raise AssertionError('Replacement classifier did not learn from its seeded initialization.')
        checks[method] = {'all_published_backbone_tensors_unchanged': True, 'replacement_head_changed': True,
            'initial_zero_bias_received_nonzero_update': bias_updated,
            'artifact_sha256': result['artifact_sha256'], 'selected_epoch': saved['metadata']['epoch'],
            'backbone_state_tensors_checked': len(model.backbone.state_dict())}
        del model, saved; gc.collect()
    write_json(ROOT / 'reports/pretrained/frozen_head_verification.json', {'ok': True, 'checked_utc': utc_now(),
        'source_sha256': sha256_file(Path(__file__)), 'methods': checks})
    print(json.dumps({'ok': True, 'models': len(checks)}), flush=True)


if __name__ == '__main__':
    main()

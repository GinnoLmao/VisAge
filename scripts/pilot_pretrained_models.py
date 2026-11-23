"""Verify strict initialization and measure common microbatch feasibility."""
from pathlib import Path
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import configure_caches, seed_everything, write_json, environment_info
configure_caches(ROOT)
import torch
from visage.model_registry import REGISTRY
from visage.pretrained_models import PretrainedAgeModel


def main():
    seed_everything(42, threads=4)
    if not torch.cuda.is_available():
        raise RuntimeError('The declared GPU pilot requires CUDA.')
    records = []
    for method, spec in REGISTRY.items():
        directory = ROOT / '.cache/pretrained_weights' / method
        provenance = json.loads((directory / 'provenance.json').read_text())
        model = PretrainedAgeModel(method, weights=directory / 'model.safetensors', provenance=provenance).cuda()
        assert model.pretrained_loaded
        reference = next(model.backbone.parameters()).detach().clone()
        model.set_backbone_trainable(False); model.train()
        assert not model.backbone.training and model.head.training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(8, 3, 256, 256, device='cuda'); y = torch.arange(8, device='cuda') % 3
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        assert torch.equal(reference, next(model.backbone.parameters()))
        model.set_backbone_trainable(True); model.train()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        trials = []
        for batch in (8, 16, 32):
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                x = torch.randn(batch, 3, 256, 256, device='cuda'); y = torch.arange(batch, device='cuda') % 3
                timings = []
                for step in range(6):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    with torch.autocast('cuda', dtype=dtype):
                        output = model(x); loss = torch.nn.functional.cross_entropy(output, y)
                    assert output.shape == (batch, 3) and torch.isfinite(loss)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
                    if step:
                        timings.append(time.perf_counter() - start)
                trials.append({'batch': batch, 'images_per_second': batch / (sum(timings) / len(timings)),
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2})
                if trials[-1]['peak_reserved_mib'] > 5300:
                    break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                trials.append({'batch': batch, 'out_of_memory': True}); break
        assert not torch.equal(reference, next(model.backbone.parameters()))
        record = {'method': method, 'pretrained_exact_match': True, 'freeze_and_unfreeze_verified': True,
            'parameters': sum(p.numel() for p in model.parameters()), 'amp_dtype': str(dtype), 'trials': trials}
        records.append(record); write_json(ROOT / 'reports/pretrained/pilot_progress.json', records)
        print(json.dumps(record), flush=True)
        del model, optimizer, x, y, output, loss, reference; gc.collect(); torch.cuda.empty_cache()
    feasible = [{v['batch'] for v in record['trials'] if not v.get('out_of_memory') and v['peak_reserved_mib'] <= 5300} for record in records]
    common = max(set.intersection(*feasible))
    write_json(ROOT / 'reports/pretrained/gpu_pilot.json', {'ok': True, 'models': records, 'microbatch': common,
        'accumulation': 32 // common, 'effective_batch': 32, 'environment': environment_info()})
    print(json.dumps({'common_microbatch': common, 'accumulation': 32 // common}), flush=True)


if __name__ == '__main__':
    main()

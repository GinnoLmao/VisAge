"""Measure channels-last before fixing the full training protocol."""
import sys
import json
import time
import gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import configure_caches, seed_everything, write_json
configure_caches(ROOT)
import torch
from visage.pretrained_models import PretrainedAgeModel
from visage.model_registry import METHODS

seed_everything(42)
records = []
for method in METHODS:
    directory = ROOT / '.cache/pretrained_weights' / method
    provenance = json.loads((directory / 'provenance.json').read_text())
    model = PretrainedAgeModel(method, weights=directory / 'model.safetensors', provenance=provenance).cuda()
    x = torch.randn(32, 3, 256, 256, device='cuda')
    model.eval()
    with torch.no_grad():
        reference = model(x).clone()
        model.to(memory_format=torch.channels_last)
        converted = model(x.contiguous(memory_format=torch.channels_last))
    difference = (reference - converted).abs().max().item()
    torch.testing.assert_close(reference, converted, atol=2e-5, rtol=2e-4)
    model.train()
    x = x.contiguous(memory_format=torch.channels_last)
    y = torch.arange(32, device='cuda') % 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(8):
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= 2:
            times.append(time.perf_counter() - start)
    record = {'method': method, 'fp32_layout_max_absolute_difference': difference,
        'images_per_second': 32 / (sum(times) / len(times)),
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2}
    records.append(record)
    print(json.dumps(record), flush=True)
    write_json(ROOT / 'reports/pretrained/layout_pilot.json', records)
    del model, optimizer, x, y, reference, converted, loss
    gc.collect(); torch.cuda.empty_cache()

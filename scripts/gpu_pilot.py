"""Verify CUDA training and profile the actual architectures; not accuracy evidence."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visage.runtime import configure_caches
ROOT = Path(__file__).resolve().parents[1]
configure_caches(ROOT)

import torch
from visage.runtime import environment_info, seed_everything, write_json
from visage.methods.ordinal import corn_loss, ordinal_probabilities
from visage.methods.visagenet import VisAgeNet
from visage.methods.convnext_ordinal import ConvNeXtOrdinal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "gpu_pilot.json")
    args = parser.parse_args()
    seed_everything()
    write_json(ROOT / "reports" / "environment.json", environment_info())
    if not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA environment is not working; no CPU fallback claimed.")
    results = []
    for name, factory in [("visagenet", VisAgeNet), ("convnext_ordinal", lambda: ConvNeXtOrdinal(pretrained=False))]:
        torch.cuda.empty_cache()
        model = factory().cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16, init_scale=1024)
        x = torch.randn(args.batch_size, 3, 224, 224, device="cuda")
        y = torch.arange(args.batch_size, device="cuda") % 3
        elapsed = []
        losses = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps + 3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x)
                loss = corn_loss(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite pilot loss for {name}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if step >= 3:
                elapsed.append(time.perf_counter() - start)
                losses.append(loss.item())
        probabilities = ordinal_probabilities(logits.float())
        torch.testing.assert_close(probabilities.sum(1), torch.ones(args.batch_size, device="cuda"))
        record = {"method": name, "batch_size": args.batch_size, "image_size": 224,
                  "steps_measured": args.steps, "seconds_per_step": sum(elapsed) / len(elapsed),
                  "images_per_second": args.batch_size * len(elapsed) / sum(elapsed),
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "first_loss": losses[0], "last_loss": losses[-1],
                  "data_kind": "synthetic diagnostic, not accuracy evaluation",
                  "pretrained_weights": False}
        record["amp_dtype"] = str(amp_dtype)
        print(record, flush=True)
        results.append(record)
        del model, optimizer, scaler, x, y, logits, loss, probabilities
    write_json(args.output, {"environment": environment_info(), "results": results})


if __name__ == "__main__":
    main()

"""Validation-selected neural training with resumable state and auditable outputs."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import load_manifest, verify_manifest
from .datasets import IMAGENET_MEAN, IMAGENET_STD, ManifestDataset, make_transform, seed_worker
from .metrics import classification_metrics
from .methods.ordinal import corn_loss, ordinal_probabilities
from .methods.visagenet import VisAgeNet
from .methods.convnext_ordinal import ConvNeXtOrdinal
from .runtime import environment_info, seed_everything, sha256_file, utc_now, write_json
from .augmentation import BATCH_AUGMENTATION_POLICY, prepare_training_batch


def construct_model(method, *, pretrained=False, pretrained_path=None, model_config=None):
    config = dict(model_config or {})
    if method == "visagenet":
        return VisAgeNet(**config)
    if method == "convnext_ordinal":
        return ConvNeXtOrdinal(pretrained=pretrained, pretrained_path=pretrained_path, **config)
    raise ValueError(f"Unknown method {method}")


def logits_to_probabilities(logits, head_type="corn"):
    return ordinal_probabilities(logits.float()) if head_type == "corn" else logits.float().softmax(1)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    partial.replace(path)


def load_neural_artifact(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_format") != 1:
        raise ValueError("Unsupported neural artifact format.")
    model = construct_model(checkpoint["method"], pretrained=False, model_config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate_model(model, loader, device, amp_dtype=None, normalization=None):
    model.eval()
    all_labels, all_predictions, all_values, all_indices = [], [], [], []
    numerator = 0.0
    count = 0
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        if images.dtype == torch.uint8:
            if normalization is None:
                raise ValueError("Uint8 evaluation inputs require explicit RGB normalization.")
            images = prepare_training_batch(images, *normalization, augment=False)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = corn_loss(logits, labels) if model.head_type == "corn" else nn.functional.cross_entropy(logits, labels)
        probabilities = logits_to_probabilities(logits, model.head_type)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Non-finite predictions during evaluation.")
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(probabilities.argmax(1).cpu().tolist())
        all_values.extend(probabilities.cpu().tolist())
        all_indices.extend(indices.tolist())
        eligible_observations = len(labels) + int((labels > 0).sum()) if model.head_type == "corn" else len(labels)
        numerator += float(loss) * eligible_observations
        count += eligible_observations
    return {"metrics": classification_metrics(all_labels, all_predictions), "loss": numerator / count,
            "labels": all_labels, "predictions": all_predictions, "values": all_values, "indices": all_indices}


def save_predictions(path, rows, result):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "relative_path", "age", "label", "prediction", "score_0", "score_1", "score_2"])
        for index, label, prediction, values in zip(result["indices"], result["labels"], result["predictions"], result["values"], strict=True):
            row = rows[index]
            writer.writerow([row["sample_id"], row["relative_path"], row["age"], label, prediction, *values])


def select_training_subset(rows, path, manifest_sha):
    """Select only complete frozen training groups; never alter validation."""
    subset = json.loads(Path(path).read_text(encoding="utf-8"))
    if (subset.get("format_version") != 1 or subset.get("split") != "train"
            or subset.get("manifest_sha256") != manifest_sha):
        raise ValueError("Training subset does not match the frozen training manifest.")
    identities = subset.get("sample_ids")
    if (not isinstance(identities, list) or not identities
            or any(not isinstance(value, str) for value in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError("Training subset sample IDs must be nonempty and unique.")
    selected_ids = set(identities)
    training = [row for row in rows if row["split"] == "train"]
    if not selected_ids.issubset({row["sample_id"] for row in training}):
        raise ValueError("Training subset contains an unknown or non-training sample.")
    selected = [row for row in training if row["sample_id"] in selected_ids]
    selected_groups = {row["group_id"] for row in selected}
    if any(row["group_id"] in selected_groups and row["sample_id"] not in selected_ids for row in training):
        raise ValueError("Training subsets must preserve complete perceptual groups.")
    return selected


def sampling_weights(labels, exponent=1.0):
    """Inverse-frequency exponent one reproduces the original sampler."""
    exponent = float(exponent)
    if not math.isfinite(exponent) or not 0 <= exponent <= 1:
        raise ValueError("sampler_exponent must be finite and between zero and one.")
    counts = np.bincount(labels, minlength=3)
    if len(counts) != 3 or np.any(counts == 0):
        raise ValueError("All three classes must be represented in training.")
    return torch.as_tensor((1.0 / counts[labels]) ** exponent, dtype=torch.double)


def backbone_warmup_factor(epoch, head_epochs, warmup_epochs):
    """Ramp only newly unfrozen backbone rates; zero preserves old behavior."""
    if warmup_epochs == 0 or epoch <= head_epochs:
        return 1.0
    return min((epoch - head_epochs) / warmup_epochs, 1.0)


def train_neural(config: dict, *, resume=False) -> dict:
    for key in ("epochs", "batch_size"):
        if int(config.get(key, 0)) < 1:
            raise ValueError(f"{key} must be positive.")
    if int(config.get("workers", 2)) < 0:
        raise ValueError("workers must be nonnegative.")
    if float(config.get("learning_rate", 0)) <= 0:
        raise ValueError("learning_rate must be positive.")
    sampler_exponent = float(config.get("sampler_exponent", 1.0))
    if not math.isfinite(sampler_exponent) or not 0 <= sampler_exponent <= 1:
        raise ValueError("sampler_exponent must be finite and between zero and one.")
    warmup_epochs = config.get("unfreeze_warmup_epochs", 0)
    if type(warmup_epochs) is not int or warmup_epochs < 0:
        raise ValueError("unfreeze_warmup_epochs must be a nonnegative integer.")
    method = config["method"]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_summary.json").exists() and not resume:
        raise FileExistsError("Training output is complete; use a new run directory instead of overwriting.")
    seed = int(config.get("seed", 42))
    seed_everything(seed, threads=int(config.get("threads", 4)))
    manifest = Path(config["manifest"])
    raw_dir = Path(config["raw_dir"])
    verify_manifest(manifest, raw_dir, verify_files=False)
    rows = load_manifest(manifest)
    training_rows = [row for row in rows if row["split"] == "train"]
    if config.get("train_subset_path"):
        subset_path = Path(config["train_subset_path"])
        training_rows = select_training_subset(rows, subset_path, sha256_file(manifest))
        config = {**config, "train_subset_sha256": sha256_file(subset_path)}
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not validation_rows:
        validation_rows = [row for row in rows if row["split"] == "val"]
    if not training_rows or not validation_rows:
        raise ValueError("Nonempty train and validation partitions required.")
    for partition in (training_rows, validation_rows):
        if {int(row["label"]) for row in partition} != {0, 1, 2}:
            raise ValueError("Train and validation partitions must each contain all three classes.")
        for row in partition:
            path = (raw_dir.resolve() / row["relative_path"]).resolve()
            if not path.is_relative_to(raw_dir.resolve()) or sha256_file(path) != row["file_sha256"]:
                raise ValueError(f"Frozen dataset file hash changed: {row['relative_path']}")
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16, init_scale=1024)
    if method == "convnext_ordinal":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    batch_augmentation = config.get("augmentation_backend", "pil") == "tensor_batch_v1"
    if config.get("augmentation_backend", "pil") not in {"pil", "tensor_batch_v1"}:
        raise ValueError("Unsupported augmentation backend.")
    augmentation_policy = BATCH_AUGMENTATION_POLICY if batch_augmentation else {
        "name": "pil_per_image", "description": "horizontal flip; rotation +/-8 degrees; mild brightness/contrast/saturation jitter"}
    config = {**config, "manifest_sha256": sha256_file(manifest),
              "transform": {"image_size": 224, "mean": list(mean), "std": list(std), "geometry": "whole aligned face resize, bicubic; no center crop",
                            "train_augmentation": augmentation_policy},
              "amp_dtype": str(amp_dtype)}
    if resume and (output / "config.json").exists():
        previous_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if previous_config != config:
            raise ValueError("Resume configuration changed; use a new run directory.")
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment_info())
    if config.get("tensor_cache_dir"):
        if not batch_augmentation:
            raise ValueError("Tensor face cache requires the tensor batch augmentation backend.")
        from .tensor_cache import ensure_tensor_cache, CachedFaceDataset
        ensure_tensor_cache(manifest, raw_dir, Path(config["tensor_cache_dir"]), image_size=224)
        train_ds = CachedFaceDataset(training_rows, Path(config["tensor_cache_dir"]))
        val_ds = CachedFaceDataset(validation_rows, Path(config["tensor_cache_dir"]))
    elif batch_augmentation:
        from torchvision import transforms as T
        uint8_transform = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.PILToTensor()])
        train_ds = ManifestDataset(training_rows, raw_dir, uint8_transform)
        val_ds = ManifestDataset(validation_rows, raw_dir, uint8_transform)
    else:
        train_ds = ManifestDataset(training_rows, raw_dir, make_transform(training=True, mean=mean, std=std))
        val_ds = ManifestDataset(validation_rows, raw_dir, make_transform(mean=mean, std=std))
    generator = torch.Generator().manual_seed(seed)
    labels = np.array([row["label"] for row in training_rows], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=3)
    if np.any(class_counts == 0):
        raise ValueError("All three classes must be represented in training.")
    weights = sampling_weights(labels, sampler_exponent)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    workers = int(config.get("workers", 2))
    loader_args = {"batch_size": int(config.get("batch_size", 16)), "num_workers": workers,
                   "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker,
                   "persistent_workers": workers > 0}
    train_loader = DataLoader(train_ds, sampler=sampler, generator=generator, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    model_config = dict(config.get("model_config", {}))
    source = config.get("pretrained_path")
    if source and config.get("pretrained_provenance"):
        if sha256_file(Path(source)) != config["pretrained_provenance"]["sha256"]:
            raise ValueError("Pretrained checkpoint hash differs from recorded provenance.")
    resuming = resume and (output / "last.pt").exists()
    model = construct_model(method, pretrained=method == "convnext_ordinal" and not resuming,
                            pretrained_path=source if not resuming else None,
                            model_config=model_config).to(device)
    head_epochs = int(config.get("head_epochs", 0)) if method == "convnext_ordinal" else 0
    if method == "convnext_ordinal":
        groups = [{"params": model.backbone.parameters(), "lr": config["learning_rate"]},
                  {"params": model.head.parameters(), "lr": config["head_learning_rate"]}]
        model.set_backbone_trainable(head_epochs == 0)
    else:
        groups = model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0005)))
    epochs = int(config["epochs"])
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    history = []
    best_score, best_epoch, start_epoch, bad_epochs = -1.0, 0, 1, 0
    total_seconds = 0.0
    if resume and (output / "last.pt").exists():
        last = torch.load(output / "last.pt", map_location=device, weights_only=True)
        if last["manifest_sha256"] != config["manifest_sha256"]:
            raise ValueError("Cannot resume against a different dataset manifest.")
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        scaler.load_state_dict(last["scaler_state"])
        generator.set_state(last["generator_state"].cpu())
        torch.set_rng_state(last["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in last["cuda_rng"]])
        random.setstate(last["python_rng"])
        np_state = last["numpy_rng"]
        np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), np_state[2], np_state[3], np_state[4]))
        history = last["history"]
        best_score, best_epoch = last["best_score"], last["best_epoch"]
        bad_epochs = last["bad_epochs"]
        total_seconds = last["total_seconds"]
        start_epoch = last["epoch"] + 1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"event": "training_started", "method": method, "train_samples": len(train_ds),
                      "validation_samples": len(val_ds), "class_counts": class_counts.tolist(),
                      "parameters": sum(p.numel() for p in model.parameters()), "config": config}), flush=True)
    for epoch in range(start_epoch, epochs + 1):
        if method == "convnext_ordinal":
            model.set_backbone_trainable(epoch > head_epochs)
        # Same smooth cosine schedule for each param group's initial rate.
        factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(epochs - 1, 1)))
        for group_index, (group, base) in enumerate(zip(optimizer.param_groups, base_lrs, strict=True)):
            ramp = backbone_warmup_factor(epoch, head_epochs, warmup_epochs) if method == "convnext_ordinal" and group_index == 0 else 1.0
            group["lr"] = base * factor * ramp
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        loss_total, loss_observations, correct, observations = 0.0, 0, 0, 0
        for batch_index, (images, targets, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            if batch_augmentation:
                images = prepare_training_batch(images, mean, std, augment=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
                loss = corn_loss(logits, targets) if model.head_type == "corn" else nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss, {method} epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            eligible_observations = len(targets) + int((targets > 0).sum()) if model.head_type == "corn" else len(targets)
            loss_total += float(loss.detach()) * eligible_observations
            loss_observations += eligible_observations
            correct += int((logits_to_probabilities(logits.detach(), model.head_type).argmax(1) == targets).sum())
            observations += len(targets)
            if batch_index % 64 == 0:
                print(json.dumps({"event": "training_batch_progress", "method": method,
                                  "epoch": epoch, "batches": batch_index,
                                  "total_batches": len(train_loader),
                                  "elapsed_seconds": time.perf_counter() - epoch_start}), flush=True)
        # Select checkpoints with full-precision inference, matching deployment evaluation.
        validation = evaluate_model(model, val_loader, device, None, normalization=(mean, std))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - epoch_start
        total_seconds += elapsed
        score = validation["metrics"]["macro_f1"]
        record = {"epoch": epoch, "train_loss": loss_total / loss_observations,
                  "train_accuracy_sampled": correct / observations, "val_loss": validation["loss"],
                  "val_accuracy": validation["metrics"]["accuracy"], "val_macro_f1": score,
                  "epoch_seconds": elapsed, "learning_rates": [g["lr"] for g in optimizer.param_groups],
                  "backbone_trainable": getattr(model, "backbone_trainable", True)}
        if method == "convnext_ordinal" and warmup_epochs:
            record["backbone_warmup_factor"] = backbone_warmup_factor(epoch, head_epochs, warmup_epochs)
        history.append(record)
        if score > best_score + 1e-5:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            artifact = {"artifact_format": 1, "method": method, "model_config": model_config,
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "manifest_sha256": config["manifest_sha256"], "transform": config["transform"],
                        "epoch": epoch, "seed": seed, "validation_metrics": validation["metrics"],
                        "pretrained_source": config.get("pretrained_provenance"), "created_utc": utc_now()}
            atomic_torch_save(output / "best.pt", artifact)
            write_json(output / "validation_metrics.json", validation["metrics"])
            save_predictions(output / "validation_predictions.csv", validation_rows, validation)
        else:
            bad_epochs += 1
        np_state = np.random.get_state()
        last = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(), "epoch": epoch, "history": history,
                "best_score": best_score, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                "total_seconds": total_seconds, "manifest_sha256": config["manifest_sha256"],
                "generator_state": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "python_rng": random.getstate(),
                "numpy_rng": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]]}
        atomic_torch_save(output / "last.pt", last)
        write_json(output / "history.json", history)
        print(json.dumps({"event": "epoch_complete", "method": method, **record, "best_epoch": best_epoch}), flush=True)
        if epoch >= int(config.get("min_epochs", 8)) and bad_epochs >= int(config.get("patience", 6)):
            break
    summary = {"method": method, "best_epoch": best_epoch, "epochs_completed": len(history),
               "training_rows": len(train_ds), "validation_rows": len(val_ds),
               "best_validation_macro_f1": best_score, "training_seconds": total_seconds,
               "parameters": sum(p.numel() for p in model.parameters()),
               "peak_training_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
               "artifact_sha256": sha256_file(output / "best.pt"), "artifact_bytes": (output / "best.pt").stat().st_size,
               "manifest_sha256": config["manifest_sha256"], "completed_utc": utc_now(),
               "selection": "highest validation macro F1; held-out test never read during training",
               "training_accuracy_note": ("Measured on augmented, class-balanced sampled training batches."
                   if sampler_exponent == 1 else f"Measured on augmented, inverse-frequency^{sampler_exponent:g} weighted replacement-sampled training batches; class masses are not balanced.")}
    summary["resume_reproducibility_note"] = "Optimizer/model/global RNG state is saved; persistent worker augmentation RNG is not bitwise-restored across process restarts."
    write_json(output / "training_summary.json", summary)
    print(json.dumps({"event": "training_complete", **summary}), flush=True)
    return summary

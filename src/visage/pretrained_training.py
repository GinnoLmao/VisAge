"""Resumable, validation-selected transfer learning; no test-pixel access."""
from __future__ import annotations
import csv
import gc
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .augmentation import prepare_training_batch
from .data import load_manifest
from .metrics import classification_metrics
from .model_registry import REGISTRY, ROOT
from .pretrained_data import FaceBatches, GEOMETRY, class_weights
from .pretrained_models import PretrainedAgeModel
from .runtime import seed_everything, sha256_file, utc_now, write_json


def atomic_save(path, record):
    path = Path(path)
    partial = path.with_suffix(path.suffix + '.partial')
    torch.save(record, partial)
    partial.replace(path)


def rng_state(generator):
    array_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [array_state[0], array_state[1].tolist(),
        array_state[2], array_state[3], array_state[4]], 'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'order': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    v = state['numpy']
    np.random.set_state((v[0], np.array(v[1], dtype=np.uint32), v[2], v[3], v[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['order'])


def learning_rates(epoch, config):
    # Epoch1 uses the declared rates; epoch24 reaches zero. The frozen
    # backbone has no gradients in warmup, while its schedule is still explicit.
    factor = .5 * (1 + math.cos(math.pi * (epoch - 1) / (config['max_epochs'] - 1)))
    return [config['backbone_lr'] * factor, config['head_lr'] * factor]


def train_epoch(model, data, optimizer, weights, spec, config, generator, *, device, dtype):
    model.train()
    order = torch.randperm(len(data.rows), generator=generator).numpy()
    total_loss = total_weight = 0.0
    predictions, truths = [], []
    effective, micro = config['effective_batch'], config['microbatch']
    for group_start in range(0, len(order), effective):
        group = order[group_start:group_start + effective]
        # Sum CE / sum class weights over the *whole* effective batch; averaging
        # weighted means of unequal microbatches is mathematically incorrect.
        denominator = weights[torch.as_tensor(data.labels[group], device=device)].sum()
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(group), micro):
            positions = group[start:start + micro]
            images, targets = data.batch(positions, device)
            inputs = prepare_training_batch(images, spec.mean, spec.std, augment=True)
            if config['memory_format'] == 'channels_last':
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
                logits = model(inputs)
                loss_sum = F.cross_entropy(logits, targets, weight=weights, reduction='sum')
                loss = loss_sum / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            loss.backward()
            total_loss += float(loss_sum.detach())
            total_weight += float(weights[targets].sum())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
            truths.extend(targets.cpu().tolist())
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
    return {'weighted_cross_entropy': total_loss / total_weight,
        **classification_metrics(np.array(truths), np.array(predictions))}


@torch.inference_mode()
def evaluate(model, data, spec, *, device, batch_size=32, memory_format='contiguous'):
    """FP32, no autocast, same input conversion as production inference."""
    model.eval()
    scores = []
    for start in range(0, len(data.rows), batch_size):
        positions = np.arange(start, min(start + batch_size, len(data.rows)))
        images, _ = data.batch(positions, device)
        inputs = prepare_training_batch(images, spec.mean, spec.std, augment=False)
        if memory_format == 'channels_last':
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite validation logits.')
        scores.append(logits.softmax(1).cpu().numpy())
    values = np.concatenate(scores)
    return classification_metrics(data.labels, values.argmax(1)), values


def save_predictions(path, rows, scores):
    path = Path(path)
    partial = path.with_suffix('.csv.partial')
    with partial.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'relative_path', 'age', 'label', 'prediction', 'score_0', 'score_1', 'score_2'])
        for row, score in zip(rows, scores, strict=True):
            writer.writerow([row['sample_id'], row['relative_path'], row['age'], row['label'], int(score.argmax()), *map(float, score)])
    partial.replace(path)


def train_run(config, protocol_sha256):
    directory = ROOT / config['output']
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('A resumed run must use its original configuration.')
    write_json(config_path, config)
    result_path = directory / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['protocol_sha256'] != protocol_sha256 or sha256_file(directory / 'best.pt') != result['artifact_sha256']:
            raise ValueError('Completed run identity mismatch.')
        return result
    seed_everything(config['seed'], threads=4)
    device = torch.device(config['device'])
    if device.type != 'cuda' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This frozen research protocol requires CUDA BF16.')
    rows = load_manifest(ROOT / 'data/processed/manifest.csv')
    train = [r for r in rows if r['split'] == 'train']
    if config['subset']:
        subset = json.loads((ROOT / config['subset']).read_text())
        chosen = set(subset['sample_ids'])
        train = [r for r in train if r['sample_id'] in chosen]
        if len(train) != len(chosen):
            raise ValueError('Training subset contains unknown/nontraining IDs.')
    validation = [r for r in rows if r['split'] == 'validation']
    train_data, validation_data = FaceBatches(train), FaceBatches(validation)
    if set(r['sample_id'] for r in train) & set(r['sample_id'] for r in validation):
        raise ValueError('Training/validation overlap.')
    method, spec = config['method'], REGISTRY[config['method']]
    weights_dir = ROOT / '.cache/pretrained_weights' / method
    provenance = json.loads((weights_dir / 'provenance.json').read_text())
    model = PretrainedAgeModel(method, weights=weights_dir / 'model.safetensors', provenance=provenance).to(device)
    if config['memory_format'] == 'channels_last':
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['backbone_lr']},
        {'params': model.head.parameters(), 'lr': config['head_lr']}], weight_decay=config['weight_decay'])
    weights = class_weights(train_data.labels).to(device)
    generator = torch.Generator().manual_seed(config['seed'])
    history, best, bad_epochs, first_epoch = [], -1.0, 0, 1
    elapsed_before = 0.0
    last_path = directory / 'last.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        if saved['config'] != config or saved['protocol_sha256'] != protocol_sha256:
            raise ValueError('Resume checkpoint belongs to a different protocol/configuration.')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        history, best, bad_epochs = saved['history'], saved['best'], saved['bad_epochs']
        first_epoch, elapsed_before = saved['epoch'] + 1, saved['elapsed_seconds']
        restore_rng(saved['rng'], generator)
        del saved
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    interrupted_early = bool(history and len(history) >= config['min_epochs'] and bad_epochs >= config['patience'])
    for epoch in range(first_epoch, config['max_epochs'] + 1):
        if interrupted_early:
            break
        epoch_start = time.perf_counter()
        trainable = not config['frozen_backbone'] and epoch > config['warmup_epochs']
        model.set_backbone_trainable(trainable)
        rates = learning_rates(epoch, config)
        for group, rate in zip(optimizer.param_groups, rates, strict=True):
            group['lr'] = rate
        train_metrics = train_epoch(model, train_data, optimizer, weights, spec, config, generator,
            device=device, dtype=torch.bfloat16)
        validation_metrics, scores = evaluate(model, validation_data, spec, device=device,
            batch_size=config['microbatch'], memory_format=config['memory_format'])
        improved = validation_metrics['macro_f1'] > best + config['minimum_improvement']
        metadata = {'format_version': 2, 'method': method, 'architecture': spec.checkpoint,
            'head_type': 'softmax', 'classes': ['Under 18', '18–59', '60+'], 'input': spec.to_dict(),
            'geometry': GEOMETRY, 'pretrained': provenance, 'protocol_sha256': protocol_sha256,
            'manifest_sha256': train_data.metadata['manifest_sha256'], 'seed': config['seed'],
            'epoch': epoch, 'run_id': config['run_id'], 'memory_format': config['memory_format'],
            'class_counts': np.bincount(train_data.labels, minlength=3).tolist(),
            'class_weights': weights.cpu().tolist(), 'validation_metrics': validation_metrics,
            'parameters': sum(p.numel() for p in model.parameters())}
        if improved:
            best, bad_epochs = validation_metrics['macro_f1'], 0
            atomic_save(directory / 'best.pt', {'metadata': metadata, 'state_dict': model.state_dict()})
            save_predictions(directory / 'validation_predictions.csv', validation, scores)
            write_json(directory / 'best.json', metadata)
        else:
            bad_epochs += 1
        record = {'epoch': epoch, 'stage': 'fine_tune' if trainable else 'head_only',
            'learning_rates': rates, 'train': train_metrics, 'validation': validation_metrics,
            'improved': improved, 'bad_epochs': bad_epochs,
            'seconds': time.perf_counter() - epoch_start}
        history.append(record)
        elapsed = elapsed_before + time.perf_counter() - started
        atomic_save(last_path, {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'config': config, 'protocol_sha256': protocol_sha256, 'history': history,
            'epoch': epoch, 'best': best, 'bad_epochs': bad_epochs, 'rng': rng_state(generator),
            'elapsed_seconds': elapsed})
        write_json(directory / 'history.json', history)
        print(json.dumps({'event': 'epoch', 'run': config['run_id'], 'epoch': epoch,
            'stage': record['stage'], 'train_macro_f1': train_metrics['macro_f1'],
            'validation_macro_f1': validation_metrics['macro_f1'], 'best': best,
            'bad_epochs': bad_epochs, 'seconds': record['seconds']}), flush=True)
        if epoch >= config['min_epochs'] and bad_epochs >= config['patience']:
            break
    result = {'run_id': config['run_id'], 'method': method, 'status': 'complete',
        'protocol_sha256': protocol_sha256, 'seed': config['seed'], 'epochs': len(history),
        'stop_reason': 'early_stopping' if len(history) < config['max_epochs'] else 'maximum_epochs',
        'best': json.loads((directory / 'best.json').read_text()),
        'artifact_sha256': sha256_file(directory / 'best.pt'),
        'predictions_sha256': sha256_file(directory / 'validation_predictions.csv'),
        'elapsed_seconds': elapsed_before + time.perf_counter() - started,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
        'completed_utc': utc_now()}
    write_json(result_path, result)
    del model, optimizer, train_data, validation_data, weights
    gc.collect(); torch.cuda.empty_cache()
    return result

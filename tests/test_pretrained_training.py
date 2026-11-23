"""Transfer-learning gates: provenance, reload, pixels, accumulation and resume."""
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from visage.model_registry import REGISTRY, ROOT
from visage.pretrained_data import pixels, FaceBatches, class_weights
from visage.pretrained_models import PretrainedAgeModel
from visage.pretrained_training import train_epoch, rng_state, restore_rng, atomic_save, learning_rates
from visage.runtime import seed_everything


@pytest.mark.parametrize('method', REGISTRY)
def test_verified_pretrained_head_freeze_and_full_checkpoint_reload(method, tmp_path):
    seed_everything(42, threads=2)
    directory = ROOT / '.cache/pretrained_weights' / method
    if not directory.exists():
        pytest.skip('Optional published weights not installed in this distribution.')
    provenance = json.loads((directory / 'provenance.json').read_text())
    model = PretrainedAgeModel(method, weights=directory / 'model.safetensors', provenance=provenance)
    assert model.pretrained_loaded
    if method == 'swinv2_tiny':
        assert len(model.verified_attention_masks) == 5
    model.set_backbone_trainable(False); model.train()
    assert not model.backbone.training and model.head.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    model.set_backbone_trainable(True); model.train()
    assert model.backbone.training and all(p.requires_grad for p in model.parameters())
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        expected = model(x)
    assert expected.shape == (1, 3) and torch.isfinite(expected).all()
    path = tmp_path / 'model.pt'
    atomic_save(path, {'state_dict': model.state_dict()})
    del model
    reloaded = PretrainedAgeModel(method)
    reloaded.load_state_dict(torch.load(path, weights_only=True)['state_dict'], strict=True)
    reloaded.eval()
    with torch.inference_mode():
        torch.testing.assert_close(expected, reloaded(x), rtol=0, atol=0)
    bad = dict(provenance, sha256='0' * 64)
    with pytest.raises(ValueError, match='pinned provenance'):
        PretrainedAgeModel(method, weights=directory / 'model.safetensors', provenance=bad)


def test_pixel_geometry_has_no_center_crop_and_agrees_for_prepared224():
    original = Image.fromarray(np.random.default_rng(7).integers(0, 256, (200, 200, 3), dtype=np.uint8))
    prepared = original.resize((224, 224), Image.Resampling.BICUBIC)
    assert np.array_equal(pixels(original), pixels(prepared))
    assert pixels(original).shape == (3, 256, 256)
    assert pixels(original).dtype == np.uint8


def test_cache_contains_only_train_validation_and_frozen_counts():
    path = ROOT / 'data/cache/pretrained_faces256/metadata.json'
    if not path.exists():
        pytest.skip('Research-only image cache is excluded from distribution.')
    from visage.data import load_manifest
    rows = load_manifest(ROOT / 'data/processed/manifest.csv')
    record = json.loads(path.read_text())
    expected = {r['sample_id'] for r in rows if r['split'] != 'test'}
    assert record['complete'] and set(record['ids']) == expected and len(expected) == 19792
    with pytest.raises(KeyError):
        FaceBatches([next(r for r in rows if r['split'] == 'test')])


def test_train_only_inverse_sqrt_weights_and_missing_class_rejection():
    weights = class_weights([0, 1, 1, 1, 1, 2])
    torch.testing.assert_close(weights, torch.tensor([1.2, .6, 1.2]))
    with pytest.raises(ValueError):
        class_weights([0, 1])


def test_accumulation_uses_effective_batch_weighted_denominator(monkeypatch):
    import visage.pretrained_training as module
    monkeypatch.setattr(module, 'prepare_training_batch', lambda x, *a, **k: x.float() / 255)
    rng = np.random.default_rng(17)
    images = torch.from_numpy(rng.integers(0, 256, (45, 3, 2, 2), dtype=np.uint8))
    labels = np.array([0] * 5 + [1] * 30 + [2] * 10)
    data = SimpleNamespace(rows=list(range(45)), labels=labels,
        batch=lambda positions, device: (images[positions].to(device), torch.tensor(labels[positions], device=device)))
    spec = SimpleNamespace(mean=(0, 0, 0), std=(1, 1, 1))
    first = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3))
    second = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3))
    second.load_state_dict(first.state_dict())
    for model, micro in ((first, 32), (second, 8)):
        train_epoch(model, data, torch.optim.SGD(model.parameters(), lr=.02), class_weights(labels), spec,
            {'effective_batch': 32, 'microbatch': micro, 'memory_format': 'contiguous', 'gradient_clip': 5},
            torch.Generator().manual_seed(7), device=torch.device('cpu'), dtype=torch.bfloat16)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_all_rng_streams_resume_and_cosine_schedule(tmp_path):
    seed_everything(43, threads=2)
    generator = torch.Generator().manual_seed(43)
    state = rng_state(generator)
    path = tmp_path / 'rng.pt'
    atomic_save(path, state)
    def draws():
        return [random.random(), float(np.random.rand()), float(torch.rand(())),
            torch.randperm(8, generator=generator).tolist()]
    expected = draws()
    restore_rng(torch.load(path, weights_only=True), generator)
    assert draws() == expected
    cfg = {'max_epochs': 24, 'backbone_lr': 1e-4, 'head_lr': 5e-4}
    assert learning_rates(1, cfg) == [1e-4, 5e-4]
    assert learning_rates(24, cfg) == [0, 0]

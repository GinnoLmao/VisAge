"""Refuse corrupted or mislabeled serving artifacts before model construction."""
import json
import torch
import pytest
from visage.model_registry import METHODS, REGISTRY
from visage.pretrained_data import GEOMETRY
from visage.runtime import sha256_file, write_json
import visage.pretrained_inference as module


def fixture(directory, **changes):
    method = METHODS[0]
    metadata = {'format_version': 2, 'method': method, 'architecture': REGISTRY[method].checkpoint,
        'input': REGISTRY[method].to_dict(), 'geometry': GEOMETRY, 'head_type': 'softmax',
        'manifest_sha256': 'a' * 64, 'protocol_sha256': 'b' * 64, 'run_id': 'example_full_seed42',
        'seed': 42, 'memory_format': 'channels_last'}
    metadata.update(changes)
    torch.save({'metadata': metadata, 'state_dict': {}}, directory / 'best.pt')
    summary = {key: metadata[key] for key in ('manifest_sha256', 'protocol_sha256', 'run_id', 'seed')}
    summary.update(method=method, artifact_sha256=sha256_file(directory / 'best.pt'))
    write_json(directory / 'training_summary.json', summary)
    return method


def test_corruption_is_rejected_before_constructing_a_model(tmp_path, monkeypatch):
    method = fixture(tmp_path)
    with (tmp_path / 'best.pt').open('ab') as stream:
        stream.write(b'corruption')
    monkeypatch.setattr(module, 'PretrainedAgeModel', lambda *a, **k: pytest.fail('Corrupt weights reached model construction'))
    with pytest.raises(ValueError, match='checkpoint differs'):
        module.PretrainedPredictor(method, tmp_path)


@pytest.mark.parametrize('changes', [
    {'architecture': REGISTRY[METHODS[1]].checkpoint},
    {'format_version': 1}, {'method': METHODS[1]},
    {'geometry': 'separate center crop'}, {'input': {'image_size': 224}},
    {'head_type': 'ordinal'}, {'seed': 43}])
def test_mislabeled_transform_head_or_family_never_loads_as_an_active_model(tmp_path, monkeypatch, changes):
    method = fixture(tmp_path, **changes)
    monkeypatch.setattr(module, 'PretrainedAgeModel', lambda *a, **k: pytest.fail('Invalid metadata reached model construction'))
    with pytest.raises(ValueError):
        module.PretrainedPredictor(method, tmp_path)

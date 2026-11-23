"""Small CPU checks for ordinal semantics and neural model integration."""

from __future__ import annotations

import io
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from visage.methods.convnext_ordinal import (
    MODEL_NAME,
    ConvNeXtOrdinal,
    set_backbone_trainable,
)
from visage.methods.ordinal import corn_loss, ordinal_probabilities, predict_age_group
from visage.methods.visagenet import VisAgeNet


def test_probabilities_form_ordered_normalized_distribution() -> None:
    logits = torch.tensor([[0.0, 0.0], [-1000.0, 1000.0], [1000.0, -1000.0]])
    probabilities = ordinal_probabilities(logits)
    torch.testing.assert_close(probabilities[0], torch.tensor([0.5, 0.25, 0.25]))
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3))
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    cumulative_first = probabilities[:, 1:].sum(dim=1)
    cumulative_second = probabilities[:, 2]
    assert torch.all(cumulative_first >= cumulative_second)
    assert predict_age_group(logits).tolist() == [0, 0, 1]


def test_corn_loss_matches_hand_calculation_and_weighted_denominator() -> None:
    conditional = torch.tensor([[0.2, 0.4], [0.7, 0.3], [0.8, 0.6]])
    logits = torch.logit(conditional)
    labels = torch.tensor([0, 1, 2])
    first_sum = -math.log(0.8) - math.log(0.7) - math.log(0.8)
    second_sum = -math.log(0.7) - math.log(0.6)
    assert corn_loss(logits, labels).item() == pytest.approx(
        (first_sum + second_sum) / 5, rel=1e-6
    )
    assert corn_loss(logits, labels, task_weights=[1, 2]).item() == pytest.approx(
        (first_sum + 2 * second_sum) / 7, rel=1e-6
    )


def test_corn_gradients_exclude_ineligible_second_task() -> None:
    logits = torch.zeros((3, 2), requires_grad=True)
    loss = corn_loss(logits, torch.tensor([0, 1, 2]))
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert logits.grad[1, 1] > 0
    assert logits.grad[2, 1] < 0
    assert torch.isfinite(logits.grad).all()


def test_unsigned_integer_labels_have_identical_conditional_masks() -> None:
    logits = torch.tensor([[-0.5, 0.3], [0.1, -1.2], [0.8, 1.3]])
    labels = torch.tensor([0, 1, 2])
    torch.testing.assert_close(corn_loss(logits, labels.to(torch.uint8)), corn_loss(logits, labels))


@pytest.mark.parametrize("class_index", [0, 1, 2])
def test_missing_classes_in_minibatch_have_finite_loss(class_index: int) -> None:
    logits = torch.zeros((2, 2), requires_grad=True)
    loss = corn_loss(logits, torch.full((2,), class_index, dtype=torch.long))
    assert loss.item() == pytest.approx(math.log(2), rel=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    if class_index == 0:
        torch.testing.assert_close(logits.grad[:, 1], torch.zeros(2))


def test_empty_batch_and_absent_weighted_task_return_differentiable_zero() -> None:
    empty = torch.empty((0, 2), requires_grad=True)
    zero = corn_loss(empty, torch.empty((0,), dtype=torch.long))
    assert zero.ndim == 0 and zero.item() == 0
    zero.backward()
    logits = torch.zeros((2, 2), requires_grad=True)
    loss = corn_loss(logits, torch.zeros(2, dtype=torch.long), task_weights=[0, 1])
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_large_logits_have_stable_loss_and_gradients() -> None:
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
    loss = corn_loss(logits, torch.tensor([0, 2]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_ordinal_input_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        ordinal_probabilities(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="0, 1, or 2"):
        corn_loss(torch.zeros(2, 2), torch.tensor([0, 3]))
    with pytest.raises(TypeError, match="integer"):
        corn_loss(torch.zeros(2, 2), torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        corn_loss(torch.zeros(2, 2), torch.tensor([0, 1]), task_weights=[0, 0])


def test_visagenet_forward_backward_and_parameter_budget() -> None:
    model = VisAgeNet()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 1_000_000 <= count <= 3_000_000
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 2)
    corn_loss(logits, torch.tensor([0, 2])).backward()
    assert model.stem[0].weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_visagenet_seed_and_checkpoint_roundtrip() -> None:
    torch.manual_seed(37)
    first = VisAgeNet().eval()
    torch.manual_seed(37)
    second = VisAgeNet().eval()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    inputs = torch.randn(1, 3, 64, 64)
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    restored = VisAgeNet().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(first(inputs), restored(inputs), rtol=0, atol=0)


def test_visagenet_ablation_preserves_three_class_task() -> None:
    model = VisAgeNet(head_type="softmax", use_se=False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 3)


def test_convnext_freezing_and_new_head_without_download(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyBackbone(nn.Module):
        num_features = 8

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=1), nn.AdaptiveAvgPool2d(1)
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.features(inputs).flatten(1)

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_create_model(name: str, **kwargs: object) -> TinyBackbone:
        calls.append((name, kwargs))
        return TinyBackbone()

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=fake_create_model))
    model = ConvNeXtOrdinal(pretrained=False, freeze_backbone=True)
    assert calls == [(MODEL_NAME, {"pretrained": False, "num_classes": 0, "global_pool": "avg"})]
    model.train()
    assert model.training and model.head.training and not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    logits = model(torch.randn(2, 3, 16, 16))
    corn_loss(logits, torch.tensor([1, 2])).backward()
    assert model.head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    set_backbone_trainable(model, True)
    assert model.backbone.training
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    model.eval()
    set_backbone_trainable(model, True)
    assert not model.backbone.training


def test_real_convnext_offline_forward() -> None:
    pytest.importorskip("timm")
    model = ConvNeXtOrdinal(pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_local_pretrained_artifact_loads_only_matching_backbone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")

    class TinyBackbone(nn.Module):
        num_features = 8

        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(3, 8, kernel_size=1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.projection(inputs).mean(dim=(2, 3))

    calls: list[dict[str, object]] = []

    def fake_create_model(name: str, **kwargs: object) -> TinyBackbone:
        assert name == MODEL_NAME
        calls.append(kwargs)
        return TinyBackbone()

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=fake_create_model))
    expected = TinyBackbone().state_dict()
    artifact = dict(expected)
    artifact["head.fc.weight"] = torch.randn(1000, 8)
    artifact["head.fc.bias"] = torch.randn(1000)
    path = tmp_path / "pinned.safetensors"
    safetensors.save_file(artifact, str(path))
    model = ConvNeXtOrdinal(pretrained=True, pretrained_path=path)
    assert calls[-1]["pretrained"] is False
    assert model.pretrained_requested is True
    assert model.pretrained_path == str(path.resolve())
    assert model.head.out_features == 2
    for key, value in expected.items():
        torch.testing.assert_close(model.backbone.state_dict()[key], value)

    # Only the original head keys are discardable; other mismatches fail.
    artifact["unrecognized.weight"] = torch.ones(1)
    safetensors.save_file(artifact, str(path))
    with pytest.raises(RuntimeError, match="unexpected=.*unrecognized.weight"):
        ConvNeXtOrdinal(pretrained=True, pretrained_path=path)
    del artifact["unrecognized.weight"]
    del artifact["projection.bias"]
    safetensors.save_file(artifact, str(path))
    with pytest.raises(RuntimeError, match="missing=.*projection.bias"):
        ConvNeXtOrdinal(pretrained=True, pretrained_path=path)


def test_local_pretrained_path_cannot_be_silently_ignored() -> None:
    with pytest.raises(ValueError, match="requires pretrained=True"):
        ConvNeXtOrdinal(pretrained=False, pretrained_path="unused.safetensors")

"""Meaningful research admission, sampling, warm-up, and promotion checks."""

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from visage.research import nested_training_subsets, promotion_decision, write_immutable_json
from visage.training import backbone_warmup_factor, sampling_weights, select_training_subset


def _rows():
    rows = []
    for label in range(3):
        for group in range(12):
            for member in range(2):
                rows.append({"sample_id": f"{label}-{group}-{member}", "group_id": f"{label}-{group}", "split": "train", "label": label})
    rows.append({"sample_id": "heldout", "group_id": "heldout", "split": "test", "label": 0})
    return rows


def test_sampling_exponent_preserves_default_and_moderates_class_sampling():
    labels = np.array([0] * 2 + [1] * 8 + [2] * 2)
    original = torch.as_tensor(1.0 / np.bincount(labels)[labels], dtype=torch.double)
    torch.testing.assert_close(sampling_weights(labels), original, rtol=0, atol=0)
    half = sampling_weights(labels, 0.5)
    masses = [float(half[labels == label].sum()) for label in range(3)]
    assert masses[1] / masses[0] == pytest.approx(2.0)
    torch.testing.assert_close(sampling_weights(labels, 0), torch.ones(len(labels), dtype=torch.double))
    with pytest.raises(ValueError):
        sampling_weights(labels, float("nan"))


def test_nested_subsets_keep_groups_classes_and_original_split(tmp_path):
    rows = _rows()
    results = nested_training_subsets(rows, "a" * 64)
    assert results == nested_training_subsets(rows, "a" * 64)
    assert set(results[0.25]["sample_ids"]) < set(results[0.5]["sample_ids"])
    for subset in results.values():
        path = tmp_path / "subset.json"
        path.write_text(json.dumps(subset))
        selected = select_training_subset(rows, path, "a" * 64)
        assert len(selected) == subset["rows"]
        assert {row["label"] for row in selected} == {0, 1, 2}
        assert "heldout" not in subset["sample_ids"]
        assert all(sum(row["group_id"] == group for row in selected) == 2 for group in subset["group_ids"])


@pytest.mark.parametrize("case", ["test", "partial_group", "duplicate", "manifest"])
def test_training_subset_rejects_leakage_and_changed_identity(tmp_path, case):
    rows = _rows()
    value = nested_training_subsets(rows, "a" * 64)[0.25]
    if case == "test": value["sample_ids"].append("heldout")
    elif case == "partial_group": value["sample_ids"].pop()
    elif case == "duplicate": value["sample_ids"].append(value["sample_ids"][0])
    else: value["manifest_sha256"] = "b" * 64
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        select_training_subset(rows, path, "a" * 64)


def test_unfreeze_warmup_only_ramps_newly_active_epochs():
    assert [backbone_warmup_factor(epoch, 2, 0) for epoch in range(1, 6)] == [1] * 5
    assert [backbone_warmup_factor(epoch, 2, 2) for epoch in range(1, 6)] == [1, 1, 0.5, 1, 1]


def test_promotion_requires_mean_improvement_and_boundary_nonregression():
    def metrics(f1, minor=0.12, senior=0.78):
        return {"macro_f1": f1, "under18_predicted_older_rate": minor, "per_class": [{}, {}, {"recall": senior}]}
    baseline = [metrics(0.85)] * 3
    assert promotion_decision(baseline, [metrics(0.86)] * 3)["promote"]
    assert not promotion_decision(baseline, [metrics(0.853)] * 3)["promote"]
    assert not promotion_decision(baseline, [metrics(0.87, minor=0.14)] * 3)["promote"]
    assert not promotion_decision(baseline, [metrics(0.87, senior=0.75)] * 3)["promote"]
    assert not promotion_decision(baseline, [metrics(0.87)])["promote"]


def test_predeclared_record_cannot_be_silently_overwritten(tmp_path):
    path = tmp_path / "protocol.json"
    write_immutable_json(path, {"seed": 42})
    write_immutable_json(path, {"seed": 42})
    with pytest.raises(ValueError, match="would change"):
        write_immutable_json(path, {"seed": 43})


def test_classical_data_efficiency_fits_only_selected_training_rows(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from visage.methods.classical import FeatureConfig, LinearSVM
    from visage.runtime import sha256_file, write_json
    path = Path(__file__).resolve().parents[1] / "scripts/research_classical.py"
    spec = importlib.util.spec_from_file_location("classical_research_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "STUDY", tmp_path / "reports/research")
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path / "artifacts/research")
    config = FeatureConfig(image_size=16, lbp_grid=2, lbp_radii=(1,))
    rows = []
    for split, count in (("train", 9), ("validation", 3), ("test", 3)):
        for index in range(count):
            identity = f"{len(rows):064x}"
            rows.append({"sample_id": identity, "group_id": identity, "split": split,
                         "label": index % 3, "relative_path": identity + ".jpg"})
    features = np.random.default_rng(3).normal(size=(len(rows), config.feature_dimension)).astype(np.float32)
    directory = tmp_path / "feature_cache"
    directory.mkdir()
    np.save(directory / "features.npy", features)
    write_json(directory / "features.json", {"manifest_sha256": "a" * 64,
               "features_sha256": sha256_file(directory / "features.npy"),
               "feature_config": config.to_dict(), "extraction_seconds": 0.0})
    baseline = tmp_path / "artifacts/classical"
    write_json(baseline / "training_config.json", {"cache_directory": str(directory)})
    write_json(baseline / "selection.json", {"feature_config": config.to_dict(),
               "selected_configuration": LinearSVM(epochs=2, batch_size=3)._configuration()})
    protocol = {"manifest_sha256": "a" * 64, "protected_original_files": {}}
    write_json(module.STUDY / "protocol.json", protocol)
    subset = nested_training_subsets(rows, "a" * 64)[0.25]
    write_json(module.STUDY / "subsets/train_25.json", subset)
    selected_indices = [i for i, row in enumerate(rows) if row["sample_id"] in subset["sample_ids"]]
    original_fit = LinearSVM.fit

    def fit_only_selected(self, values, labels):
        np.testing.assert_array_equal(values, features[selected_indices])
        assert len(labels) == len(selected_indices)
        return original_fit(self, values, labels)

    monkeypatch.setattr(LinearSVM, "fit", fit_only_selected)
    helpers = SimpleNamespace(_split_features=lambda values, indices, *args: values[indices])
    module.run("classical_data25_seed42", "combined", 25, rows, protocol, helpers)
    selection = json.loads((module.ARTIFACTS / "classical_data25_seed42/selection.json").read_text())
    assert selection["training_rows"] == len(selected_indices)
    assert selection["validation_rows"] == 3 and selection["test_evaluated"] is False

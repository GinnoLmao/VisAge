"""Tiny CPU integration checks for frozen data and reloadable training artifacts."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from visage.data import MANIFEST_COLUMNS, load_manifest, prepare_dataset
from visage.datasets import ManifestDataset, make_transform
from visage.training import evaluate_model, load_neural_artifact, train_neural
import visage.training as training_module


@pytest.fixture
def frozen_training_data(tmp_path: Path) -> tuple[dict, list[dict]]:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(18):
        age = (12, 35, 70)[index % 3]
        pixels = np.random.default_rng(index + 100).integers(
            0, 256, size=(40, 40, 3), dtype=np.uint8
        )
        filename = f"{age}_0_0_{20260101000000000 + index}.jpg.chip.jpg"
        Image.fromarray(pixels).save(raw / filename, format="JPEG", quality=95)
    prepared = tmp_path / "prepared"
    prepare_dataset(raw, prepared, seed=19, phash_distance=0)
    manifest = prepared / "manifest.csv"
    rows = load_manifest(manifest)
    assert {row["label"] for row in rows if row["split"] == "train"} == {0, 1, 2}
    assert all(any(row["split"] == split for row in rows) for split in ("train", "validation", "test"))
    config = {
        "method": "visagenet",
        "manifest": str(manifest),
        "raw_dir": str(raw),
        "output_dir": str(tmp_path / "run"),
        "device": "cpu",
        "seed": 19,
        "threads": 1,
        "workers": 0,
        "batch_size": 6,
        "epochs": 2,
        "min_epochs": 2,
        "patience": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "model_config": {
            "widths": [8, 16, 32, 64],
            "blocks_per_stage": [1, 1, 1, 1],
            "dropout": 0.0,
        },
    }
    return config, rows


def test_tiny_training_never_decodes_test_and_reloads_selected_predictions(
    frozen_training_data: tuple[dict, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, rows = frozen_training_data
    accessed_splits: set[str] = set()
    original_getitem = ManifestDataset.__getitem__
    original_image_open = Image.open
    test_paths = {
        (Path(config["raw_dir"]) / row["relative_path"]).resolve()
        for row in rows if row["split"] == "test"
    }

    def guarded_getitem(dataset: ManifestDataset, index: int):
        split = dataset.rows[index]["split"]
        accessed_splits.add(split)
        assert split != "test", "Held-out test data was used during training."
        return original_getitem(dataset, index)

    def guarded_image_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            assert Path(path).resolve() not in test_paths, "Held-out test pixels were decoded during training."
        return original_image_open(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(ManifestDataset, "__getitem__", guarded_getitem)
        context.setattr(Image, "open", guarded_image_open)
        summary = train_neural(config)
    assert accessed_splits == {"train", "validation"}
    assert summary["epochs_completed"] == 2
    assert summary["best_epoch"] in (1, 2)
    output = Path(config["output_dir"])
    for name in (
        "best.pt", "last.pt", "config.json", "environment.json", "history.json",
        "validation_predictions.csv", "validation_metrics.json", "training_summary.json",
    ):
        assert (output / name).is_file(), name
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert all(np.isfinite(record["train_loss"]) for record in history)

    model, checkpoint = load_neural_artifact(output / "best.pt", device="cpu")
    assert checkpoint["epoch"] == summary["best_epoch"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    transform = checkpoint["transform"]
    dataset = ManifestDataset(
        validation_rows,
        config["raw_dir"],
        make_transform(
            image_size=transform["image_size"], mean=transform["mean"], std=transform["std"]
        ),
    )
    result = evaluate_model(
        model, DataLoader(dataset, batch_size=config["batch_size"], num_workers=0), torch.device("cpu")
    )
    with (output / "validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert [row["sample_id"] for row in saved] == [row["sample_id"] for row in validation_rows]
    assert [int(row["prediction"]) for row in saved] == result["predictions"]
    np.testing.assert_allclose(
        [[float(row[f"score_{label}"]) for label in range(3)] for row in saved],
        result["values"],
        rtol=0,
        atol=1e-7,
    )
    assert result["metrics"] == checkpoint["validation_metrics"]
    verifier_path = Path(__file__).resolve().parents[1] / "scripts/check_artifacts.py"
    spec = importlib.util.spec_from_file_location("visage_artifact_verifier", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    checked = verifier.check_neural(
        "visagenet", output, Path(config["raw_dir"]), validation_rows,
        np.arange(len(validation_rows)), summary["manifest_sha256"],
    )
    assert checked["ok"] and checked["validation_rows_checked"] == len(validation_rows)
    assert checked["maximum_score_absolute_error"] <= 1e-7
    with pytest.raises(FileExistsError, match="complete"):
        train_neural(config)


def test_training_rejects_changed_frozen_training_pixels(
    frozen_training_data: tuple[dict, list[dict]]
) -> None:
    config, rows = frozen_training_data
    row = next(row for row in rows if row["split"] == "train")
    path = Path(config["raw_dir"]) / row["relative_path"]
    Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8)).save(path, format="JPEG")
    with pytest.raises(ValueError, match="(?i)(frozen|checksum|hash|changed|integrity)"):
        train_neural(config)
    assert not (Path(config["output_dir"]) / "best.pt").exists()


def test_training_rejects_modified_freeze_metadata(
    frozen_training_data: tuple[dict, list[dict]]
) -> None:
    config, _ = frozen_training_data
    audit = Path(config["manifest"]).parent / "audit.json"
    audit.write_text(audit.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        train_neural(config)


def test_training_rejects_unknown_split_before_opening_images(
    frozen_training_data: tuple[dict, list[dict]]
) -> None:
    config, rows = frozen_training_data
    rows[0]["split"] = "development"
    with Path(config["manifest"]).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Invalid split"):
        train_neural(config)


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_training_requires_every_class_in_each_selection_partition(
    frozen_training_data: tuple[dict, list[dict]], monkeypatch: pytest.MonkeyPatch,
    partition: str,
) -> None:
    config, rows = frozen_training_data
    incomplete = [row for row in rows if not (row["split"] == partition and row["label"] == 2)]
    # The frozen fixture's integrity checks still run; this isolates the
    # subsequent classifier-specific all-class partition admission guard.
    monkeypatch.setattr(training_module, "load_manifest", lambda path: incomplete)
    with pytest.raises(ValueError, match="each contain all three classes"):
        train_neural(config)
    assert not (Path(config["output_dir"]) / "best.pt").exists()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [("epochs", 0, "epochs"), ("batch_size", 0, "batch_size"),
     ("workers", -1, "workers"), ("learning_rate", 0, "learning_rate")],
)
def test_invalid_training_configuration_fails_before_creating_output(
    frozen_training_data: tuple[dict, list[dict]], key: str, value: int, message: str
) -> None:
    config, _ = frozen_training_data
    with pytest.raises(ValueError, match=message):
        train_neural({**config, key: value})
    assert not Path(config["output_dir"]).exists()


def test_cpu_resume_restores_weights_rng_and_rejects_configuration_changes(
    frozen_training_data: tuple[dict, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = frozen_training_data
    train_neural(config)
    uninterrupted = torch.load(
        Path(config["output_dir"]) / "last.pt", map_location="cpu", weights_only=True
    )
    resumed_config = {**config, "output_dir": str(Path(config["output_dir"]).with_name("resumed"))}
    original_save = training_module.atomic_torch_save

    def save_then_interrupt(path, checkpoint):
        original_save(path, checkpoint)
        if Path(path).name == "last.pt" and checkpoint["epoch"] == 1:
            raise InterruptedError("Simulated interruption after complete epoch checkpoint.")

    with monkeypatch.context() as context:
        context.setattr(training_module, "atomic_torch_save", save_then_interrupt)
        with pytest.raises(InterruptedError, match="Simulated interruption"):
            train_neural(resumed_config)
    output = Path(resumed_config["output_dir"])
    original_config = (output / "config.json").read_bytes()
    with pytest.raises(ValueError, match="Resume configuration changed"):
        train_neural({**resumed_config, "learning_rate": 0.002}, resume=True)
    assert (output / "config.json").read_bytes() == original_config
    resumed_summary = train_neural(resumed_config, resume=True)
    resumed = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert resumed_summary["epochs_completed"] == 2
    for key, expected in uninterrupted["model_state"].items():
        torch.testing.assert_close(resumed["model_state"][key], expected, rtol=0, atol=0)
    for key in ("generator_state", "torch_rng"):
        torch.testing.assert_close(resumed[key], uninterrupted[key], rtol=0, atol=0)


def test_cached_uint8_backend_preserves_validation_normalization(
    frozen_training_data: tuple[dict, list[dict]]
) -> None:
    config, rows = frozen_training_data
    config = {
        **config, "epochs": 1, "min_epochs": 1,
        "augmentation_backend": "tensor_batch_v1",
        "tensor_cache_dir": str(Path(config["output_dir"]).with_name("tensor_cache")),
    }
    summary = train_neural(config)
    output = Path(config["output_dir"])
    model, checkpoint = load_neural_artifact(output / "best.pt", "cpu")
    validation_rows = [row for row in rows if row["split"] == "validation"]
    transform = checkpoint["transform"]
    assert transform["train_augmentation"]["name"] == "visage_tensor_batch_v1"
    # Replay through the original deterministic PIL normalization path,
    # independently of the uint8 cache and batch-normalization implementation.
    dataset = ManifestDataset(
        validation_rows, config["raw_dir"],
        make_transform(image_size=transform["image_size"], mean=transform["mean"], std=transform["std"]),
    )
    result = evaluate_model(
        model, DataLoader(dataset, batch_size=config["batch_size"], num_workers=0), torch.device("cpu")
    )
    with (output / "validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    np.testing.assert_allclose(
        [[float(row[f"score_{label}"]) for label in range(3)] for row in saved],
        result["values"], rtol=0, atol=1e-7,
    )
    assert [int(row["prediction"]) for row in saved] == result["predictions"]
    assert summary["epochs_completed"] == 1


def test_training_subset_and_sampling_are_recorded_without_changing_validation(
    frozen_training_data: tuple[dict, list[dict]]
) -> None:
    from visage.research import nested_training_subsets
    from visage.runtime import sha256_file
    config, rows = frozen_training_data
    subset = nested_training_subsets(rows, sha256_file(Path(config["manifest"])))[0.5]
    path = Path(config["output_dir"]).with_name("subset.json")
    path.write_text(json.dumps(subset), encoding="utf-8")
    summary = train_neural({**config, "epochs": 1, "min_epochs": 1,
                            "train_subset_path": str(path), "sampler_exponent": 0.5})
    assert summary["training_rows"] == len(subset["sample_ids"])
    assert summary["validation_rows"] == sum(row["split"] == "validation" for row in rows)
    recorded = json.loads((Path(config["output_dir"]) / "config.json").read_text())
    assert recorded["train_subset_sha256"] == sha256_file(path)
    assert recorded["sampler_exponent"] == 0.5

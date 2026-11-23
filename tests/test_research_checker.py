"""Independent-verifier arithmetic and evidence-tampering rejection checks."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("check_research", Path(__file__).resolve().parents[1] / "scripts/check_research.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_independent_metrics_match_hand_computed_confusion_and_boundary_rates():
    result = checker.independent_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2])
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 0.8) / 3)
    assert result["under18_predicted_older_rate"] == 0.5
    assert result["senior_predicted_below60_rate"] == 0
    assert result["non_senior_predicted_senior_rate"] == 0.25
    assert result["per_class"][2]["precision"] == pytest.approx(2 / 3)


def test_missing_class_denominators_are_explicitly_zero_and_nonfinite_evidence_fails():
    result = checker.independent_metrics([1], [1])
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["per_class"][0]["recall"] == 0
    with pytest.raises(ValueError):
        checker.close_values({"value": float("nan")}, {"value": 0.5})


def example_rows():
    frozen = [{"sample_id": f"sample-{i}", "relative_path": f"{i}.jpg", "age": str(5 + i * 30), "label": str(i)} for i in range(3)]
    saved = []
    for i, row in enumerate(frozen):
        scores = ["0.1"] * 3
        scores[i] = "0.8"
        saved.append({**row, "prediction": str(i), **{f"score_{k}": value for k, value in enumerate(scores)}})
    return frozen, saved


@pytest.mark.parametrize("change", ["order", "label", "path", "age", "nonfinite", "argmax", "not_normalized", "missing_row"])
def test_saved_validation_prediction_tampering_is_rejected(change):
    frozen, saved = example_rows()
    assert checker.check_prediction_rows(saved, frozen, score_prefix="score")["accuracy"] == 1
    if change == "order": saved[0], saved[1] = saved[1], saved[0]
    elif change == "label": saved[0]["label"] = "1"
    elif change == "path": saved[0]["relative_path"] = "different.jpg"
    elif change == "age": saved[0]["age"] = "18"
    elif change == "nonfinite": saved[0]["score_0"] = "nan"
    elif change == "argmax": saved[0]["prediction"] = "1"
    elif change == "not_normalized": saved[0]["score_0"] = "0.4"
    else: saved.pop()
    with pytest.raises(ValueError):
        checker.check_prediction_rows(saved, frozen, score_prefix="score")


def subset_fixture(tmp_path):
    rows = [{"sample_id": f"{label}-{group}-{member}", "label": str(label), "group_id": f"{label}-{group}", "split": "train"}
            for label in range(3) for group in range(4) for member in range(2)]
    rows.append({"sample_id": "test-only", "label": "0", "group_id": "external", "split": "test"})
    protocol = {"manifest_sha256": "a" * 64, "subsets": {}}
    for percent, groups_per_label in (("25", 1), ("50", 2)):
        selected = [row for row in rows if row["split"] == "train" and int(row["group_id"].split("-")[1]) < groups_per_label]
        record = {"split": "train", "manifest_sha256": "a" * 64, "sample_ids": [row["sample_id"] for row in selected],
                  "group_ids": sorted({row["group_id"] for row in selected}), "rows": len(selected),
                  "fraction_actual": len(selected) / 24, "class_counts": [len(selected) // 3] * 3}
        relative = f"subsets/{percent}.json"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record))
        protocol["subsets"][percent] = {"path": relative, "sha256": checker.sha256_file(path),
                                        **{key: record[key] for key in ("rows", "fraction_actual", "class_counts")}}
    return rows, protocol


def test_subset_checker_independently_requires_train_only_ordered_nested_whole_groups(tmp_path):
    rows, protocol = subset_fixture(tmp_path)
    checked = checker.check_subsets(tmp_path, protocol, rows)
    assert checked["25"]["rows"] == 6 and checked["50"]["rows"] == 12
    path = tmp_path / protocol["subsets"]["25"]["path"]
    value = json.loads(path.read_text())
    value["sample_ids"].pop()  # one member of a two-image group
    path.write_text(json.dumps(value))
    protocol["subsets"]["25"]["sha256"] = checker.sha256_file(path)
    with pytest.raises(ValueError, match="perceptual group"):
        checker.check_subsets(tmp_path, protocol, rows)


def test_subset_checker_rejects_test_ids_even_if_record_checksum_is_updated(tmp_path):
    rows, protocol = subset_fixture(tmp_path)
    path = tmp_path / protocol["subsets"]["25"]["path"]
    value = json.loads(path.read_text())
    value["sample_ids"].append("test-only")
    path.write_text(json.dumps(value))
    protocol["subsets"]["25"]["sha256"] = checker.sha256_file(path)
    with pytest.raises(ValueError, match="non-training"):
        checker.check_subsets(tmp_path, protocol, rows)


def test_expected_run_contract_is_eleven_neural_four_classical_for_either_candidate():
    for candidate in checker.SCREENS:
        neural, classical = checker.expected_runs(candidate)
        assert len(neural) == len(set(neural)) == 11
        assert len(classical) == len(set(classical)) == 4
        assert candidate.replace("seed42", "seed43") in neural
        assert "convnext_ordinal_data25_seed42" in neural


def test_independent_age_and_quality_strata_count_all_images_and_errors():
    rows = [{"label": "0"}, {"label": "1"}, {"label": "2"}]
    saved = [{"prediction": "1"}, {"prediction": "1"}, {"prediction": "2"}]
    cohorts = [{"age_band": "young", "brightness_band": "dim", "sharpness_band": "low"},
               {"age_band": "adult", "brightness_band": "dim", "sharpness_band": "high"},
               {"age_band": "senior", "brightness_band": "bright", "sharpness_band": "high"}]
    result = checker.independent_strata(rows, saved, cohorts)
    assert result["brightness_band"]["dim"] == {"samples": 2, "correct": 1, "errors": 1, "accuracy": 0.5}
    assert result["sharpness_band"]["high"]["accuracy"] == 1
    assert sum(entry["samples"] for entry in result["age_band"].values()) == 3
    with pytest.raises(ValueError, match="every validation"):
        checker.independent_strata(rows, saved, cohorts[:-1])


def test_independent_promotion_rejects_senior_regression_despite_higher_macro_f1(tmp_path):
    candidate = "visagenet_sampler05_seed42"
    baseline = {"macro_f1": 0.85, "under18_predicted_older_rate": 0.1, "per_class": [{}, {}, {"recall": 0.8}]}
    challenger = deepcopy(baseline)
    challenger.update(macro_f1=0.87)
    challenger["per_class"][2]["recall"] = 0.75
    paths = [(tmp_path / "artifacts/visagenet/validation_metrics.json", baseline)]
    for seed in (42, 43, 44):
        paths.append((tmp_path / "artifacts/research" / candidate.replace("seed42", f"seed{seed}") / "validation_metrics.json", challenger))
        if seed != 42:
            paths.append((tmp_path / f"artifacts/research/visagenet_baseline_seed{seed}/validation_metrics.json", baseline))
    for path, record in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
    protocol = {"promotion": {"mean_macro_f1_delta_min": 0.005, "minor_error_mean_increase_max": 0.01, "senior_recall_mean_drop_max": 0.01}}
    result = checker.independent_promotion(tmp_path, protocol, candidate)
    assert not result["promote"]
    assert result["checks"]["mean_macro_f1_improvement_at_least_0_005"]
    assert not result["checks"]["senior_recall_drop_at_most_0_01"]

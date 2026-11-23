"""Integration diagnostic denominators and paired-subset comparison accounting."""

import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/evaluate_face_pipeline.py"
spec = importlib.util.spec_from_file_location("face_pipeline_evaluation", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def outcome(label, accepted, prediction=0, baseline=0):
    return {"label": label, "status": "accepted" if accepted else "rejected",
            "error_code": "" if accepted else "NO_FACE", "rejection_reasons": "[]",
            **{m: prediction if accepted else None for m in module.METHODS},
            **{f"baseline_{m}": baseline for m in module.METHODS}}


def test_relative_diagnostic_paths_are_normalized_before_recording(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report, predictions = module.output_paths(Path('reports/result.json'), Path('reports/predictions.csv'), root=tmp_path)
    assert report == tmp_path/'reports/result.json'
    assert predictions.relative_to(tmp_path).as_posix() == 'reports/predictions.csv'
    with pytest.raises(ValueError, match='distinct'):
        module.output_paths(report, report, root=tmp_path)
    with pytest.raises(ValueError, match='workspace'):
        module.output_paths(tmp_path.parent/'elsewhere.json', predictions, root=tmp_path)


def test_rejections_remain_in_end_to_end_denominator_and_paired_comparison():
    rows = [outcome(0, True, 0, 1), outcome(1, False, baseline=1), outcome(2, False, baseline=2)]
    result = module.summarize_rows(rows)
    assert result["coverage"] == pytest.approx(1 / 3)
    assert result["rejection_codes"] == {"NO_FACE": 2}
    assert result["by_class"]["0"]["coverage"] == 1
    for values in result["methods"].values():
        assert values["accepted_image_metrics"]["accuracy"] == 1
        assert values["end_to_end_correct_rate"] == pytest.approx(1 / 3)
        assert values["original_aligned_metrics_same_accepted_images"]["accuracy"] == 0
        assert values["original_aligned_metrics_all_images"]["accuracy"] == pytest.approx(2 / 3)
        assert values["prediction_changes_on_accepted"] == 1


def test_all_rejected_has_zero_correct_rate_and_undefined_conditional_metrics():
    rows = [outcome(0, False), outcome(1, False, baseline=1), outcome(2, False, baseline=2)]
    rows[0]["rejection_reasons"] = json.dumps(["blur", "lighting"])
    result = module.summarize_rows(rows)
    assert result["coverage"] == 0 and result["rejection_reasons"] == {"blur": 1, "lighting": 1}
    for values in result["methods"].values():
        assert values["end_to_end_correct_rate"] == 0
        assert values["accepted_image_metrics"] is None
        assert values["macro_f1_delta_on_identical_accepted_subset"] is None


def test_invalid_labels_or_status_do_not_become_missing_outcomes():
    with pytest.raises(ValueError):
        module.summarize_rows([])
    row = outcome(0, True)
    row["status"] = "silently_dropped"
    with pytest.raises(ValueError):
        module.summarize_rows([row])
    row = outcome(0, True)
    row["classical"] = 8
    with pytest.raises(ValueError):
        module.summarize_rows([row])

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from visage.evaluation_registry import compact_curve, evaluation_snapshot, load_compact_record, classification_details
from visage.runtime import sha256_file


def test_display_curve_keeps_endpoints_without_changing_auc():
    curve = {"fpr": list(range(1001)), "tpr": list(range(1001)), "auc": .7}
    result = compact_curve(curve, "fpr", "tpr", limit=5)
    assert result["fpr"] == [0, 250, 500, 750, 1000]
    assert curve["auc"] == .7 and result["full_points"] == 1001


def test_pending_evaluation_is_explicit_without_fabricated_values(tmp_path):
    result = evaluation_snapshot(tmp_path, SimpleNamespace())
    assert result["classification"] == {"status": "pending"}
    assert result["detection"] == {"status": "pending"}


def test_aggregate_cache_is_compact_reused_and_invalidates_on_file_change(tmp_path):
    path=tmp_path/'detection.json'
    def save(value):
        path.write_text(json.dumps({'metrics':{'AP50':{'ap_percent':value,'iou_threshold':.5,'ground_truth_faces':10,
            'recall':list(range(2000)),'precision':list(range(2000)),'thresholds':list(range(2000))}}}))
    save(5)
    first=load_compact_record(path,'detection')
    assert first is load_compact_record(path,'detection')
    assert len(first['metrics']['AP50']['recall'])==501
    assert first['metrics']['AP50']['full_points']==2000 and 'thresholds' not in first['metrics']['AP50']
    save(99)
    changed=load_compact_record(path,'detection')
    assert changed is not first and changed['metrics']['AP50']['ap_percent']==99


def test_served_model_hash_and_freeze_binding(tmp_path):
    artifact = tmp_path / "classical"
    artifact.mkdir()
    (artifact / "svm.npz").write_bytes(b"model bytes")
    service = SimpleNamespace(manifest_sha256="a" * 64, predictors={"classical": SimpleNamespace(directory=artifact)})
    curve = {"auc": .5, "fpr": [0, 1], "tpr": [0, 1]}
    record = {"ok": True, "manifest_sha256": service.manifest_sha256, "task": "age_group_classification",
              "split": "validation", "samples": 3, "methods": {"classical": {
                  "artifact_sha256": sha256_file(artifact / "svm.npz"), "accuracy_percent": 70,
                  "score_type": "margins", "roc": {"macro_auc_mean": .5, "per_class": {str(i): curve for i in range(3)}}}}}
    path = tmp_path / "classification_roc.json"
    path.write_text(json.dumps(record))
    assert evaluation_snapshot(tmp_path, service)["classification"]["methods"]["classical"]["macro_auc"] == .5
    (artifact / "svm.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        evaluation_snapshot(tmp_path, service)
    service.manifest_sha256 = "b" * 64
    with pytest.raises(ValueError, match="freeze"):
        evaluation_snapshot(tmp_path, service)


def test_annotation_parser_handles_invalid_and_zero_face_rows(tmp_path):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_extended_metrics.py"
    spec = importlib.util.spec_from_file_location("extended_evaluation_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "annotations.txt"
    path.write_text("event/one.jpg\n2\n1 2 3 4 0 0 0 0 0 0\n5 6 7 8 0 0 0 1 0 0\nevent/empty.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n")
    result = module.parse_wider(path)
    assert result["event/one.jpg"] == {"boxes": [[1, 2, 3, 4]], "ignore_boxes": [[5, 6, 7, 8]]}
    assert result["event/empty.jpg"]["boxes"] == []
    path.write_text("../unsafe.jpg\n0\n")
    with pytest.raises(ValueError, match="path"):
        module.parse_wider(path)


def test_dashboard_details_derive_f1_and_matrix_from_checksum_bound_predictions(tmp_path):
    path = tmp_path / "validation_predictions.csv"
    path.write_text("sample_id,label,prediction\na,0,0\nb,0,1\nc,1,1\nd,1,1\ne,2,2\nf,2,0\n")
    digest = sha256_file(path)
    result = classification_details(tmp_path, digest, 6)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert result["row_normalized_confusion_matrix"] == [[.5, .5, 0], [0, 1, 0], [.5, 0, .5]]
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx((.5 + .8 + 2 / 3) / 3)
    assert classification_details(tmp_path, digest, 6) is result
    path.write_text(path.read_text() + "g,2,1\n")
    with pytest.raises(ValueError, match="changed"):
        classification_details(tmp_path, digest, 6)


def test_dashboard_details_refuse_duplicate_sample_ids_and_invalid_classes(tmp_path):
    path = tmp_path / "validation_predictions.csv"
    path.write_text("sample_id,label,prediction\na,0,0\na,1,1\nb,2,2\n")
    with pytest.raises(ValueError, match="unique"):
        classification_details(tmp_path, sha256_file(path), 3)
    path.write_text("sample_id,label,prediction\na,0,0\nb,1,1\nc,2,9\n")
    with pytest.raises(ValueError):
        classification_details(tmp_path, sha256_file(path), 3)


def test_dashboard_details_are_explicitly_missing_without_saved_predictions(tmp_path):
    assert classification_details(tmp_path, "unknown", 3) is None


def test_detection_cache_preserves_a_bounded_froc_curve(tmp_path):
    path = tmp_path / "detection.json"
    path.write_text(json.dumps({"metrics": {"AP50": {"ap_percent": 64, "iou_threshold": .5,
        "ground_truth_faces": 10, "recall": [i / 1999 for i in range(2000)],
        "precision": [1] * 2000, "false_positives_per_image": list(range(2000))}}}))
    value = load_compact_record(path, "detection")["metrics"]["AP50"]
    assert len(value["froc"]["recall"]) == 501
    assert value["froc"]["full_points"] == 2000
    assert value["froc"]["false_positives_per_image"][0] == 0
    assert value["froc"]["false_positives_per_image"][-1] == 1999
    compact = load_compact_record(path, "detection")
    restored = tmp_path / "compact_detection.json"
    restored.write_text(json.dumps(compact))
    again = load_compact_record(restored, "detection")["metrics"]["AP50"]
    assert again["froc"] == value["froc"] and again["ap_percent"] == value["ap_percent"]


def test_replaced_disk_artifacts_cannot_relabel_metrics_for_a_loaded_model(tmp_path):
    method = "convnextv2_nano"
    directory = tmp_path / method
    directory.mkdir()
    path = directory / "best.pt"
    path.write_bytes(b"original loaded checkpoint")
    loaded_hash = sha256_file(path)
    predictor = SimpleNamespace(directory=directory, artifact_sha256=loaded_hash)
    service = SimpleNamespace(manifest_sha256="a" * 64, predictors={method: predictor})
    curve = {"auc": .5, "fpr": [0, 1], "tpr": [0, 1]}
    path.write_bytes(b"a new checkpoint on disk that was never loaded")
    record = {"ok": True, "manifest_sha256": service.manifest_sha256, "task": "age_group_classification",
        "split": "validation", "samples": 3, "methods": {method: {"artifact_sha256": sha256_file(path),
        "accuracy_percent": 50, "score_type": "softmax", "roc": {"macro_auc_mean": .5,
        "per_class": {str(i): curve for i in range(3)}}}}}
    (tmp_path / "classification_roc.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="artifact"):
        evaluation_snapshot(tmp_path, service)


def test_real_cost_profile_matches_checkpoint_and_recorded_training_run():
    from visage.model_registry import EVALUATION, ARTIFACTS, METHODS
    path = EVALUATION / "application_profile.json"
    if not path.exists():
        pytest.skip("Recorded application profile is optional in source-only installs")
    profile = json.loads(path.read_text())
    assert profile["ok"] and set(profile["methods"]) == set(METHODS)
    for method, value in profile["methods"].items():
        summary = json.loads((ARTIFACTS / method / "training_summary.json").read_text())
        assert value["artifact_sha256"] == sha256_file(ARTIFACTS / method / "best.pt")
        assert value["parameters"] == summary["parameters"]
        assert value["training_run_id"] == summary["run_id"]
        assert value["training_seconds"] > 0 and value["runtime_median_ms"] > 0

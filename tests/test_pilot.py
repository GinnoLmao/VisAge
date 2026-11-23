import csv
from pathlib import Path

import pytest

from visage.pilot import PILOT_COLUMNS, run_pilot_rows, validate_pilot_manifest


def fixture_pilot(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    rows = []
    for i in range(3):
        (root / f"{i}.jpg").write_bytes(f"unique photo {i}".encode())
        rows.append(dict(zip(PILOT_COLUMNS, [f"s{i}", f"{i}.jpg", f"p{i}", str(i), "holdout", "true", ""])))
    return root, rows


def save(tmp_path, rows):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PILOT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_valid_manifest_tracks_counts_hashes_and_no_inference(tmp_path):
    root, source = fixture_pilot(tmp_path)
    rows, summary = validate_pilot_manifest(save(tmp_path, source), root)
    assert summary["counts"]["holdout"] == {"0": 1, "1": 1, "2": 1}
    assert len(rows[0]["file_sha256"]) == 64 and rows[0]["face_index"] is None


@pytest.mark.parametrize("field,value,match", [
    ("consent", "false", "consent"), ("relative_path", "../outside.jpg", "inside"),
    ("relative_path", "C:/outside.jpg", "inside"), ("face_index", "-1", "nonnegative"),
    ("split", "train", "splits"), ("label", "60+", "labels"),
    ("sample_id", "s1", "unique"), ("participant_id", "p1", "conflicting"),
])
def test_bad_manifest_rejected_before_inference(tmp_path, field, value, match):
    root, rows = fixture_pilot(tmp_path)
    rows[0][field] = value
    with pytest.raises(ValueError, match=match):
        validate_pilot_manifest(save(tmp_path, rows), root)


def test_duplicate_bytes_and_missing_class_are_rejected(tmp_path):
    root, rows = fixture_pilot(tmp_path)
    (root / "1.jpg").write_bytes((root / "0.jpg").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_pilot_manifest(save(tmp_path, rows), root)
    (root / "1.jpg").write_bytes(b"restored different photo")
    with pytest.raises(ValueError, match="all three"):
        validate_pilot_manifest(save(tmp_path, rows[:2]), root)


def test_rejections_remain_in_primary_denominator(tmp_path):
    root, source = fixture_pilot(tmp_path)
    rows, _ = validate_pilot_manifest(save(tmp_path, source), root)

    class BadFace(ValueError):
        code = "no_face"

    def predict(data, face_index=None):
        if data.endswith(b"1"):
            raise BadFace()
        label = int(data[-1:])
        return {"predictions": [{"method": name, "class_id": label} for name in ("efficientnetv2_s", "convnextv2_nano", "swinv2_tiny")]}

    result = run_pilot_rows(rows, root, predict)
    assert result["coverage"] == pytest.approx(2 / 3)
    for value in result["methods"].values():
        assert value["end_to_end_correct_rate"] == pytest.approx(2 / 3)
        assert value["accepted_image_metrics"]["accuracy"] == 1


def test_all_rejections_produce_zero_coverage_without_fabricated_metrics(tmp_path):
    root, source = fixture_pilot(tmp_path)
    rows, _ = validate_pilot_manifest(save(tmp_path, source), root)

    class NoFace(ValueError):
        code = "no_face"

    def predict(*args, **kwargs):
        raise NoFace()

    result = run_pilot_rows(rows, root, predict)
    assert result["coverage"] == 0
    assert all(value["accepted_image_metrics"] is None for value in result["methods"].values())


def test_changed_photo_or_internal_failure_cannot_be_silently_omitted(tmp_path):
    root, source = fixture_pilot(tmp_path)
    rows, _ = validate_pilot_manifest(save(tmp_path, source), root)

    def predict(*args, **kwargs):
        raise RuntimeError("internal failure")

    with pytest.raises(RuntimeError, match="internal"):
        run_pilot_rows(rows, root, predict)
    (root / "0.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run_pilot_rows(rows, root, predict)


def test_duplicate_csv_header_is_not_silently_collapsed(tmp_path):
    root, rows = fixture_pilot(tmp_path)
    path = save(tmp_path, rows)
    text = path.read_text()
    path.write_text(text.replace("face_index\n", "face_index,consent\n", 1))
    with pytest.raises(ValueError, match="exactly"):
        validate_pilot_manifest(path, root)


def test_duplicate_method_predictions_are_not_silently_collapsed(tmp_path):
    root, source = fixture_pilot(tmp_path)
    rows, _ = validate_pilot_manifest(save(tmp_path, source), root)

    def predict(*args, **kwargs):
        values = [{"method": name, "class_id": 0} for name in ("efficientnetv2_s", "convnextv2_nano", "swinv2_tiny", "convnextv2_nano")]
        return {"predictions": values}

    with pytest.raises(ValueError, match="one valid class"):
        run_pilot_rows(rows, root, predict)

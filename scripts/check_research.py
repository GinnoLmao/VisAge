"""Independently verify all research records, metrics and CPU artifact reloads.

Metrics and promotion arithmetic use plain Python rather than the production
metric/research helpers. Only fixed validation images are opened for reloads;
the original held-out reports may be hashed, but no test image is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visage.runtime import configure_caches, sha256_file, utc_now, write_json

METHODS = ("classical", "visagenet", "convnext_ordinal")
NAMES = ("Under 18", "18-59", "60+")
SCREENS = ("visagenet_sampler05_seed42", "visagenet_softmax_seed42")
CONDITIONS = ("clean", "dim", "bright", "blur", "jpeg", "low_resolution", "crop_shift")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close_values(actual, expected, context="record"):
    """Require all expected fields and reject invalid floating-point evidence."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected).issubset(actual):
            raise ValueError(f"{context}: missing structured fields.")
        for key, value in expected.items():
            close_values(actual[key], value, f"{context}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{context}: list length differs.")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            close_values(a, b, f"{context}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"{context}: numeric evidence differs.")
    elif actual != expected:
        raise ValueError(f"{context}: evidence differs.")


def independent_metrics(truth, predicted):
    if not truth or len(truth) != len(predicted) or any(value not in (0, 1, 2) for value in (*truth, *predicted)):
        raise ValueError("Three-class nonempty prediction pairs are required.")
    counts = Counter(zip(truth, predicted, strict=True))
    matrix = [[counts[(a, b)] for b in range(3)] for a in range(3)]
    support = [sum(row) for row in matrix]
    per_class = []
    for label in range(3):
        true_positive = matrix[label][label]
        predicted_count = sum(matrix[other][label] for other in range(3))
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support[label] if support[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label, "name": NAMES[label], "support": support[label],
                          "precision": precision, "recall": recall, "f1": f1})
    return {"samples": len(truth), "accuracy": sum(matrix[k][k] for k in range(3)) / len(truth),
            "macro_f1": sum(item["f1"] for item in per_class) / 3,
            "confusion_matrix": matrix,
            "row_normalized_confusion_matrix": [[n / support[k] if support[k] else 0.0 for n in row] for k, row in enumerate(matrix)],
            "per_class": per_class,
            "under18_predicted_older_rate": sum(matrix[0][1:]) / support[0] if support[0] else 0.0,
            "senior_predicted_below60_rate": sum(matrix[2][:2]) / support[2] if support[2] else 0.0,
            "non_senior_predicted_senior_rate": (matrix[0][2] + matrix[1][2]) / sum(support[:2]) if sum(support[:2]) else 0.0}


def check_prediction_rows(saved, validation, *, score_prefix=None):
    if len(saved) != len(validation):
        raise ValueError("Saved validation prediction count differs from the manifest.")
    predictions = []
    for current, frozen in zip(saved, validation, strict=True):
        if current["sample_id"] != frozen["sample_id"] or int(current["label"]) != int(frozen["label"]):
            raise ValueError("Saved prediction IDs, order or labels differ from validation.")
        for key in ("relative_path", "age"):
            if key in current and str(current[key]) != str(frozen[key]):
                raise ValueError(f"Saved validation {key} differs.")
        if "split" in current and current["split"] != "validation":
            raise ValueError("A prediction is not marked as validation.")
        predicted = int(current["prediction"])
        if predicted not in (0, 1, 2):
            raise ValueError("A saved prediction has an invalid class.")
        if score_prefix:
            values = [float(current[f"{score_prefix}_{label}"]) for label in range(3)]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Saved classifier values are nonfinite.")
            if score_prefix == "score" and (any(value < -1e-6 or value > 1 + 1e-6 for value in values)
                                             or not math.isclose(sum(values), 1.0, abs_tol=1e-5)):
                raise ValueError("Neural class values must be finite and sum to one.")
            if max(range(3), key=lambda index: values[index]) != predicted:
                raise ValueError("Saved classifier values disagree with the chosen class.")
        predictions.append(predicted)
    return independent_metrics([int(row["label"]) for row in validation], predictions)


def check_subsets(root, protocol, rows):
    training = [row for row in rows if row["split"] == "train"]
    by_id = {row["sample_id"]: row for row in training}
    previous = set()
    results = {}
    for percent in ("25", "50"):
        record = protocol["subsets"][percent]
        path = root / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError("A frozen training subset checksum changed.")
        subset = read_json(path)
        identities = subset["sample_ids"]
        chosen = set(identities)
        if (subset["split"] != "train" or subset["manifest_sha256"] != protocol["manifest_sha256"]
                or not chosen or len(identities) != len(chosen) or not chosen <= set(by_id)
                or not previous <= chosen):
            raise ValueError("Subset IDs are invalid, non-training or not nested.")
        ordered = [row for row in training if row["sample_id"] in chosen]
        if identities != [row["sample_id"] for row in ordered]:
            raise ValueError("Subset sample order differs from the frozen manifest.")
        groups = {row["group_id"] for row in ordered}
        if subset["group_ids"] != sorted(groups) or any(row["group_id"] in groups and row["sample_id"] not in chosen for row in training):
            raise ValueError("A perceptual group was split within a training subset.")
        counts = [sum(int(row["label"]) == label for row in ordered) for label in range(3)]
        if not all(counts):
            raise ValueError("A training subset is missing a class.")
        expected = {"rows": len(ordered), "fraction_actual": len(ordered) / len(training), "class_counts": counts}
        close_values(subset, expected, "subset")
        close_values(record, expected, "protocol subset")
        results[percent] = {**expected, "sha256": sha256_file(path), "whole_groups": True, "nested": True}
        previous = chosen
    return results


def expected_runs(candidate):
    neural = [*SCREENS, "convnext_warmup_seed42"]
    for seed in (43, 44):
        neural.extend([f"visagenet_baseline_seed{seed}", candidate.replace("seed42", f"seed{seed}")])
    neural.extend(f"{method}_data{percent}_seed42" for percent in (25, 50) for method in ("visagenet", "convnext_ordinal"))
    classical = [f"classical_data{percent}_seed42" for percent in (25, 50)] + ["classical_hog_seed42", "classical_lbp_seed42"]
    return neural, classical


def declared_neural_config(root, name, protocol_sha, training_source_sha):
    method = "convnext_ordinal" if name.startswith("convnext") else "visagenet"
    config = read_json(root / "artifacts" / method / "config.json")
    for key in ("manifest_sha256", "transform", "amp_dtype"):
        config.pop(key, None)
    seed = int(name.rsplit("seed", 1)[1])
    config.update(seed=seed, output_dir=f"artifacts/research/{name}", research_protocol_sha256=protocol_sha,
                  research_run_name=name, research_training_source_sha256=training_source_sha)
    if "sampler05" in name:
        config["sampler_exponent"] = 0.5
    if "softmax" in name:
        config["model_config"] = {"head_type": "softmax", "dropout": 0.2}
    if name == "convnext_warmup_seed42":
        config.update(learning_rate=3e-5, unfreeze_warmup_epochs=2)
    for percent in (25, 50):
        if f"_data{percent}_" in name:
            config["train_subset_path"] = f"reports/research/subsets/train_{percent}.json"
    return method, config


def reload_values(predictor, saved, positions, rows, raw, score_prefix):
    import numpy as np
    from PIL import Image, ImageOps
    import torch

    maximum_error = 0.0
    for position in positions:
        row = rows[position]
        path = (raw / row["relative_path"]).resolve(strict=True)
        if not path.is_relative_to(raw.resolve()) or sha256_file(path) != row["file_sha256"]:
            raise ValueError("A selected frozen validation image changed.")
        with Image.open(path) as source:
            face = ImageOps.exif_transpose(source).convert("RGB")
        try:
            if score_prefix == "margin":
                values = predictor.model.decision_function([face])[0]
            else:
                with torch.inference_mode():
                    logits = predictor.model(predictor.transform(face).unsqueeze(0)).float()
                    # Independent conversion of the two conditional CORN logits,
                    # or ordinary softmax for the explicit head ablation.
                    if predictor.model.head_type == "corn":
                        q1, q2 = torch.sigmoid(logits)[0]
                        values = torch.stack([1 - q1, q1 * (1 - q2), q1 * q2]).numpy()
                    else:
                        values = torch.softmax(logits, dim=1)[0].numpy()
            expected = np.array([float(saved[position][f"{score_prefix}_{label}"]) for label in range(3)])
            np.testing.assert_allclose(values, expected, rtol=1e-4, atol=2e-5)
            if int(values.argmax()) != int(saved[position]["prediction"]):
                raise ValueError("Reloaded CPU class differs from the saved validation prediction.")
            maximum_error = max(maximum_error, float(np.max(np.abs(values - expected))))
        finally:
            face.close()
    return maximum_error


def check_neural_run(root, name, protocol, validation, positions, file_hashes):
    import torch
    from visage.inference import AgeGroupPredictor

    directory = root / "artifacts/research" / name
    summary = read_json(directory / "training_summary.json")
    metrics = read_json(directory / "validation_metrics.json")
    config = read_json(directory / "config.json")
    history = read_json(directory / "history.json")
    protocol_sha = sha256_file(root / "reports/research/protocol.json")
    method, expected_config = declared_neural_config(root, name, protocol_sha, protocol["implementation"]["source_files"]["src/visage/training.py"])
    declared = read_json(root / f"reports/research/configs/{name}.json")
    if declared != expected_config or any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"{name}: declared training settings differ from the predeclared experiment.")
    expected_training = protocol["training_rows"]
    if config.get("train_subset_path"):
        subset = read_json(root / config["train_subset_path"])
        expected_training = subset["rows"]
        if config["train_subset_sha256"] != sha256_file(root / config["train_subset_path"]):
            raise ValueError("A run used changed training-subset bytes.")
    close_values(summary, {"method": method, "manifest_sha256": protocol["manifest_sha256"],
                          "training_rows": expected_training, "validation_rows": len(validation)}, name)
    if len(history) != summary["epochs_completed"] or [row["epoch"] for row in history] != list(range(1, len(history) + 1)) or len(history) > config["epochs"]:
        raise ValueError(f"{name}: incomplete or misordered epoch history.")
    best, best_epoch, bad_epochs = -1.0, 0, 0
    for entry in history:
        epoch = entry["epoch"]
        factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(config["epochs"] - 1, 1)))
        rates = [config["learning_rate"] * factor]
        if method == "convnext_ordinal":
            head_epochs = config.get("head_epochs", 0)
            warmup = config.get("unfreeze_warmup_epochs", 0)
            ramp = min((epoch - head_epochs) / warmup, 1.0) if warmup and epoch > head_epochs else 1.0
            rates[0] *= ramp
            rates.append(config["head_learning_rate"] * factor)
            if entry["backbone_trainable"] != (epoch > head_epochs):
                raise ValueError("Backbone freeze history differs from the declared head stage.")
        close_values(entry["learning_rates"], rates, name + " learning rates")
        if entry["val_macro_f1"] > best + 1e-5:
            best, best_epoch, bad_epochs = entry["val_macro_f1"], entry["epoch"], 0
        else:
            bad_epochs += 1
    if len(history) < config["epochs"] and (len(history) < config["min_epochs"] or bad_epochs < config["patience"]):
        raise ValueError(f"{name}: training stopped without reaching its budget or declared early stopping.")
    close_values(summary, {"best_epoch": best_epoch, "best_validation_macro_f1": best,
                          "training_seconds": sum(row["epoch_seconds"] for row in history)}, name)
    artifact_sha = sha256_file(directory / "best.pt")
    if artifact_sha != summary["artifact_sha256"] or (directory / "best.pt").stat().st_size != summary["artifact_bytes"]:
        raise ValueError(f"{name}: trained checkpoint differs from its summary.")
    saved = read_csv(directory / "validation_predictions.csv")
    measured = check_prediction_rows(saved, validation, score_prefix="score")
    close_values(metrics, measured, name + " metrics")
    close_values(summary, {"best_validation_macro_f1": measured["macro_f1"]}, name)
    predictor = AgeGroupPredictor(method, directory, device="cpu")
    checkpoint = predictor.metadata
    if checkpoint["model_config"] != config["model_config"] or checkpoint["transform"] != config["transform"] or checkpoint["epoch"] != best_epoch:
        raise ValueError(f"{name}: loaded checkpoint configuration differs.")
    close_values(checkpoint["validation_metrics"], measured, name + " checkpoint metrics")
    if sum(parameter.numel() for parameter in predictor.model.parameters()) != summary["parameters"]:
        raise ValueError(f"{name}: model parameter count differs.")
    maximum_error = reload_values(predictor, saved, positions, validation, root / "data/raw/UTKFace", "score")
    del predictor
    for filename in ("best.pt", "config.json", "training_summary.json", "history.json", "validation_metrics.json", "validation_predictions.csv"):
        path = directory / filename
        file_hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return {"ok": True, "method": method, "artifact_sha256": artifact_sha,
            "validation_rows_checked": len(saved), "training_rows": expected_training, "best_epoch": best_epoch,
            "reload_inference_samples": len(positions), "maximum_score_absolute_error": maximum_error,
            "independent_metrics": measured}


def check_classical_run(root, name, protocol, validation, positions, file_hashes):
    from visage.inference import AgeGroupPredictor

    directory = root / "artifacts/research" / name
    selection = read_json(directory / "selection.json")
    metrics = read_json(directory / "validation_metrics.json")
    baseline = read_json(root / "artifacts/classical/selection.json")
    feature_config = dict(baseline["feature_config"])
    descriptor = "hog" if "_hog_" in name else "lbp" if "_lbp_" in name else "combined"
    feature_config.update(include_hog=descriptor != "lbp", include_lbp=descriptor != "hog")
    percent = 25 if "data25" in name else 50 if "data50" in name else 100
    expected_training = protocol["training_rows"] if percent == 100 else protocol["subsets"][str(percent)]["rows"]
    training_manifest = [row for row in read_csv(root / "data/processed/manifest.csv") if row["split"] == "train"]
    class_counts = ([sum(int(row["label"]) == label for row in training_manifest) for label in range(3)]
                    if percent == 100 else protocol["subsets"][str(percent)]["class_counts"])
    close_values(selection, {"method": "classical", "manifest_sha256": protocol["manifest_sha256"],
                            "research_protocol_sha256": sha256_file(root / "reports/research/protocol.json"),
                            "research_run_name": name, "descriptor": descriptor, "feature_config": feature_config,
                            "selected_configuration": baseline["selected_configuration"],
                            "training_rows": expected_training, "class_counts": class_counts,
                            "validation_rows": len(validation), "test_evaluated": False}, name)
    saved = read_csv(directory / "validation_predictions.csv")
    measured = check_prediction_rows(saved, validation, score_prefix="margin")
    close_values(metrics, measured, name + " metrics")
    close_values(selection["validation_metrics"], measured, name + " selection metrics")
    predictor = AgeGroupPredictor("classical", directory, device="cpu")
    maximum_error = reload_values(predictor, saved, positions, validation, root / "data/raw/UTKFace", "margin")
    del predictor
    for filename in ("svm.npz", "svm.json", "features.json", "selection.json", "validation_metrics.json", "validation_predictions.csv"):
        path = directory / filename
        file_hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return {"ok": True, "method": "classical", "artifact_sha256": sha256_file(directory / "svm.npz"),
            "validation_rows_checked": len(saved), "training_rows": expected_training,
            "reload_inference_samples": len(positions), "maximum_score_absolute_error": maximum_error,
            "independent_metrics": measured}


def check_robustness(root, protocol, validation, file_hashes):
    path = root / "reports/research/robustness.json"
    record = read_json(path)
    if record["manifest_sha256"] != protocol["manifest_sha256"] or record["split"] != "validation" or record["samples"] != len(validation):
        raise ValueError("Robustness provenance differs from the validation freeze.")
    if set(record["conditions"]) != set(CONDITIONS) or record["perturbation_source_sha256"] != sha256_file(root / "src/visage/robustness.py"):
        raise ValueError("Robustness conditions or perturbation source changed.")
    results = {}
    for method in METHODS:
        entry = record["methods"][method]
        artifact_path = root / "artifacts" / method / ("svm.npz" if method == "classical" else "best.pt")
        if entry["artifact_sha256"] != sha256_file(artifact_path) or set(entry["conditions"]) != set(CONDITIONS):
            raise ValueError("Robustness is missing conditions or uses another artifact.")
        baseline = read_csv(root / "artifacts" / method / "validation_predictions.csv")
        baseline_metrics = check_prediction_rows(baseline, validation)
        conditions = {}
        for condition in CONDITIONS:
            evidence = entry["conditions"][condition]
            csv_path = (path.parent / evidence["predictions_csv"]).resolve()
            if not csv_path.is_relative_to(path.parent.resolve()) or sha256_file(csv_path) != evidence["predictions_sha256"]:
                raise ValueError("A robustness prediction CSV changed or escaped the evidence directory.")
            saved = read_csv(csv_path)
            measured = check_prediction_rows(saved, validation)
            close_values(evidence["metrics"], measured, f"robustness {method} {condition}")
            close_values(evidence["macro_f1_drop_from_clean"], baseline_metrics["macro_f1"] - measured["macro_f1"], "robustness drop")
            if condition == "clean" and [row["prediction"] for row in saved] != [row["prediction"] for row in baseline]:
                raise ValueError("Clean robustness predictions differ from the original saved validation predictions.")
            file_hashes[csv_path.relative_to(root).as_posix()] = sha256_file(csv_path)
            conditions[condition] = {"ok": True, "rows": len(saved), "predictions_sha256": sha256_file(csv_path),
                                     "macro_f1": measured["macro_f1"]}
        results[method] = conditions
    file_hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    # Independently recompute age and image-quality cohorts. Only validation
    # photos are opened, and only aggregate counts are retained in this report.
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    analysis_path = path.parent / "error_analysis.json"
    analysis = read_json(analysis_path)
    close_values(analysis, {"split": "validation", "manifest_sha256": protocol["manifest_sha256"],
                           "samples": len(validation), "analysis_source_sha256": sha256_file(root / "scripts/research_error_analysis.py")}, "error-analysis provenance")
    cohorts = []
    raw = (root / "data/raw/UTKFace").resolve()
    for row in validation:
        image_path = (raw / row["relative_path"]).resolve(strict=True)
        if not image_path.is_relative_to(raw) or sha256_file(image_path) != row["file_sha256"]:
            raise ValueError("A validation image used for quality-strata verification changed.")
        with Image.open(image_path) as source:
            face = ImageOps.exif_transpose(source).convert("RGB")
        try:
            gray = cv2.cvtColor(np.asarray(face), cv2.COLOR_RGB2GRAY)
            brightness = float(gray.mean())
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        finally:
            face.close()
        age = int(row["age"])
        cohorts.append({
            "age_band": "below15" if age < 15 else "15_to20" if age <= 20 else "21_to54" if age <= 54 else "55_to64" if age <= 64 else "65plus",
            "brightness_band": "dim_below60" if brightness < 60 else "middle60_to180" if brightness <= 180 else "bright_above180",
            "sharpness_band": "low_below100" if sharpness < 100 else "middle100_to500" if sharpness <= 500 else "high_above500",
        })
    for method in METHODS:
        baseline_path = root / "artifacts" / method / "validation_predictions.csv"
        if analysis["baseline_predictions_sha256"][method] != sha256_file(baseline_path):
            raise ValueError("Error analysis used different baseline predictions.")
        saved = read_csv(baseline_path)
        expected_strata = independent_strata(validation, saved, cohorts)
        close_values(analysis["methods"][method], expected_strata, "error-analysis strata")
    file_hashes[analysis_path.relative_to(root).as_posix()] = sha256_file(analysis_path)
    return results


def independent_strata(rows, saved, cohorts):
    if len(rows) != len(saved) or len(rows) != len(cohorts):
        raise ValueError("Strata must account for every validation image.")
    results = {}
    for variable in ("age_band", "brightness_band", "sharpness_band"):
        bands = {}
        for band in sorted({cohort[variable] for cohort in cohorts}):
            indices = [index for index, cohort in enumerate(cohorts) if cohort[variable] == band]
            correct = sum(int(rows[index]["label"]) == int(saved[index]["prediction"]) for index in indices)
            bands[band] = {"samples": len(indices), "correct": correct, "errors": len(indices) - correct,
                           "accuracy": correct / len(indices)}
        results[variable] = bands
    return results


def independent_promotion(root, protocol, candidate):
    baseline = [read_json(root / "artifacts/visagenet/validation_metrics.json")]
    challenger = [read_json(root / "artifacts/research" / candidate / "validation_metrics.json")]
    for seed in (43, 44):
        baseline.append(read_json(root / f"artifacts/research/visagenet_baseline_seed{seed}/validation_metrics.json"))
        challenger.append(read_json(root / "artifacts/research" / candidate.replace("seed42", f"seed{seed}") / "validation_metrics.json"))
    def averages(values):
        return {"macro_f1": sum(item["macro_f1"] for item in values) / 3,
                "minor_error": sum(item["under18_predicted_older_rate"] for item in values) / 3,
                "senior_recall": sum(item["per_class"][2]["recall"] for item in values) / 3}
    old, new = averages(baseline), averages(challenger)
    criteria = protocol["promotion"]
    checks = {
        "mean_macro_f1_improvement_at_least_0_005": new["macro_f1"] - old["macro_f1"] >= criteria["mean_macro_f1_delta_min"],
        "minor_error_increase_at_most_0_01": new["minor_error"] - old["minor_error"] <= criteria["minor_error_mean_increase_max"],
        "senior_recall_drop_at_most_0_01": old["senior_recall"] - new["senior_recall"] <= criteria["senior_recall_mean_drop_max"],
    }
    return {"promote": all(checks.values()), "checks": checks, "baseline_mean": old, "candidate_mean": new}


def verify(root, samples=24):
    import numpy as np
    import torch

    configure_caches(root)
    torch.set_num_threads(4)
    study = root / "reports/research"
    protocol_path = study / "protocol.json"
    protocol = read_json(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    manifest_path = root / "data/processed/manifest.csv"
    manifest_sha = sha256_file(manifest_path)
    if protocol["format_version"] != 2 or protocol["manifest_sha256"] != manifest_sha:
        raise ValueError("The research protocol does not match the frozen manifest.")
    sources = protocol["implementation"]["source_files"]
    if len(sources) != 16:
        raise ValueError("The predeclared study must pin all sixteen executable source files.")
    for mapping in (sources, protocol["protected_original_files"]):
        for relative, expected in mapping.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or sha256_file(path) != expected:
                raise ValueError(f"Protected source or original evidence changed: {relative}")
    rows = read_csv(manifest_path)
    validation = [row for row in rows if row["split"] == "validation"]
    train = [row for row in rows if row["split"] == "train"]
    identity = hashlib.sha256(json.dumps([(row["sample_id"], int(row["label"])) for row in validation], separators=(",", ":")).encode()).hexdigest()
    if (identity != protocol["validation_identity_sha256"] or len(validation) != protocol["validation_rows"]
            or len(train) != protocol["training_rows"]):
        raise ValueError("Research train/validation identity or counts changed.")
    subsets = check_subsets(root, protocol, rows)
    scores = {name: read_json(root / "artifacts/research" / name / "training_summary.json")["best_validation_macro_f1"] for name in SCREENS}
    candidate = max(enumerate(SCREENS), key=lambda value: (scores[value[1]], -value[0]))[1]
    close_values(read_json(study / "candidate_selection.json"), {"selected_seed42_run": candidate,
                 "protocol_sha256": protocol_sha, "screen_scores": scores}, "candidate selection")
    neural_names, classical_names = expected_runs(candidate)
    summary = read_json(study / "study_summary.json")
    if set(summary["runs"]) != set(neural_names + classical_names):
        raise ValueError("Research summary must contain exactly the eleven neural and four classical runs.")
    close_values(summary, {"manifest_sha256": manifest_sha, "protocol_sha256": protocol_sha,
                          "expected_new_runs": 15, "completed_new_runs": 15,
                          "training_experiments_complete": True, "study_complete": True,
                          "deployment_changed": False, "new_test_evaluations": 0,
                          "original_test_evidence_unchanged": True}, "study summary")
    positions = np.unique(np.linspace(0, len(validation) - 1, min(samples, len(validation)), dtype=np.int64)).tolist()
    files = {"reports/research/protocol.json": protocol_sha, "reports/research/study_summary.json": sha256_file(study / "study_summary.json"),
             "reports/research/candidate_selection.json": sha256_file(study / "candidate_selection.json")}
    files.update({record["path"]: record["sha256"] for record in protocol["subsets"].values()})
    checks = {}
    for name in neural_names + classical_names:
        check = check_neural_run if name in neural_names else check_classical_run
        checks[name] = check(root, name, protocol, validation, positions, files)
        run_record = summary["runs"][name]
        close_values(run_record["validation_metrics"], checks[name]["independent_metrics"], name + " summary metrics")
        source_name = "training_summary.json" if name in neural_names else "selection.json"
        source_key = "summary" if name in neural_names else "selection"
        if run_record[source_key] != read_json(root / "artifacts/research" / name / source_name):
            raise ValueError(f"{name}: summary did not preserve the actual completed run record.")
        print(json.dumps({"run": name, "ok": True, "validation_rows": len(validation), "reload_samples": len(positions)}), flush=True)
    robustness = check_robustness(root, protocol, validation, files)
    promotion = independent_promotion(root, protocol, candidate)
    close_values(summary["promotion"], promotion, "independent promotion")
    if summary["proposed_champion"] != (candidate if promotion["promote"] else "original_visagenet_seed42"):
        raise ValueError("The proposed champion differs from the predeclared promotion criteria.")
    for label, names in (("baseline", [None, "visagenet_baseline_seed43", "visagenet_baseline_seed44"]),
                         ("selected_candidate", [candidate, candidate.replace("seed42", "seed43"), candidate.replace("seed42", "seed44")])):
        values = [(read_json(root / "artifacts/visagenet/validation_metrics.json") if name is None
                   else checks[name]["independent_metrics"])["macro_f1"] for name in names]
        mean = sum(values) / 3
        close_values(summary["seed_confirmation"][label], {"seeds_completed": 3, "validation_macro_f1": values,
                     "mean": mean, "sample_std": math.sqrt(sum((value - mean) ** 2 for value in values) / 2)}, "seed confirmation")
    if len(summary["data_efficiency_table"]) != 9:
        raise ValueError("The data-efficiency table must contain three fractions for all three methods.")
    for method in METHODS:
        for percent in (25, 50, 100):
            table_rows = [entry for entry in summary["data_efficiency_table"] if entry["method"] == method and entry["fraction_requested"] == percent / 100]
            if len(table_rows) != 1:
                raise ValueError("Data-efficiency rows are missing or duplicated.")
            count = len(train) if percent == 100 else subsets[str(percent)]["rows"]
            measured = read_json(root / "artifacts" / method / "validation_metrics.json") if percent == 100 else checks[f"{method}_data{percent}_seed42"]["independent_metrics"]
            close_values(table_rows[0], {"training_rows": count, "fraction_actual": count / len(train), "validation_metrics": measured}, "data efficiency")
    for name in ("robustness.json", "error_analysis.json"):
        close_values(summary["external_evidence"][name], {"status": "present", "sha256": sha256_file(study / name)}, "external evidence")
    # Recheck pinned/protected records after reloads to detect concurrent mutation.
    for relative, expected in {**sources, **protocol["protected_original_files"], **files}.items():
        if sha256_file(root / relative) != expected:
            raise ValueError("Evidence changed while independent verification was running.")
    return {"ok": True, "study_complete": True, "checked_utc": utc_now(), "manifest_sha256": manifest_sha,
            "protocol_sha256": protocol_sha, "expected_runs": 15, "completed_runs": len(checks),
            "neural_runs": len(neural_names), "classical_runs": len(classical_names), "subsets": subsets,
            "validation_rows": len(validation), "validation_positions": positions, "per_run_checks": checks,
            "robustness": robustness, "robustness_condition_checks": 21,
            "selected_candidate": candidate, "promotion": promotion,
            "source_checksums": sources, "protected_original_file_checksums": protocol["protected_original_files"],
            "evidence_checksums": files, "score_tolerance": {"rtol": 1e-4, "atol": 2e-5},
            "scope": "Independent Counter-based arithmetic, fixed validation-row identity, configuration/history/artifact hashes and CPU reload predictions. No training or test-image inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=24, help="Evenly spaced validation reload samples per run; positive only.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/research/independent_review.json")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive; CPU artifact reloads are required.")
    result = verify(ROOT, samples=args.samples)
    write_json(args.output, result)
    print(f"Verified {result['completed_runs']} research runs and 21 robustness conditions: {args.output}")


if __name__ == "__main__":
    main()

"""Run the predeclared validation-only improvement study, sequentially.

Original artifacts and held-out reports are checksum-protected. Every new run
has its own directory/config/log, and completed matching runs are reused.
No held-out image inference or automatic deployment occurs here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visage.data import load_manifest, verify_manifest
from visage.research import (
    identity_sha, nested_training_subsets, promotion_decision,
    verify_protected_files, write_immutable_json,
)
from visage.runtime import configure_caches, sha256_file, utc_now, write_json

configure_caches(ROOT)
STUDY = ROOT / "reports/research"
ARTIFACTS = ROOT / "artifacts/research"
SCREENS = ("visagenet_sampler05_seed42", "visagenet_softmax_seed42")
STUDY_SOURCES = (
    "src/visage/training.py", "src/visage/research.py", "scripts/run_research_study.py",
    "scripts/research_classical.py", "src/visage/methods/visagenet.py",
    "src/visage/methods/convnext_ordinal.py", "src/visage/methods/ordinal.py",
    "src/visage/methods/classical.py", "src/visage/augmentation.py",
    "src/visage/datasets.py", "src/visage/tensor_cache.py", "src/visage/metrics.py",
    "src/visage/runtime.py", "src/visage/data.py", "scripts/train_neural.py", "scripts/train_classical.py",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def initialize_protocol():
    path = STUDY / "protocol.json"
    manifest = ROOT / "data/processed/manifest.csv"
    verify_manifest(manifest, ROOT / "data/raw/UTKFace", verify_files=False)
    rows = load_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    if path.exists():
        protocol = read(path)
        if protocol.get("format_version") != 2:
            raise ValueError("Archive the superseded preparation draft before freezing research protocol v2.")
        if protocol["manifest_sha256"] != manifest_sha:
            raise ValueError("Research protocol belongs to another dataset freeze.")
        verify_protected_files(ROOT, protocol["protected_original_files"])
        verify_protected_files(ROOT, protocol["implementation"]["source_files"])
        for fraction in (25, 50):
            subset_path = STUDY / f"subsets/train_{fraction}.json"
            if sha256_file(subset_path) != protocol["subsets"][str(fraction)]["sha256"]:
                raise ValueError("A predeclared training subset changed.")
        return protocol
    protected = {}
    for method in ("classical", "visagenet", "convnext_ordinal"):
        for file in (ROOT / "artifacts" / method).rglob("*"):
            if file.is_file(): protected[relative(file)] = sha256_file(file)
    for file in (ROOT / "reports/evaluation").rglob("*"):
        if file.is_file(): protected[relative(file)] = sha256_file(file)
    subsets = {}
    for fraction, subset in nested_training_subsets(rows, manifest_sha).items():
        percent = str(round(100 * fraction))
        subset_path = STUDY / f"subsets/train_{percent}.json"
        write_immutable_json(subset_path, subset)
        subsets[percent] = {"path": relative(subset_path), "sha256": sha256_file(subset_path),
                            "rows": subset["rows"], "fraction_actual": subset["fraction_actual"],
                            "class_counts": subset["class_counts"]}
    protocol = {
        "format_version": 2, "study_name": "VisAge bounded validation-only improvement study v2",
        "created_utc": utc_now(), "manifest_sha256": manifest_sha,
        "training_rows": sum(row["split"] == "train" for row in rows),
        "preparation_history": "Superseded preparation drafts, if any, are archived separately before any research training; protocol v2 freezes executable source semantics.",
        "validation_rows": sum(row["split"] == "validation" for row in rows),
        "validation_identity_sha256": identity_sha([row for row in rows if row["split"] == "validation"]),
        "protected_original_files": protected, "subsets": subsets,
        "selection_source": "Existing frozen validation split only; no new test-image reads or test-result selection.",
        "test_status": "Original test results already observed; preserved as original reporting. Any future reuse is exploratory; fresh confirmation needs new independent data.",
        "screens": {
            "visagenet_sampler05_seed42": "Original CORN architecture; inverse-frequency sampling exponent 0.5 instead of 1.0; all other settings unchanged.",
            "visagenet_softmax_seed42": "Original residual SE architecture; three-class softmax head/cross-entropy instead of two-logit CORN; original sampling exponent 1.0.",
            "convnext_warmup_seed42": "Combined optimization candidate: backbone LR 3e-5 instead of 1e-4 plus two-epoch backbone-only unfreeze warm-up; head LR unchanged. Exploratory single seed; no causal attribution to individual components.",
        },
        "confirmation": {"seeds": [42, 43, 44], "baseline_seed42": "Reuse original verified VisAgeNet artifact.",
                         "candidate_selection": "Highest seed42 validation macro F1 among the two VisAgeNet screens; declared order sampler05 then softmax breaks exact ties. Confirm selected candidate even if initial improvement is small/negative.",
                         "modern_candidate": "Single-seed exploratory only; cannot replace the champion without separate repeat-seed confirmation."},
        "promotion": {"mean_macro_f1_delta_min": 0.005, "minor_error_mean_increase_max": 0.01,
                      "senior_recall_mean_drop_max": 0.01, "paired_seeds": 3,
                      "deployment": "Record proposed champion only; do not overwrite baseline artifacts or automatically change application model."},
        "data_efficiency": "Nested dominant-label-stratified whole training groups at approximate 25/50%; same subset IDs for all three methods, unchanged validation; baseline configurations/epoch budgets and seed42; reuse original full100 runs.",
        "classical_ablations": "HOG-only and multiscale LBP-only versus original combined descriptors; core NumPy implementation, original selected SVM hyperparameters and 60 epochs, seed42; no new hyperparameter grid.",
        "neural_budgets": {"visagenet_epochs": 25, "convnext_epochs": 18,
                           "early_stopping": "Original min-epoch/patience settings, no arbitrary truncation.",
                           "estimated_gpu_minutes": 110, "execution": "One GPU training job at a time; CPU/GPU stage times recorded."},
        "robustness": {"split": "full frozen validation", "methods": ["classical", "visagenet", "convnext_ordinal"],
                       "conditions": {"clean": "Recompute and check exact original saved validation label predictions", "dim": "PIL brightness factor0.65", "bright": "PIL brightness factor1.35", "blur": "PIL Gaussian radius1.5", "jpeg": "JPEG quality35/subsampling2 in memory", "low_resolution": "56x56 then original decoded dimensions bicubic, before common per-method resize", "crop_shift": "Translate right by max(1,round(5% width)) pixels with black padding"},
                       "scope": "Aligned face-crop perturbation robustness; no full-photo/detector claim.",
                       "owner_outputs": ["reports/research/robustness.json", "reports/research/error_analysis.json"]},
        "error_analysis": "Provided-age bins15–20/55–64 and brightness/blur indicators on validation data; metadata/statistics only, no face images retained.",
        "local_pilot": "Pending: no participant data available. Root-owned protocol/labels template/runner; never fabricate outcomes.",
        "colab": "Not required by user.",
        "implementation": {"source_files": {name: sha256_file(ROOT / name) for name in STUDY_SOURCES}},
    }
    write_immutable_json(path, protocol)
    return protocol


def base_config(method):
    config = read(ROOT / "artifacts" / method / "config.json")
    for key in ("manifest_sha256", "transform", "amp_dtype"):
        config.pop(key, None)
    return config


def config_for(name, method="visagenet", **overrides):
    config = base_config(method)
    config.update(overrides)
    config["output_dir"] = relative(ARTIFACTS / name)
    config["research_protocol_sha256"] = sha256_file(STUDY / "protocol.json")
    config["research_run_name"] = name
    config["research_training_source_sha256"] = sha256_file(ROOT / "src/visage/training.py")
    return config


def record_status(event, **values):
    status = {"event": event, "recorded_utc": utc_now(), **values}
    write_json(STUDY / "study_status.json", status)
    print(json.dumps(status), flush=True)


def run_neural(name, config, protocol):
    verify_protected_files(ROOT, protocol["protected_original_files"])
    verify_protected_files(ROOT, protocol["implementation"]["source_files"])
    output = ROOT / config["output_dir"]
    if not output.resolve().is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("Research outputs must remain separate from original artifacts.")
    config_path = STUDY / "configs" / f"{name}.json"
    write_immutable_json(config_path, config)
    summary_path = output / "training_summary.json"
    if summary_path.exists():
        summary = read(summary_path)
        actual_config = read(output / "config.json")
        if any(actual_config.get(key) != value for key, value in config.items()):
            raise ValueError("Completed research configuration differs from declared run.")
        if summary["artifact_sha256"] != sha256_file(output / "best.pt"):
            raise ValueError("Completed research artifact changed.")
        record_status("completed_run_reused", run=name, validation_macro_f1=summary["best_validation_macro_f1"])
        return summary
    log_path = STUDY / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "scripts/train_neural.py"), str(config_path)]
    if (output / "last.pt").exists(): command.append("--resume")
    record_status("training_started", run=name, log=relative(log_path))
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                       check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    summary = read(summary_path)
    write_json(output / "research_execution.json", {"wall_seconds_this_invocation": time.perf_counter() - started,
               "completed_utc": utc_now(), "protocol_sha256": sha256_file(STUDY / "protocol.json"),
               "config_sha256": sha256_file(config_path), "log_sha256": sha256_file(log_path)})
    record_status("training_complete", run=name, validation_macro_f1=summary["best_validation_macro_f1"],
                  training_seconds=summary["training_seconds"], artifact_sha256=summary["artifact_sha256"])
    return summary


def screens(protocol):
    run_neural(SCREENS[0], config_for(SCREENS[0], sampler_exponent=0.5), protocol)
    run_neural(SCREENS[1], config_for(SCREENS[1], model_config={"head_type": "softmax", "dropout": 0.2}), protocol)
    name = "convnext_warmup_seed42"
    run_neural(name, config_for(name, "convnext_ordinal", learning_rate=3e-5, unfreeze_warmup_epochs=2), protocol)


def selected_candidate():
    choices = [(read(ARTIFACTS / name / "training_summary.json")["best_validation_macro_f1"], -index, name)
               for index, name in enumerate(SCREENS)]
    return max(choices)[2]


def confirmation(protocol):
    candidate = selected_candidate()
    candidate_overrides = ({"sampler_exponent": 0.5} if candidate == SCREENS[0]
                           else {"model_config": {"head_type": "softmax", "dropout": 0.2}})
    write_immutable_json(STUDY / "candidate_selection.json", {
        "selected_seed42_run": candidate, "criterion": protocol["confirmation"]["candidate_selection"],
        "screen_scores": {name: read(ARTIFACTS / name / "training_summary.json")["best_validation_macro_f1"] for name in SCREENS},
        "protocol_sha256": sha256_file(STUDY / "protocol.json"),
    })
    for seed in (43, 44):
        baseline_name = f"visagenet_baseline_seed{seed}"
        run_neural(baseline_name, config_for(baseline_name, seed=seed), protocol)
        name = candidate.replace("seed42", f"seed{seed}")
        run_neural(name, config_for(name, seed=seed, **candidate_overrides), protocol)


def data_efficiency(protocol):
    for percent in (25, 50):
        for method in ("visagenet", "convnext_ordinal"):
            name = f"{method}_data{percent}_seed42"
            run_neural(name, config_for(name, method, train_subset_path=f"reports/research/subsets/train_{percent}.json"), protocol)


def summarize(protocol):
    results = {}
    for directory in sorted(ARTIFACTS.glob("*")):
        summary = directory / "training_summary.json"
        selection = directory / "selection.json"
        if summary.exists():
            results[directory.name] = {"summary": read(summary), "validation_metrics": read(directory / "validation_metrics.json"),
                                       "artifact_dir": relative(directory)}
        elif selection.exists():
            results[directory.name] = {"selection": read(selection), "validation_metrics": read(directory / "validation_metrics.json"),
                                       "artifact_dir": relative(directory)}
    seed_results, promotion = {}, {"promote": False, "reason": "Confirmation pending; preserve baseline champion."}
    if all((ARTIFACTS / name / "training_summary.json").exists() for name in SCREENS):
        candidate = selected_candidate()
        baseline = [read(ROOT / "artifacts/visagenet/validation_metrics.json")]
        challenger = [read(ARTIFACTS / candidate / "validation_metrics.json")]
        for seed in (43, 44):
            baseline_path = ARTIFACTS / f"visagenet_baseline_seed{seed}/validation_metrics.json"
            challenger_path = ARTIFACTS / candidate.replace("seed42", f"seed{seed}") / "validation_metrics.json"
            if baseline_path.exists() and challenger_path.exists():
                baseline.append(read(baseline_path)); challenger.append(read(challenger_path))
        promotion = promotion_decision(baseline, challenger)
        for label, metrics in (("baseline", baseline), ("selected_candidate", challenger)):
            values = [value["macro_f1"] for value in metrics]
            seed_results[label] = {"seeds_completed": len(values), "validation_macro_f1": values,
                                   "mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None}
    data_table = []
    for method in ("classical", "visagenet", "convnext_ordinal"):
        for percent in (25, 50, 100):
            name = f"{method}_data{percent}_seed42"
            if percent == 100:
                source = ROOT / "artifacts" / method
                metrics = read(source / "validation_metrics.json")
                record = read(source / ("selection.json" if method == "classical" else "training_summary.json"))
                count = protocol["training_rows"]
            elif name in results:
                source = ARTIFACTS / name
                metrics = results[name]["validation_metrics"]
                record = results[name].get("summary", results[name].get("selection"))
                count = record["training_rows"]
            else:
                continue
            data_table.append({"method": method, "fraction_requested": percent / 100, "training_rows": count,
                               "fraction_actual": count / protocol["training_rows"], "validation_metrics": metrics,
                               "training_seconds": record.get("training_seconds", record.get("selected_fit_seconds")),
                               "artifact_dir": relative(source)})
    expected_count = 15
    external = {name: {"status": "present" if (STUDY / name).exists() else "pending",
                       "sha256": sha256_file(STUDY / name) if (STUDY / name).exists() else None}
                for name in ("robustness.json", "error_analysis.json")}
    summary = {"created_utc": utc_now(), "manifest_sha256": protocol["manifest_sha256"],
               "protocol_sha256": sha256_file(STUDY / "protocol.json"), "runs": results,
               "training_experiments_complete": len(results) == expected_count,
               "expected_new_runs": expected_count, "completed_new_runs": len(results),
               "data_efficiency_table": data_table, "external_evidence": external,
               "study_complete": len(results) == expected_count and all(value["status"] == "present" for value in external.values()),
               "seed_confirmation": seed_results, "promotion": promotion,
               "proposed_champion": selected_candidate() if promotion["promote"] else "original_visagenet_seed42",
               "deployment_changed": False, "modern_candidate_status": "Exploratory single seed; not eligible for promotion.",
               "original_test_evidence_unchanged": True, "new_test_evaluations": 0,
               "local_pilot": "pending: no local participant data provided", "colab": "not required"}
    verify_protected_files(ROOT, protocol["protected_original_files"])
    write_json(STUDY / "study_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screens", "confirmation", "data_efficiency", "classical", "summary", "all"), default="all")
    args = parser.parse_args()
    protocol = initialize_protocol()
    record_status("protocol_verified", protocol_sha256=sha256_file(STUDY / "protocol.json"), stage=args.stage)
    if args.stage in {"screens", "all"}: screens(protocol)
    if args.stage in {"confirmation", "all"}: confirmation(protocol)
    if args.stage in {"data_efficiency", "all"}: data_efficiency(protocol)
    if args.stage in {"classical", "all"}:
        command = [sys.executable, str(ROOT / "scripts/research_classical.py")]
        subprocess.run(command, cwd=ROOT, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    summary = summarize(protocol)
    record_status("stage_complete", stage=args.stage, completed_runs=len(summary["runs"]), promotion=summary["promotion"])


if __name__ == "__main__":
    main()

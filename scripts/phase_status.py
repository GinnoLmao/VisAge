"""Read saved progress for the current model phase; starts no jobs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    status = {}
    for method in ("classical", "visagenet", "convnext_ordinal"):
        directory = ROOT / "artifacts" / method
        complete = directory / ("selection.json" if method == "classical" else "training_summary.json")
        if complete.exists():
            record = json.loads(complete.read_text(encoding="utf-8"))
            score = record.get("best_validation_macro_f1", record.get("validation_metrics", {}).get("macro_f1"))
            status[method] = {"status": "training_complete", "validation_macro_f1": score,
                              "best_epoch": record.get("best_epoch"), "epochs": record.get("epochs_completed")}
        elif (directory / "history.json").exists():
            history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
            status[method] = {"status": "in_progress", "last_completed_epoch": history[-1]["epoch"],
                              "last_validation_macro_f1": history[-1]["val_macro_f1"]}
        else:
            status[method] = {"status": "not_complete"}
    for name, relative in (("data_audit", "reports/data_audit_review.json"),
                           ("artifact_verification", "reports/artifact_verification.json"),
                           ("notebook", "reports/notebook_runner.json")):
        path = ROOT / relative
        status[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "pending"}
    comparison = ROOT / "reports/evaluation/comparison.json"
    status["held_out_evaluation"] = {"status": "complete" if comparison.exists() else "pending"}
    status["report_pdf"] = {"status": "present" if (ROOT / "output/pdf/VisAge_Phase_2_to_6_Implementation_Report.pdf").exists() else "pending"}
    research = ROOT / "reports/research/study_status.json"
    status["research"] = json.loads(research.read_text(encoding="utf-8")) if research.exists() else {"status": "pending"}
    current_run = status["research"].get("run")
    if current_run:
        current_history = ROOT / "artifacts/research" / current_run / "history.json"
        if current_history.exists():
            history = json.loads(current_history.read_text(encoding="utf-8"))
            status["research"]["last_completed_epoch"] = history[-1]["epoch"]
            status["research"]["last_validation_macro_f1"] = history[-1]["val_macro_f1"]
    for name, relative in (("backend_http_check", "reports/api_http_smoke.json"),
                           ("research_review", "reports/research/independent_review.json"),
                           ("backend_benchmark", "reports/backend_benchmark.json")):
        path = ROOT / relative
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        status[name] = {"status": "passed" if record.get("ok") else "pending"}
    status["local_pilot"] = {"status": "awaiting_consented_labeled_photos", "protocol": "configs/local_pilot_protocol.json"}
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

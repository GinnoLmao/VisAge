"""Build a factual backend/improvement report only from completed local evidence.

Use the bundled document Python runtime for PDF generation. It invokes the
project Python only for standalone Matplotlib figures; no training or inference
is started here. Missing, incomplete or changed evidence stops the build.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("classical", "visagenet", "convnext_ordinal")
LABELS = {"classical": "LBP/HOG + SVM", "visagenet": "VisAgeNet", "convnext_ordinal": "ConvNeXt V2"}
EXPECTED_PAGES = 19
RECORDS = {
    "protocol": "reports/research/protocol.json",
    "study": "reports/research/study_summary.json",
    "review": "reports/research/independent_review.json",
    "robustness": "reports/research/robustness.json",
    "errors": "reports/research/error_analysis.json",
    "face_smoke": "reports/face_pipeline_smoke.json",
    "http_smoke": "reports/api_http_smoke.json",
    "benchmark": "reports/backend_benchmark.json",
    "pilot": "configs/local_pilot_protocol.json",
    "detector": ".cache/face_detector/provenance.json",
    "freeze": "data/processed/manifest.freeze.json",
    "audit": "data/processed/audit.json",
    "face_validation": "reports/face_pipeline_validation.json",
    "classification_roc": "reports/extended_evaluation/classification_roc.json",
    "detection_ap": "reports/extended_evaluation/detection_ap.json",
    "extended_review": "reports/extended_evaluation/independent_review.json",
    "alignment_fit": "reports/training_alignment_comparison.json",
    "quality_fit": "reports/face_quality_fit.json",
    "pipeline_review": "reports/face_pipeline_independent_review.json",
    "deployment": "reports/deployment/decision.json",
    "cuda_smoke": "reports/backend_cuda_smoke.json",
    "detector_reload": "reports/extended_evaluation/detector_reload_review.json",
    "software_checks": "reports/final_software_checks.json",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value in {path}: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def tests_summary(path):
    tree = ET.parse(path)
    cases = tree.findall(".//testcase")
    result = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        for kind in ("failure", "error", "skipped"):
            result[{"failure": "failures", "error": "errors", "skipped": "skipped"}[kind]] += case.find(kind) is not None
    result["passed"] = result["tests"] - sum(result[k] for k in ("failures", "errors", "skipped"))
    result["groups"] = sorted({c.attrib.get("classname", "").split(".")[-1] for c in cases})
    return result


def checked_metrics(metrics, samples):
    matrix = metrics["confusion_matrix"]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Expected a three-class confusion matrix.")
    if metrics["samples"] != samples or sum(sum(row) for row in matrix) != samples:
        raise ValueError("Validation predictions do not cover the shared validation partition.")
    for key in ("accuracy", "macro_f1", "under18_predicted_older_rate", "senior_predicted_below60_rate"):
        if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise ValueError("Invalid classification metric.")


def load_evidence(root):
    required = [*RECORDS.values(), "reports/tests_backend_research.xml", "reports/tests_integration.xml",
                "data/processed/manifest.csv", ".cache/face_detector/LICENSE",
                ".cache/face_detector/face_detection_yunet_2023mar.onnx",
                "configs/local_pilot_labels_template.csv"]
    required += [f"artifacts/{method}/validation_metrics.json" for method in METHODS]
    required += ["reports/extended_evaluation/age_group_roc.png", "reports/extended_evaluation/detection_ap_pr_froc.png",
                 "reports/extended_evaluation/detection_predictions.json",
                 "reports/face_pipeline_validation.csv", "data/detection_benchmark/provenance.json",
                 "data/detection_benchmark/extracted/wider_face_split/wider_face_val_bbx_gt.txt"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Completed evidence is required before report generation:\n" + "\n".join(missing))
    data = {key: read_json(root / name) for key, name in RECORDS.items()}
    if not data["software_checks"].get("ok") or any(item["returncode"] != 0 for item in data["software_checks"]["commands"]):
        raise ValueError("Final lint/dependency verification must pass.")
    if any(not data[key].get("ok") for key in ("deployment", "cuda_smoke", "detector_reload")):
        raise ValueError("Final local model decision and actual CPU/CUDA checks must pass.")
    for relative, expected in data["deployment"]["evidence_checksums"].items():
        if sha256(root / relative) != expected:
            raise ValueError("Evidence changed after the final local model decision.")
        required.append(relative)
    if data["cuda_smoke"]["artifact_sha256"] != data["deployment"]["artifact_sha256"]:
        raise ValueError("CPU/CUDA smoke does not match the active model bundle.")
    if data["detector_reload"]["source_sha256"]["src/visage/faces.py"] != sha256(root / "src/visage/faces.py"):
        raise ValueError("Detector source changed after benchmark reload verification.")
    if data["detector_reload"]["predictions_sha256"] != sha256(root / "reports/extended_evaluation/detection_predictions.json"):
        raise ValueError("Detector reload check belongs to other benchmark predictions.")
    if any(not data[key].get("ok") for key in ("face_validation", "classification_roc", "detection_ap", "extended_review", "pipeline_review")):
        raise ValueError("Full face-pipeline diagnostics, AP/ROC and their independent review must pass.")
    if data["pipeline_review"]["report_sha256"] != sha256(root / RECORDS["face_validation"]):
        raise ValueError("The shared-pipeline review belongs to a different diagnostic run.")
    data["tests"] = tests_summary(root / "reports/tests_backend_research.xml")
    data["integration"] = tests_summary(root / "reports/tests_integration.xml")
    if (data["software_checks"]["full_test_report_sha256"] != sha256(root / "reports/tests_backend_research.xml")
            or data["software_checks"]["integration_report_sha256"] != sha256(root / "reports/tests_integration.xml")
            or data["software_checks"]["lock_sha256"] != sha256(root / "requirements-lock-backend-windows-cu128.txt")):
        raise ValueError("Software-check evidence must match the final tests and dependency lock.")
    data["baseline"] = {method: read_json(root / f"artifacts/{method}/validation_metrics.json") for method in METHODS}
    manifest_sha = sha256(root / "data/processed/manifest.csv")
    if data["freeze"]["manifest_sha256"] != manifest_sha:
        raise ValueError("Current dataset differs from its recorded freeze.")
    if any(data[key]["manifest_sha256"] != manifest_sha for key in ("face_validation", "classification_roc")):
        raise ValueError("Face-pipeline/ROC diagnostics belong to another dataset freeze.")
    if sha256(root / data["face_validation"]["predictions_csv"]) != data["face_validation"]["predictions_sha256"]:
        raise ValueError("Face-pipeline diagnostic predictions changed.")
    for relative, expected in data["face_validation"]["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError("Face-pipeline diagnostic source changed.")
    for name, expected in data["extended_review"]["evidence_checksums"].items():
        if sha256(root / "reports/extended_evaluation" / name) != expected:
            raise ValueError("AP/ROC evidence changed after independent review.")
    if data["detection_ap"]["images"] != 3226:
        raise ValueError("Full WIDER FACE validation evidence is required.")
    for key in ("protocol", "study", "robustness", "errors"):
        if data[key]["manifest_sha256"] != manifest_sha:
            raise ValueError(f"The {key} evidence belongs to another data freeze.")
    protocol, study = data["protocol"], data["study"]
    if study["protocol_sha256"] != sha256(root / RECORDS["protocol"]):
        raise ValueError("Research summary and predeclared protocol differ.")
    if not data["review"].get("study_complete"):
        raise ValueError("The research study is incomplete; no final PDF is generated.")
    if len(study["runs"]) != 15:
        raise ValueError("All fifteen declared research runs are required.")
    if (study["new_test_evaluations"] != 0 or study["deployment_changed"]
            or not study["original_test_evidence_unchanged"]):
        raise ValueError("The report expects separate validation-only research and preserved originals.")
    for relative, expected in protocol["protected_original_files"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Protected baseline evidence changed: {relative}")
    if not data["review"].get("ok"):
        raise ValueError("The independent research review has not passed.")
    for key in ("manifest_sha256", "protocol_sha256"):
        if key in data["review"] and data["review"][key] != study[key]:
            raise ValueError("Independent review belongs to another study.")
    samples = data["audit"]["split_counts"]["validation"]
    for method, metrics in data["baseline"].items():
        checked_metrics(metrics, samples)
    for name, record in study["runs"].items():
        directory = (root / record["artifact_dir"]).resolve()
        if not directory.is_relative_to((root / "artifacts/research").resolve()):
            raise ValueError("Research artifact path is outside its separate research directory.")
        metrics = record["validation_metrics"]
        checked_metrics(metrics, samples)
        if "summary" in record:
            summary = record["summary"]
            local = read_json(directory / "training_summary.json")
            if local != summary or sha256(directory / "best.pt") != summary["artifact_sha256"]:
                raise ValueError(f"Research checkpoint changed: {name}")
            config = read_json(directory / "config.json")
            if config["research_protocol_sha256"] != study["protocol_sha256"]:
                raise ValueError(f"Research run uses a different protocol: {name}")
            if summary["epochs_completed"] != len(read_json(directory / "history.json")):
                raise ValueError(f"Incomplete neural training history: {name}")
            required += [str((directory / item).relative_to(root)).replace("\\", "/")
                         for item in ("config.json", "training_summary.json", "history.json", "validation_metrics.json", "validation_predictions.csv")]
        else:
            local = read_json(directory / "selection.json")
            if local != record["selection"] or sha256(directory / "svm.npz") != local["svm_sha256"]:
                raise ValueError(f"Research SVM changed: {name}")
            if local["research_protocol_sha256"] != study["protocol_sha256"] or local["test_evaluated"]:
                raise ValueError("Classical run differs from the validation-only protocol.")
            required += [str((directory / item).relative_to(root)).replace("\\", "/")
                         for item in ("selection.json", "validation_metrics.json", "validation_predictions.csv")]
    for key in ("baseline", "selected_candidate"):
        if study["seed_confirmation"][key]["seeds_completed"] != 3:
            raise ValueError("Three paired confirmation seeds must be completed.")
    if data["robustness"]["split"] != "validation" or data["errors"]["split"] != "validation":
        raise ValueError("New analysis must use validation only.")
    if set(data["robustness"]["methods"]) != set(METHODS) or set(data["errors"]["methods"]) != set(METHODS):
        raise ValueError("All three methods are required in validation analysis.")
    for method in METHODS:
        conditions = data["robustness"]["methods"][method]["conditions"]
        if set(conditions) != set(data["robustness"]["conditions"]):
            raise ValueError("Some declared perturbations are missing.")
        for condition, record in conditions.items():
            checked_metrics(record["metrics"], samples)
            path = root / "reports/research" / record["predictions_csv"]
            if not path.resolve().is_relative_to((root / "reports/research").resolve()):
                raise ValueError("Invalid robustness prediction path.")
            if sha256(path) != record["predictions_sha256"]:
                raise ValueError("Robustness prediction bytes changed.")
            required.append(path.relative_to(root).as_posix())
    if any(not data[key].get("ok") for key in ("face_smoke", "http_smoke", "benchmark")):
        raise ValueError("Functional smoke tests and measured backend benchmark must pass.")
    if (not data["benchmark"].get("no_concurrent_training")
            or not data["benchmark"]["cleanup"]["server_process_stopped"]):
        raise ValueError("Isolated completed backend timing and server cleanup evidence are required.")
    for key in ("tests", "integration"):
        if data[key]["tests"] == 0 or any(data[key][name] for name in ("failures", "errors", "skipped")):
            raise ValueError("All local verification tests must pass without skips.")
    if data["integration"]["tests"] < 5:
        raise ValueError("Real API/CLI/model integration coverage is incomplete.")
    detector = data["detector"]
    detector_path = root / ".cache/face_detector" / detector["filename"]
    if sha256(detector_path) != detector["sha256"] or sha256(root / ".cache/face_detector/LICENSE") != detector["license_sha256"]:
        raise ValueError("Detector or license bytes changed.")
    if data["pilot"]["status"] != "awaiting_consented_labeled_local_photos" or data["pilot"]["results"] is not None:
        raise ValueError("This report expects an explicitly pending, unperformed local-person pilot.")
    if "not required" not in str(study["colab"]).lower():
        raise ValueError("The latest user-approved Colab scope must be explicit.")
    required += [p.relative_to(root).as_posix() for section in ("src", "scripts", "tests", "configs")
                 for p in (root / section).rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".csv"}]
    required += ["README.md", "pyproject.toml", "requirements-lock-windows-cu128.txt", "requirements-lock-backend-windows-cu128.txt"]
    data["manifest_sha256"] = manifest_sha
    data["source_checksums"] = {name: sha256(root / name) for name in sorted(set(required))}
    return data


def metric_for(data, run):
    return data["study"]["runs"][run]["validation_metrics"]


def selected_screen(data):
    scores = [(metric_for(data, name)["macro_f1"], -index, name) for index, name in enumerate(
        ("visagenet_sampler05_seed42", "visagenet_softmax_seed42"))]
    return max(scores)[2]


def efficiency(data):
    return {method: [(fraction, data["baseline"][method] if fraction == 100 else
                      metric_for(data, f"{method}_data{fraction}_seed42")) for fraction in (25, 50, 100)]
            for method in METHODS}


def generate_plots(root, data):
    cache = root / ".cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = root / "reports/research/figures"
    directory.mkdir(parents=True, exist_ok=True)
    palette = ("#173e65", "#188985", "#b57322")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    runs = ("visagenet_sampler05_seed42", "visagenet_softmax_seed42", "convnext_warmup_seed42")
    names = ("VisAgeNet sampler 0.5", "VisAgeNet softmax", "ConvNeXt optimization")
    fig, ax = plt.subplots(figsize=(8, 3.7))
    values = [metric_for(data, run)["macro_f1"] for run in runs]
    ax.barh(names, values, color=palette)
    ax.axvline(data["baseline"]["visagenet"]["macro_f1"], color=palette[0], linestyle="--", label="Original VisAgeNet")
    ax.set(xlim=(0, 1), xlabel="Validation macro F1", title="Single-seed screens; no held-out selection")
    for i, value in enumerate(values):
        ax.text(value + .01, i, f"{value:.4f}", va="center")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(directory / "screens.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, key in enumerate(("baseline", "selected_candidate")):
        item = data["study"]["seed_confirmation"][key]
        values = item["validation_macro_f1"]
        ax.scatter([i - .08, i, i + .08], values, color=palette[i], zorder=4)
        ax.errorbar(i, item["mean"], yerr=item["sample_std"], fmt="D", color=palette[i], capsize=8)
    ax.set(xticks=(0, 1), xticklabels=("Original VisAgeNet", "Selected VisAgeNet candidate"),
           ylabel="Validation macro F1", title="Three paired seeds: points and mean +/- sample SD")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(directory / "seed_confirmation.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for method, color in zip(METHODS, palette, strict=True):
        items = efficiency(data)[method]
        ax.plot([x for x, _ in items], [m["macro_f1"] for _, m in items], "o-", color=color, label=LABELS[method])
    ax.set(xlabel="Requested training fraction (%)", ylabel="Validation macro F1", xticks=(25, 50, 100),
           title="Nested whole-group data efficiency; seed 42")
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(directory / "data_efficiency.png", dpi=180); plt.close(fig)
    conditions = list(data["robustness"]["conditions"])
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    for offset, method, color in zip((-.24, 0, .24), METHODS, palette, strict=True):
        ax.bar([i + offset for i in range(len(conditions))],
               [data["robustness"]["methods"][method]["conditions"][c]["metrics"]["macro_f1"] for c in conditions],
               width=.23, color=color, label=LABELS[method])
    ax.set(xticks=range(len(conditions)), xticklabels=conditions, ylabel="Validation macro F1", ylim=(0, 1),
           title="Original classifiers under aligned-face perturbations")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(directory / "robustness.png", dpi=180); plt.close(fig)
    return [p for p in sorted(directory.glob("*.png"))]


def percentage(value):
    return "Undefined" if value is None else f"{100 * value:.2f}%"


def number(value, digits=2):
    return f"{float(value):,.{digits}f}"


def build_pdf(root, output, font_dir, data):
    from PIL import Image as PILImage
    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

    for name, file in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")
    styles = {
        "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=24, leading=29, spaceAfter=15),
        "heading": ParagraphStyle("heading", fontName="Arial-Bold", fontSize=19, leading=24, spaceAfter=14),
        "sub": ParagraphStyle("sub", fontName="Arial-Bold", fontSize=12.5, leading=17, spaceBefore=9, spaceAfter=7),
        "body": ParagraphStyle("body", fontName="Arial", fontSize=10.2, leading=14.2, spaceAfter=9),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=8.7, leading=11.4, spaceAfter=7),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.7, leading=11.4),
        "header": ParagraphStyle("header", fontName="Arial-Bold", fontSize=8.7, leading=11.4, textColor=colors.white),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=9, leading=11.5, spaceAfter=10),
    }
    story = []

    def p(text, style="body"):
        story.append(Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style]))

    def page(title, first=False):
        if story: story.append(PageBreak())
        p(title, "title" if first else "heading")

    def table(headings, rows, widths):
        values = [[Paragraph(escape(str(x)).replace("\n", "<br/>"), styles["header"]) for x in headings]]
        values += [[Paragraph(escape(str(x)).replace("\n", "<br/>"), styles["cell"]) for x in row] for row in rows]
        t = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183e65")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f6fa")]),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#ccd3d9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend((t, Spacer(1, 10)))

    def code(text):
        story.append(Preformatted(text, styles["code"]))

    def image(name, max_height=245):
        path = root / name if name.startswith("reports/") else root / "reports/research/figures" / name
        with PILImage.open(path) as im: width, height = im.size
        scale = min(514 / width, max_height / height)
        story.extend((Image(str(path), width=width * scale, height=height * scale), Spacer(1, 9)))

    study, protocol, tests = data["study"], data["protocol"], data["tests"]
    selected = selected_screen(data)
    date = datetime.now(timezone(timedelta(hours=8))).strftime("%d %B %Y")
    page("VisAge Backend and Improvement Report", first=True)
    p("Completed local implementation, validation research and verification", "sub")
    p(f"Prepared {date} | Academic research prototype")
    p("VisAge now accepts an ordinary still photo, detects faces, requires explicit selection when needed, prepares one shared face crop, and returns age-group predictions through a persistent-model backend API. The research study preserves the earlier verified models while measuring controlled alternatives on the unchanged validation partition.")
    table(["Deliverable", "Observed outcome"], [
        ("Face preprocessing", "Pinned official YuNet; deterministic selection; shared five-point RGB crop; bounded admission and quality errors."),
        ("Backend", "Real loopback HTTP checked; all three persistent classifiers; API and both CLI paths agree."),
        ("Research", f"{len(study['runs'])} separate runs; three paired confirmation seeds; common data subsets; fixed corruption and component studies."),
        ("Automated verification", f"{tests['passed']} passed; {tests['failures']} failures; {tests['errors']} errors; {tests['skipped']} skips."),
        ("Local-person pilot", "Protocol, consent manifest validation and runner prepared; participant data not supplied; no pilot results claimed."),
        ("Colab", "Outside the current scope, following the user's explicit decision. All work here is local."),
    ], [135, 379])
    p("Research promotion decision", "sub")
    p(("The selected candidate met every predeclared research promotion criterion. " if study["promotion"]["promote"] else
       "The selected candidate did not meet every predeclared research promotion criterion. ") +
      f"Proposed champion: {study['proposed_champion']}. " + data["deployment"]["reason"])
    p("Completion covers the backend and reproducible research evidence described here. A browser interface, public deployment and independent local-participant accuracy remain separate milestones. An estimated age group does not verify chronological age or transaction eligibility.", "small")

    page("System architecture and completed scope")
    table(["Stage", "Concrete behavior"], [
        ("1 Admission", "Read a bounded raw request body; validate decoded type/dimensions and apply EXIF orientation."),
        ("2 Detection", "Run one pinned CPU YuNet instance at a bounded image size; map boxes and five landmarks to EXIF-oriented image coordinates."),
        ("3 Selection", "One detected face can proceed; multiple detections return explicit indexes for a user decision."),
        ("4 Shared crop", "Use one versioned five-point similarity crop and engineering quality checks; no per-method face selection."),
        ("5 Classification", "Pass that same PIL image object to requested verified artifacts; recorded method-specific resize/normalization follows."),
        ("6 Response/cleanup", "Return groups, coordinate/crop metadata and timing; close image/crop objects on success and failure."),
    ], [125, 389])
    p("The service uses src/visage/faces.py, service.py, api.py and the existing inference adapter. scripts/predict_image.py exercises the same ordinary-photo service outside HTTP. scripts/predict.py remains the adapter for an already aligned face crop. Tests compare both command paths with API output on the identical prepared pixels.")
    p("The three required comparison slots remain intact: a classical NumPy descriptor/SVM method, a randomly initialized custom residual-SE network, and a pretrained ConvNeXt V2 adaptation. The integration adds no numeric-age output or percentage-confidence claim.")
    p("The original model-phase PDF and ZIP are historical artifacts and remain intact. Their previous Colab milestone is superseded by the user's current local-only requirement. Their original held-out comparison is preserved as baseline reporting rather than a target for further tuning.")

    page("Image admission and detector provenance")
    face_config = data["face_smoke"]["detector"]["configuration"]
    table(["Control", "Recorded default"], [
        ("Upload bound", f"{face_config['max_bytes']:,} bytes (8 MiB); stream checked even without Content-Length."),
        ("Decoded images", "Still JPEG, PNG or WebP; RGB; EXIF orientation applied; transparency composited consistently."),
        ("Dimensions", f"At most {face_config['max_pixels']:,} pixels; minimum 32-pixel sides."),
        ("Detection input", f"Maximum side {face_config['max_detection_side']:,}; original coordinate mapping retained."),
        ("Detection settings", f"Score threshold {face_config['score_threshold']}; NMS {face_config['nms_threshold']}; maximum {face_config['max_faces']} faces."),
        ("Execution", "CPU OpenCV FaceDetectorYN, loaded once; mutable detection calls protected by a lock."),
    ], [155, 359])
    detector = data["detector"]
    p("Official detector pin", "sub")
    p(f"Repository: {detector['repository']}\nRevision: {detector['revision']}\nFilename: {detector['filename']}\nSize: {detector['bytes']:,} bytes\nSHA-256: {detector['sha256']}", "small")
    p("scripts/download_face_detector.py verifies the pinned revision's Git LFS SHA/size pointer, retrieves the actual ONNX, and checks the model and license bytes. Ordinary startup verifies those bytes again. No model substitution or silent network download occurs during an inference request.")
    p("The detector directory carries its MIT license from Shiqi Yu. Preserve that notice when distributing the detector. This detector license does not change the separate research-data and age-classifier weight restrictions already documented in the model phase.")

    page("Face selection crop geometry and quality")
    p("Indexes are deterministic left-to-right, then top-to-bottom for the exact submitted image. There is no silent largest-face choice. /v1/faces returns candidates; the caller sends the same photo and chosen index to /v1/predict. A new photo requires a new selection.")
    table(["Policy component", "Actual implementation"], [
        ("Policy identifier", "yunet_training_five_point_v3"),
        ("Geometry", "Orientation-preserving five-point similarity fit to a template from 498 unambiguous faces among 500 fixed training crops; no age labels used."),
        ("Region", "224 x 224 RGB. Canonical eye midpoint near 49% width / 29% height; learned nose/mouth geometry matches the training-crop domain."),
        ("Edge handling", "Reflect101 context outside the source; full context visibility and central-face visibility reported separately."),
        ("Tiny/implausible face", "Minimum 64-pixel face side; finite plausible landmarks; both eyes/mouth inside image; bounded roll and eye separation."),
        ("Quality measurements", "Central-face visibility, grayscale mean, p95-p5 contrast, Laplacian variance, roll and landmark-fit residual."),
        ("Default rejection thresholds", "Central visibility < 0.60; brightness outside 15-245; contrast < 20; gross-blur variance < 1; absolute roll > 40 degrees; landmark RMSE > 10% of crop size."),
    ], [160, 354])
    p("Full validation exposed a serious initial crop-domain error. The original 40%-height, box-expanded policy changed face scale/placement and damaged classification. Training-pixel reconstruction selected the five-point template over an eye-only alternative; their mean absolute RGB errors were 15.17 and 18.44. A training-only quality check also showed the initial blur floor of 10 was too aggressive for ordinary smooth crops; the conservative gross-blur floor is 1.0. Earlier complete/partial diagnostics are retained separately.", "small")
    p(f"Actual functional smoke: {data['face_smoke']['validation_samples']} frozen validation images; {data['face_smoke']['accepted']} accepted. Blank-image rejection and a two-face composite with required explicit selection also passed.", "small")
    p("These thresholds are conservative engineering checks. They do not certify pose, occlusion, identity, fairness, liveness or actual age. The live crop differs from the dataset's original alignment, so full-photo age accuracy still needs the independent local pilot.", "small")

    page("HTTP contract and representative errors")
    table(["Endpoint", "Input / output"], [
        ("GET /health", "Readiness, loaded-model count and configured neural device."),
        ("GET /v1/models", "Methods, default, class vocabulary, common dataset hash, detector and processing policy."),
        ("GET /v1/evaluation", "Read-only aggregate age accuracy/ROC, detection AP/PR and pipeline coverage; records must match served artifacts."),
        ("POST /v1/faces", "Raw image bytes -> EXIF-oriented image size, boxes, landmarks, face indexes and selection requirement."),
        ("POST /v1/predict", "Raw image bytes; method=all or one method; optional nonnegative face_index -> selected face, shared crop metadata, quality, predictions and timings."),
    ], [145, 369])
    p("Content-Type must identify JPEG, PNG, WebP or generic binary; the decoded image format is checked independently. The request body is raw bytes, not multipart data. This keeps bounded upload handling in memory and avoids an implicit spooled upload file.")
    table(["Status", "Error examples", "Caller action"], [
        ("400", "EMPTY_IMAGE, INVALID_IMAGE, INVALID_FACE_INDEX", "Choose readable image data or a returned face index."),
        ("408", "UPLOAD_TIMEOUT", "Retry a smaller image or improve upload connection."),
        ("413", "IMAGE_TOO_LARGE, IMAGE_DIMENSIONS_TOO_LARGE", "Reduce byte size or resolution."),
        ("415", "UNSUPPORTED_MEDIA_TYPE / IMAGE / CONTENT_ENCODING", "Send an admitted still format as uncompressed request bytes."),
        ("422", "NO_FACE, MULTIPLE_FACES, FACE_TOO_SMALL, UNUSABLE_FACE, INVALID_REQUEST", "Show a clear face or explicitly select one."),
        ("503", "NOT_READY, SERVER_BUSY, QUEUE_TIMEOUT", "Wait for readiness or retry later."),
        ("500", "INFERENCE_FAILED", "Generic operator-facing failure; no internal traceback returned."),
    ], [50, 244, 220])
    p("Success contains method/class_id/age_group plus inference-and-preprocessing milliseconds. The response also records decode, detection, alignment/quality, classifiers, upload, queue and request-total timing. Error envelopes contain code, message, details and a request ID; age confidence percentages are not returned.", "small")

    page("Model lifetime concurrency and image cleanup")
    p("A server process loads three checksum-verified classifiers and the detector during lifespan startup. A common dataset-freeze guard rejects mixed model sets. Health cannot report ready until startup succeeds. The launcher uses one worker so it does not accidentally duplicate the laptop's model set.")
    table(["Operational concern", "Implemented behavior"], [
        ("Admission", "Default one active plus two pending requests; excess traffic receives SERVER_BUSY before an upload is read."),
        ("Bounded waits", "Default 15-second upload and queue timeouts; content-length and streamed-byte limits."),
        ("Model serialization", "One API work lock protects the full face/preprocessing/classifier operation; native inference executes outside the async event loop."),
        ("Cancellation", "An executor future is shielded; cancellation waits for active native work to complete before the mutable model/detector lock is released."),
        ("Images", "No upload/photo output directory or training-data insertion; input buffers released and PIL source/crop images closed in finally blocks."),
        ("Responses/logs", "No-store and nosniff headers; request IDs; no upload echo or internal stack trace; launcher access logs disabled."),
        ("Browser access", "No origins allowed by default; explicit --allow-origin configuration supports a later local frontend."),
    ], [145, 369])
    p("The local HTTP smoke used real Uvicorn loopback traffic rather than only mocked request handlers. Cancellation, streamed oversize input, upload timeout, bounded queue, structured errors, startup reuse and cleanup are also exercised by focused tests.")
    p("Memory-only processing means the application does not intentionally save photos. Python cannot promise secure physical-memory erasure or control external OS/proxy caching. This local prototype has no public authentication or production reverse-proxy configuration; such deployment work is outside this phase.", "small")

    page("Verification and measured backend latency")
    table(["Evidence", "Result"], [
        ("Final automated suite", f"{tests['tests']} tests: {tests['passed']} passed, {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skips."),
        ("Real-artifact integration", f"{data['integration']['passed']} passed: shared pixels/direct-adapter parity, both CLIs, EXIF and selection/error paths."),
        ("Real HTTP smoke", data["http_smoke"]["transport"] + "; all three models ready on CPU."),
        ("Actual CPU/CUDA check", f"{data['cuda_smoke']['samples']} fixed validation inputs; shared crop hashes and predicted labels agree across devices."),
        ("Source image scope", "Frozen validation images only; no new held-out image inference."),
        ("Independent research review", "Passed; numerical results, common splits, protected originals and research artifact hashes checked."),
    ], [155, 359])
    p("Measured backend benchmark", "sub")
    benchmark = data["benchmark"]
    timing_rows = [("HTTP round trip", benchmark["http_round_trip"])]
    timing_rows += [("Server " + key.replace("_", " "), benchmark["server_timings"][key])
                    for key in ("processing_total", "detection", "alignment_and_quality")]
    timing_rows += [(LABELS[method] + " preprocessing/inference", benchmark["method_preprocessing_and_inference"][method])
                    for method in METHODS]
    table(["Measured scope", "Median ms", "p95 ms"],
          [(label, number(item["median_ms"]), number(item["p95_ms"])) for label, item in timing_rows],
          [320, 97, 97])
    environment = benchmark["environment"]
    p(f"CPU device; {environment['api_torch_threads']} Torch threads. Startup {number(benchmark['startup_seconds'])} seconds; "
      f"{benchmark['warmups']} warm-ups and {benchmark['iterations']} measured requests. "
      f"Platform: {environment['platform']}. Processor: {environment.get('processor', 'recorded in JSON')}.", "small")
    p("Read timing scope and measurement counts alongside each result. A repeated local validation image measures this machine/path, not production load or camera-domain accuracy. Warm-up/startup are separate; no simultaneous training should contaminate latency measurements.")
    p("Managed Windows sandbox note: asyncio socket-pair/loopback creation stalled before API code under the restricted environment. The actual HTTP and TestClient checks were rerun with loopback enabled; this is an execution-environment constraint rather than a hidden application failure.", "small")

    page("Predeclared research design and protected data")
    p(protocol["study_name"])
    p("The study uses the same 16,300 training and 3,492 validation images from the frozen manifest. Whole training groups are retained in nested approximate 25%/50% subsets; all three methods use the same subset IDs. Validation membership/order and class counts stay fixed.")
    table(["Study component", "Controlled comparison"], [
        ("VisAgeNet sampling", "Inverse-frequency exponent 0.5 versus original full inverse-frequency sampling; original CORN head and remaining settings."),
        ("VisAgeNet head", "Three-class softmax/cross-entropy versus two-logit CORN; original architecture and balanced sampling."),
        ("ConvNeXt optimization", "Backbone LR 3e-5 plus two-epoch unfreeze warm-up versus original LR 1e-4; combined exploratory candidate."),
        ("Paired-seed confirmation", "Seed 42 baseline reused; baseline and selected VisAgeNet candidate repeated at seeds 43 and 44."),
        ("Data efficiency", "25%/50% whole-group train subsets, original budgets/configurations and seed 42; original 100% records reused."),
        ("Classical components", "HOG-only and LBP-only, fixed original selected SVM settings, versus original combined descriptors."),
        ("Perturbations", "Fixed original models on all validation aligned crops; declared deterministic image corruptions."),
    ], [155, 359])
    p(f"Protocol SHA-256: {study['protocol_sha256']}\nManifest SHA-256: {data['manifest_sha256']}", "small")
    p("Original checkpoints and held-out result files are checksum-protected. Every new run has a separate output/config/log; no original model is overwritten. The study records zero new test evaluations. Existing test scores were already observed and are preserved as historical reporting. New independent participants are needed for fresh confirmation.", "small")

    page("Screening results and exploratory modern candidate")
    screen_rows = []
    for name, label in (("visagenet_sampler05_seed42", "VisAgeNet sampling 0.5"),
                        ("visagenet_softmax_seed42", "VisAgeNet softmax"),
                        ("convnext_warmup_seed42", "ConvNeXt lower LR + warm-up")):
        m = metric_for(data, name)
        s = study["runs"][name]["summary"]
        screen_rows.append((label, number(m["macro_f1"], 4), percentage(m["accuracy"]),
                            str(s["best_epoch"]), number(s["training_seconds"] / 60)))
    table(["Single-seed candidate", "Macro F1", "Accuracy", "Best epoch", "Train min"],
          screen_rows, [210, 65, 75, 70, 94])
    image("screens.png", 225)
    p(f"Selected VisAgeNet screen: {selected}. Selection uses seed 42 validation macro F1; exact ties follow the predeclared order. This screen is then checked across paired seeds even if its initial gain is small or negative.")
    p("The ConvNeXt candidate changes two optimization components together. Its result cannot isolate the effect of learning rate from warm-up. It is a single-seed exploratory measurement and cannot replace the champion without a separate repeat-seed confirmation. More epochs or a larger backbone are not assumed to improve deployment behavior.")
    p("Training minutes are measured successful training/data/augmentation/validation epochs. Startup, preflight, cache/download work and checkpoint serialization are separate. A single timing is hardware/path dependent.", "small")

    page("Three-seed confirmation and promotion decision")
    seeds = study["seed_confirmation"]
    table(["Configuration", "Seed 42", "Seed 43", "Seed 44", "Mean", "Sample SD"], [
        (label, *[number(x, 4) for x in seeds[key]["validation_macro_f1"]],
         number(seeds[key]["mean"], 4), number(seeds[key]["sample_std"], 4))
        for key, label in (("baseline", "Original VisAgeNet"), ("selected_candidate", "Selected candidate"))
    ], [140, 68, 68, 68, 85, 85])
    image("seed_confirmation.png", 215)
    promotion = study["promotion"]
    check_labels = {
        "mean_macro_f1_improvement_at_least_0_005": "Mean macro F1 gain >= 0.005",
        "minor_error_increase_at_most_0_01": "Under-18-to-older error increase <= 1 percentage point",
        "senior_recall_drop_at_most_0_01": "Senior recall drop <= 1 percentage point",
    }
    table(["Predeclared check", "Outcome"], [(check_labels[name], "PASS" if value else "FAIL")
          for name, value in promotion["checks"].items()], [420, 94])
    p("Promotion result: " + ("QUALIFIED" if promotion["promote"] else "BASELINE RETAINED") +
      ". " + promotion["reason"])
    p("Three seeds characterize training variation on this fixed split; they do not establish population confidence or identity-independent generalization. The screen was chosen using the same validation set, so repeated selection remains a limitation. A future qualified candidate must also pass identical-crop, error-rate, latency and functional checks before local activation. The originals remain intact.", "small")

    page("Data efficiency with shared whole-group subsets")
    rows = []
    for method, items in efficiency(data).items():
        for fraction, m in items:
            size = 16300 if fraction == 100 else protocol["subsets"][str(fraction)]["rows"]
            rows.append((LABELS[method], f"{fraction}%", f"{size:,}", percentage(m["accuracy"]), number(m["macro_f1"], 4)))
    table(["Method", "Train fraction", "Actual rows", "Val accuracy", "Val macro F1"], rows, [150, 88, 88, 94, 94])
    image("data_efficiency.png", 235)
    p("Fractions are approximate because connected perceptual groups cannot be divided. Subsets are nested, class-aware whole-group prefixes chosen before training. All methods share the same sample IDs. No image moved from validation/test into training.")
    p("These runs measure performance at the original fixed epoch budgets and one seed. They are not estimates of optimal tuning for each data size. The original 100% runs are reused and are not counted as new research training.", "small")

    page("Component ablations and what they establish")
    rows = [("Combined original HOG + LBP", number(data["baseline"]["classical"]["macro_f1"], 4),
             percentage(data["baseline"]["classical"]["accuracy"]))]
    for name, label in (("classical_hog_seed42", "HOG only"), ("classical_lbp_seed42", "Multiscale LBP only")):
        m = metric_for(data, name)
        rows.append((label, number(m["macro_f1"], 4), percentage(m["accuracy"])))
    table(["Classical descriptors", "Val macro F1", "Val accuracy"], rows, [300, 107, 107])
    p("Each component fit uses the original selected learning-rate/regularization configuration and class-balanced SVM weighting for 60 epochs. Feature scaling is refit on training rows. No new grid search is performed; reduced descriptor dimensions and their own train-only scaler are part of each declared component model.")
    rows = [("Original CORN", number(data["baseline"]["visagenet"]["macro_f1"], 4),
             percentage(data["baseline"]["visagenet"]["per_class"][2]["recall"]))]
    softmax = metric_for(data, "visagenet_softmax_seed42")
    rows.append(("Three-class softmax", number(softmax["macro_f1"], 4), percentage(softmax["per_class"][2]["recall"])))
    table(["VisAgeNet output formulation", "Val macro F1", "Senior recall"], rows, [300, 107, 107])
    p("CORN models learn conditional ordered boundaries; softmax treats the three classes as mutually exclusive categories. This comparison changes the output/loss formulation while retaining the backbone, sampling and other stated settings. One seed is descriptive; the selected candidate's paired-seed confirmation provides stronger evidence only for that selected configuration.")
    p("A component result explains behavior under the fixed experimental budget. It does not establish universal superiority of a descriptor, attention gate or ordinal formulation. No architecture novelty beyond the documented combination of established components is claimed.", "small")

    page("Robustness under declared crop perturbations")
    rows = []
    for condition in data["robustness"]["conditions"]:
        rows.append((condition, *[number(data["robustness"]["methods"][m]["conditions"][condition]["metrics"]["macro_f1"], 4)
                                  for m in METHODS]))
    table(["Condition", "Classical F1", "VisAgeNet F1", "ConvNeXt F1"], rows, [160, 118, 118, 118])
    image("robustness.png", 240)
    p("All conditions use the same 3,492 validation crop rows. Conditions include brightness factors 0.65/1.35, Gaussian blur radius 1.5, JPEG quality 35, 56-pixel intermediate resolution, and a 5% right translation with black padding. These are declared stress conditions; their labels remain source annotations.")
    worst = []
    for method in METHODS:
        conditions = data["robustness"]["methods"][method]["conditions"]
        name = min((key for key in conditions if key != "clean"), key=lambda key: conditions[key]["metrics"]["macro_f1"])
        loss = conditions["clean"]["metrics"]["macro_f1"] - conditions[name]["metrics"]["macro_f1"]
        worst.append(f"{LABELS[method]}: {name.replace('_', ' ')} ({loss:.4f} F1 loss)")
    p("Largest measured losses: " + "; ".join(worst) + ". The classical method's crop sensitivity reinforces the need for consistent face geometry.", "small")
    p("The study evaluates fixed original classifiers after aligned-crop perturbations. It does not evaluate the face detector on ordinary scenes, estimate real-world failure rates or prove camera robustness. Corruption-specific losses should guide later development hypotheses rather than retrospective held-out tuning.", "small")

    page("Validation error strata and pending local pilot")
    rows = []
    band_labels = {"below15": "Below 15", "15_to20": "15-20", "21_to54": "21-54", "55_to64": "55-64", "65plus": "65+"}
    for band in ("below15", "15_to20", "21_to54", "55_to64", "65plus"):
        parts = [data["errors"]["methods"][m]["age_band"][band] for m in METHODS]
        rows.append((band_labels[band], str(parts[0]["samples"]), *[percentage(x["accuracy"]) for x in parts]))
    table(["Provided age band", "Samples", "Classical acc", "VisAgeNet acc", "ConvNeXt acc"], rows, [160, 60, 98, 98, 98])
    p("The companion JSON also reports brightness/sharpness strata with supports, errors and accuracy. These variables are correlated with age, class mixture and source quality; the descriptive bins are not validated quality thresholds or causal explanations. No source label is automatically corrected.")
    p("Local pilot status: pending participant data", "sub")
    p("Participant data have not been supplied. The pilot protocol and runner are prepared; no images, labels or outcomes were invented. configs/local_pilot_protocol.json and local_pilot_labels_template.csv define collection and bookkeeping; scripts/run_local_pilot.py validates all rows before any inference.")
    p("Assign pseudonymous participants to development/holdout in advance; keep all their photos in one split. Record voluntary permission and reliable age groups outside the repository. Do not store names, birthdays or identity documents in the CSV. Holdout must contain all three groups; a recommended target is at least 30 distinct participants per group, with realistic lighting/devices/pose variation.")
    p("The runner checks consent flags, paths, file bounds/hashes, duplicate bytes, labels and participant separation. Rejections remain in attempted-image coverage and end-to-end correct-rate denominators; conditional classification metrics on accepted images are reported separately. Pseudonymous IDs do not independently prove identity, chronological age or valid permission.", "small")

    page("Reproduction runbook and supervision artifacts")
    p("Start the verified local service", "sub")
    code('.\\.venv\\Scripts\\python.exe scripts/download_face_detector.py\n'
         '.\\.venv\\Scripts\\python.exe scripts/serve_api.py --device cpu\n'
         "$imagePath = 'C:\\path\\photo.jpg'\n"
         "$endpoint = 'http://127.0.0.1:8000/v1/predict?method=all'\n"
         'curl.exe -H "Content-Type: image/jpeg" --data-binary "@$imagePath" $endpoint')
    p("Run the server in one terminal and curl in another. POST the identical image with face_index when multiple detections require selection.", "small")
    p("Research and verification commands", "sub")
    code('.\\.venv\\Scripts\\python.exe scripts/run_research_study.py --stage all\n'
         '.\\.venv\\Scripts\\python.exe scripts/research_robustness.py --stage merge\n'
         '.\\.venv\\Scripts\\python.exe -m pytest -q\n'
         '.\\.venv\\Scripts\\python.exe -m ruff check src scripts tests')
    table(["Location", "Purpose"], [
        ("reports/research/protocol.json", "Immutable configuration/selection rules and protected baseline hashes."),
        ("reports/research/study_summary.json", "Actual run metrics, seed confirmation and proposed promotion."),
        ("reports/research/robustness.json", "All declared corruption measurements and prediction hashes."),
        ("reports/research/error_analysis.json", "Validation age/brightness/sharpness descriptive strata."),
        ("reports/backend_benchmark.json", "Measured full backend timing scope and hardware."),
        ("artifacts/research/", "Separate new checkpoints, configs, histories and validation predictions."),
        ("reports/tests_backend_research.xml", "Final full-suite machine-readable verification."),
    ], [245, 269])
    p("Research stages reuse matching completed runs and reject changed immutable configuration. Run one GPU job at a time. Local pilot commands require real permitted files and a fresh output directory; validate-only performs no age inference.", "small")
    activation = ("A candidate was activated locally." if data["deployment"]["status"] == "active_candidate" else "No candidate was activated.")
    p("Local model selection: " + data["deployment"]["reason"] + " The decision and evidence hashes are recorded in reports/deployment/. " + activation + " No public hosting was performed.", "small")

    page("Limitations remaining work and source records")
    p("The completed backend has functional evidence on this local machine and frozen validation inputs. It is not a finished browser/camera application or public production deployment. Next work is an interface that previews detections, obtains explicit face selection, calls this API, handles errors and displays all three age-group estimates without certainty claims.")
    p("Model results compare provided UTKFace labels, not verified birthdays. No subject IDs exist, so the perceptual grouping does not prove identity-disjoint splits. The provided data are imbalanced, the new live alignment changes the input domain, and the local-person pilot remains unperformed. Research screening/repeats share one validation set; no population or statistically definitive superiority claim follows.")
    p("The original test comparison is already observed and preserved. Further test-set reuse would be exploratory. Fresh external confirmation needs new frozen participants/data. Colab is not a missing deliverable under the latest user-approved scope.")
    p("The prototype does not verify identity, liveness, chronological age, purchase eligibility or discount entitlement. Persistent photo storage, authentication, public hosting, operational monitoring and policy decisions require explicit implementation in their appropriate later phase. The current backend processes uploads in memory without retaining photos.")
    p("Primary sources and local evidence", "sub")
    for label, url in (
        ("Official YuNet source and model directory", "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"),
        ("OpenCV FaceDetectorYN API", "https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html"),
        ("CORN ordinal formulation", "https://arxiv.org/abs/2111.08851"),
        ("ConvNeXt V2 training reference", "https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md"),
        ("UTKFace author source", "https://susanqq.github.io/UTKFace/"),
        ("WIDER FACE benchmark source", "http://shuoyang1213.me/WIDERFACE/"),
    ):
        p(label + ": " + url, "small")
    p("The report-build record records checksums of every required evidence record plus implementation/config/test sources. It rechecks them after generation and requires later visual inspection. Substantial implementation and documentation were drafted with Codex assistance; group members must review, understand, test, explain and disclose that contribution accurately.", "small")

    page("Full validation face-pipeline diagnostic")
    diagnostic = data["face_validation"]
    p(f"The production detector, selection, crop and quality path processed all {diagnostic['attempted']:,} original validation crops. It accepted {diagnostic['accepted']:,} and rejected {diagnostic['rejected']:,}; coverage was {percentage(diagnostic['coverage'])}. Rejected inputs remain in the attempted-image denominator.")
    table(["Method", "New crop F1", "Same accepted original F1", "End-to-end correct"], [
        (LABELS[method], number(diagnostic['methods'][method]['accepted_image_metrics']['macro_f1'], 4),
         number(diagnostic['methods'][method]['original_aligned_metrics_same_accepted_images']['macro_f1'], 4),
         percentage(diagnostic['methods'][method]['end_to_end_correct_rate'])) for method in METHODS
    ], [170, 105, 134, 105])
    table(["Rejection code", "Images"], sorted(diagnostic['rejection_codes'].items()), [360, 154])
    p("The comparison holds the accepted subset fixed to distinguish crop changes from image filtering. These are already aligned UTKFace crops being redetected and realigned, not ordinary camera photos or independent participant data. This diagnostic neither changes the deployed crop automatically nor establishes detector AP.")
    p("Operating quality limits are engineering defaults. The independent local pilot must validate ordinary-photo detection, crop geometry, rejection coverage and age-group accuracy together. Model age estimates remain advisory.", "small")

    page("Age-group accuracy and ROC curves")
    roc = data["classification_roc"]
    p(f"The three currently served age-group artifacts are evaluated using their saved {roc['samples']:,} validation labels and scores. Each ROC curve treats one age group as positive and the other two as negative. SVM margins and neural class values are uncalibrated scores; AUC measures ranking, not certainty for a person.")
    table(["Method", "Accuracy percent", "Mean one-vs-rest AUC"], [
        (LABELS[method], number(value['accuracy_percent']), number(value['roc']['macro_auc_mean'], 4))
        for method, value in roc['methods'].items()
    ], [264, 125, 125])
    image("reports/extended_evaluation/age_group_roc.png", 245)
    p("AUC is computed from score-threshold ROC points with tied scores grouped together; the macro value is the arithmetic mean of the three class AUCs. Independent rank-sum arithmetic reproduces every class AUC and saved-label accuracy.")
    p("Accuracy here means age-group classification. Identity recognition is outside VisAge's task; the system does not identify a person or claim identity-recognition accuracy. The curves add evaluation evidence without new age-test inference or test-driven selection.", "small")

    page("Detection AP and precision-recall evidence")
    detection = data["detection_ap"]
    p(f"The pinned YuNet detector processed all {detection['images']:,} WIDER FACE validation images using independently supplied bounding-box annotations. Dataset provenance records the CUHK-hosted revision and archive SHA-256 checksums. No model-generated boxes were used as ground truth.")
    table(["Detection metric", "AP percent", "Valid ground-truth faces"], [
        (name, number(value['ap_percent']), f"{value['ground_truth_faces']:,}") for name, value in detection['metrics'].items()
    ], [200, 130, 184])
    image("reports/extended_evaluation/detection_ap_pr_froc.png", 245)
    p(detection['protocol_note'])
    config = detection['config']
    p(f"AP confidence floor {config['score_threshold']} bounds the sweep and achievable recall; the serving API uses {diagnostic['face_config']['score_threshold']}. NMS {config['nms_threshold']}; maximum image side {config['max_detection_side']:,}; up to {config['max_faces']:,} detections. Crop-quality and age-selection rules are excluded from detector AP.", "small")
    p("AP uses global score ranking, one-to-one box matching and the all-points precision envelope. Invalid boxes are excluded through the declared ignore policy; duplicate valid hits are false positives. Independent scalar IoU/matching arithmetic verifies the reported AP values. The FROC plot shows recall against false positives per image; age-group ROC is reported separately.", "small")
    p("The current detector also exactly reproduced saved boxes and scores on twelve fixed WIDER validation inputs after crop calibration changed. Crop and age-quality rules are separate from this box-detection check.", "small")
    p("WIDER validation is a development face-detection benchmark and does not replace the pending independent local-person pilot. Research images are excluded from the delivery package.", "small")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.pdf")

    def furniture(canvas, document):
        canvas.saveState()
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#444444"))
        if document.page > 1: canvas.drawString(49, 766, "VisAge backend and validation-only improvement study")
        canvas.drawString(49, 27, f"Local research prototype | Actual completed evidence | {date}")
        canvas.drawRightString(563, 27, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(temporary), pagesize=letter, leftMargin=49, rightMargin=49,
                                 topMargin=47, bottomMargin=44, title="VisAge Backend and Improvement Report",
                                 author="VisAge project", subject="Local API implementation and validation-only research")
    document.build(story, onFirstPage=furniture, onLaterPages=furniture)
    reader = PdfReader(temporary)
    if len(reader.pages) != EXPECTED_PAGES:
        raise ValueError(f"Report produced {len(reader.pages)} pages instead of {EXPECTED_PAGES}; inspect {temporary} and correct layout.")
    for path, expected in data["source_checksums"].items():
        if sha256(root / path) != expected:
            raise ValueError(f"Evidence changed during report build: {path}")
    temporary.replace(output)
    summary = {"output": output.relative_to(root).as_posix(), "pages": len(reader.pages),
               "sha256": sha256(output), "manifest_sha256": data["manifest_sha256"],
               "protocol_sha256": study["protocol_sha256"], "all_required_evidence_validated": True,
               "local_pilot_status": "pending_participant_data", "colab": "not required",
               "evidence_checksums": data["source_checksums"], "visual_qa_required": True,
               "page_word_counts": [len((p.extract_text() or "").split()) for p in reader.pages],
               "generated_utc": datetime.now(timezone.utc).isoformat()}
    (root / "reports/backend_research_report_build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/VisAge_Backend_and_Improvement_Report.pdf")
    parser.add_argument("--font-dir", type=Path, default=Path(r"C:\Windows\Fonts"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not output.is_relative_to(root): parser.error("Report output must stay inside the workspace.")
    try:
        data = load_evidence(root)
        if args.check:
            print(json.dumps({"ready": True, "runs": len(data["study"]["runs"]), "tests": data["tests"]}, indent=2))
        elif args.plots_only:
            print("Generated:", *generate_plots(root, data), sep="\n")
        else:
            python = root / ".venv/Scripts/python.exe"
            subprocess.run([str(python), str(Path(__file__).resolve()), "--root", str(root), "--plots-only"],
                           cwd=root, check=True, timeout=120)
            for file in (root / "reports/research/figures").glob("*.png"):
                data["source_checksums"][file.relative_to(root).as_posix()] = sha256(file)
            print(json.dumps(build_pdf(root, output, args.font_dir, data), indent=2))
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

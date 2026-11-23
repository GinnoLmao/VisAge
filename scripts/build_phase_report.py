"""Build the factual phases 2-6 implementation report from completed evidence.

Run with the bundled document Python runtime, which provides ReportLab/pypdf.
No model code is imported and no training/evaluation is started by this builder.
Missing or unsuccessful required evidence stops generation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("classical", "visagenet", "convnext_ordinal")
NAMES = {
    "classical": "A Classical LBP and HOG SVM",
    "visagenet": "B VisAgeNet",
    "convnext_ordinal": "C ConvNeXt V2 ordinal",
}
EXPECTED_PAGES = 14
COURSE_SOURCE = r"C:\Users\USER\Downloads\Projects final.pdf"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(root: Path, relative: str) -> dict | list:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list)):
        raise ValueError(f"Expected structured evidence: {relative}")
    return value


def tests_summary(path: Path) -> dict:
    tree = ET.parse(path)
    cases = tree.findall(".//testcase")
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    suites = [suite for suite in tree.iter("testsuite") if not suite.findall("testsuite")]
    return {
        "tests": len(cases), "passed": len(cases) - failures - errors - skipped,
        "failures": failures, "errors": errors, "skipped": skipped,
        "seconds": sum(float(suite.attrib.get("time", 0)) for suite in suites),
        "groups": sorted({case.attrib.get("classname", "").split(".")[-1] for case in cases}),
    }


def load_evidence(root: Path) -> dict:
    required = [
        "reports/environment.json", "reports/gpu_pilot.json",
        "reports/gpu_pilot_batch64.json", "reports/tests_phase.xml",
        "reports/tensor_cache.json",
        "data/provenance.json", "data/processed/audit.json",
        "data/processed/manifest.csv", "data/processed/manifest.freeze.json",
        "data/processed/excluded.csv", "reports/data_audit_review.json",
        ".cache/weights/pretrained_provenance.json",
        "artifacts/classical/selection.json", "artifacts/classical/training_config.json",
        "artifacts/classical/validation_metrics.json", "artifacts/classical/history.json",
        "artifacts/classical/features.json", "artifacts/classical/svm.npz",
        "reports/evaluation/comparison.json", "reports/artifact_verification.json",
        "reports/notebook_runner.json", "reports/notebook_execution.json",
        "notebooks/visage_phase_models_executed.ipynb",
    ]
    for method in ("visagenet", "convnext_ordinal"):
        required.extend(f"artifacts/{method}/{name}" for name in (
            "config.json", "training_summary.json", "validation_metrics.json",
            "history.json", "best.pt",
        ))
        required.append(f"reports/evaluation/{method}_training_curves.png")
    required.extend(f"reports/evaluation/{method}_confusion.png" for method in METHODS)
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required completed evidence is missing; report not generated:\n" + "\n".join(missing)
        )
    data = {
        "environment": read_json(root, "reports/environment.json"),
        "tensor_cache": read_json(root, "reports/tensor_cache.json"),
        "pilots": [read_json(root, name) for name in (
            "reports/gpu_pilot.json", "reports/gpu_pilot_batch64.json")],
        "tests": tests_summary(root / "reports/tests_phase.xml"),
        "provenance": read_json(root, "data/provenance.json"),
        "audit": read_json(root, "data/processed/audit.json"),
        "freeze": read_json(root, "data/processed/manifest.freeze.json"),
        "data_review": read_json(root, "reports/data_audit_review.json"),
        "weights": read_json(root, ".cache/weights/pretrained_provenance.json"),
        "classical_selection": read_json(root, "artifacts/classical/selection.json"),
        "classical_config": read_json(root, "artifacts/classical/training_config.json"),
        "classical_validation": read_json(root, "artifacts/classical/validation_metrics.json"),
        "classical_history": read_json(root, "artifacts/classical/history.json"),
        "comparison": read_json(root, "reports/evaluation/comparison.json"),
        "verification": read_json(root, "reports/artifact_verification.json"),
        "runner": read_json(root, "reports/notebook_runner.json"),
        "notebook_execution": read_json(root, "reports/notebook_execution.json"),
        "neural": {
            method: {
                "config": read_json(root, f"artifacts/{method}/config.json"),
                "summary": read_json(root, f"artifacts/{method}/training_summary.json"),
                "validation": read_json(root, f"artifacts/{method}/validation_metrics.json"),
                "history": read_json(root, f"artifacts/{method}/history.json"),
            } for method in ("visagenet", "convnext_ordinal")
        },
    }
    manifest_hash = sha256(root / "data/processed/manifest.csv")
    if data["freeze"]["manifest_sha256"] != manifest_hash:
        raise ValueError("Current manifest differs from its recorded freeze.")
    for name, expected in data["freeze"]["checksums"].items():
        if sha256(root / "data/processed" / name) != expected:
            raise ValueError(f"Current {name} differs from the frozen checksum.")
    if data["data_review"]["status"] != "passed" or not all(data["data_review"]["checks"].values()):
        raise ValueError("Independent data review did not pass.")
    for evidence in (
        data["data_review"], data["classical_selection"], data["comparison"], data["verification"]
    ):
        if evidence["manifest_sha256"] != manifest_hash:
            raise ValueError("Evidence records do not use the same frozen manifest.")
    if data["comparison"]["split"] != "test":
        raise ValueError("The primary comparison must be the completed held-out test comparison.")
    if set(data["comparison"]["methods"]) != set(METHODS):
        raise ValueError("The primary comparison must include all three methods.")
    test_count = data["audit"]["split_counts"]["test"]
    for method in METHODS:
        result = data["comparison"]["methods"][method]
        if result["metrics"]["samples"] != test_count:
            raise ValueError(f"{method} test count differs from the frozen test split.")
        matrix = result["metrics"]["confusion_matrix"]
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("Expected a three-class confusion matrix.")
        if sum(sum(row) for row in matrix) != test_count:
            raise ValueError("Confusion matrix does not cover the full test partition.")
    if not data["environment"]["cuda_available"]:
        raise ValueError("CUDA environment evidence is missing.")
    if data["tests"]["tests"] < 1 or data["tests"]["failures"] or data["tests"]["errors"]:
        raise ValueError("Required automated checks did not pass.")
    if data["verification"]["ok"] is not True or set(data["verification"]["methods"]) != set(METHODS):
        raise ValueError("Independent reload verification is incomplete.")
    for method in METHODS:
        check = data["verification"]["methods"][method]
        if check["ok"] is not True or check["validation_rows_checked"] != data["audit"]["split_counts"]["validation"]:
            raise ValueError(f"{method} saved validation evidence was not independently checked.")
        path = f"artifacts/{method}/svm.npz" if method == "classical" else f"artifacts/{method}/best.pt"
        if sha256(root / path) != check["artifact_sha256"]:
            raise ValueError(f"{method} artifact differs from independently verified bytes.")
    for method in ("visagenet", "convnext_ordinal"):
        result = data["neural"][method]
        for part in ("config", "summary"):
            if result[part]["manifest_sha256"] != manifest_hash:
                raise ValueError(f"{method} metadata differs from the frozen manifest.")
        if result["summary"]["epochs_completed"] != len(result["history"]):
            raise ValueError(f"{method} training history is incomplete.")
        if sha256(root / f"artifacts/{method}/best.pt") != result["summary"]["artifact_sha256"]:
            raise ValueError(f"{method} selected checkpoint checksum changed.")
    runner = data["runner"]
    if runner["status"] != "complete" or runner["cell_errors"] != 0:
        raise ValueError("The artifact-only notebook did not complete successfully.")
    if runner["executed_code_cells"] != runner["code_cells"]:
        raise ValueError("Some notebook code cells were not executed.")
    if runner["execution_host"] != "local" or runner["mode"] != "artifacts":
        raise ValueError("This report expects the recorded local artifact-only notebook verification.")
    if runner.get("execution_engine") not in {"python_cells", "nbclient"}:
        raise ValueError("Notebook execution engine must be explicitly recorded.")
    if runner.get("execution_engine") == "python_cells" and runner.get("jupyter_kernel_executed"):
        raise ValueError("Plain Python cell execution must not claim a Jupyter kernel run.")
    execution = data["notebook_execution"]
    if execution["manifest_sha256"] != manifest_hash or not execution["all_artifacts_reloaded"]:
        raise ValueError("Notebook execution evidence does not match the completed model artifacts.")
    notebook = read_json(root, "notebooks/visage_phase_models_executed.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(code_cells) != runner["code_cells"] or any(cell["execution_count"] is None for cell in code_cells):
        raise ValueError("Executed notebook metadata does not match the runner record.")
    if any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])):
        raise ValueError("An error output remains in the executed notebook.")
    if execution.get("colab_execution_claim") or runner.get("colab_verified"):
        raise ValueError("Local notebook evidence must not be labeled Colab verification.")
    cache = data["tensor_cache"]
    if (cache["complete"] is not True or cache["manifest_sha256"] != manifest_hash
            or cache["rows"] != data["audit"]["split_counts"]["train"] + data["audit"]["split_counts"]["validation"]
            or cache["completed_rows"] != cache["rows"]):
        raise ValueError("Train/validation tensor cache evidence is incomplete or mismatched.")
    weights = data["weights"]
    if sha256(root / weights["relative_path"]) != weights["sha256"]:
        raise ValueError("Pretrained starting file checksum changed.")
    modern_source = data["neural"]["convnext_ordinal"]["config"].get("pretrained_provenance")
    if not modern_source or any(modern_source[key] != weights[key] for key in ("revision", "sha256", "repository")):
        raise ValueError("Modern training did not record the expected pretrained source.")
    data["manifest_sha256"] = manifest_hash
    data["source_checksums"] = {name: sha256(root / name) for name in required if not name.endswith(".pt")}
    return data


def number(value, digits=2) -> str:
    return f"{float(value):,.{digits}f}"


def percent(value) -> str:
    return "Not defined" if value is None else f"{100 * float(value):.2f}%"


def add_font(font_dir: Path) -> None:
    for name, filename in (
        ("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf")
    ):
        path = font_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required report font missing: {path}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                                 italic="Arial-Italic", boldItalic="Arial-Bold")


class Report:
    def __init__(self, root: Path, data: dict):
        self.root, self.data = root, data
        self.story = []
        self.styles = {
            "body": ParagraphStyle("Body", fontName="Arial", fontSize=11, leading=14.6,
                                   spaceAfter=8, textColor=colors.black),
            "small": ParagraphStyle("Small", fontName="Arial", fontSize=9.5, leading=12.4,
                                    spaceAfter=6, textColor=colors.black),
            "title": ParagraphStyle("Title", fontName="Arial-Bold", fontSize=23, leading=28,
                                    spaceAfter=14, keepWithNext=True, textColor=colors.black),
            "heading": ParagraphStyle("Heading", fontName="Arial-Bold", fontSize=17, leading=21,
                                      spaceAfter=12, keepWithNext=True, textColor=colors.black),
            "subheading": ParagraphStyle("Subheading", fontName="Arial-Bold", fontSize=12.5,
                                         leading=16, spaceBefore=8, spaceAfter=6,
                                         keepWithNext=True, textColor=colors.black),
            "cell": ParagraphStyle("Cell", fontName="Arial", fontSize=9.4, leading=12.2,
                                   spaceAfter=0, textColor=colors.black, splitLongWords=1),
            "header": ParagraphStyle("Header", fontName="Arial-Bold", fontSize=9.4,
                                     leading=12.2, textColor=colors.white, spaceAfter=0),
            "code": ParagraphStyle("Code", fontName="Courier", fontSize=8.7, leading=11.6,
                                   spaceAfter=7, splitLongWords=1, textColor=colors.black),
        }

    def p(self, text: str, style="body") -> None:
        self.story.append(Paragraph(text, self.styles[style]))

    def h(self, text: str) -> None:
        self.p(text, "subheading")

    def page(self, title: str) -> None:
        if self.story:
            self.story.append(PageBreak())
        self.p(title, "heading")

    def table(self, headers, rows, widths) -> None:
        content = [[Paragraph(escape(str(text)), self.styles["header"]) for text in headers]]
        content.extend([[Paragraph(str(text), self.styles["cell"]) for text in row] for row in rows])
        grid = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153B64")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F6FB")]),
            ("GRID", (0, 0), (-1, -1), .45, colors.HexColor("#D9D9D9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        self.story.extend([grid, Spacer(1, 9)])

    def image(self, relative: str, width=510, max_height=265) -> None:
        path = self.root / relative
        with PILImage.open(path) as image:
            source_width, source_height = image.size
        factor = min(width / source_width, max_height / source_height)
        figure = Image(str(path), width=source_width * factor, height=source_height * factor)
        figure.hAlign = "CENTER"
        self.story.extend([figure, Spacer(1, 8)])

    def objective_plot(self, losses) -> None:
        drawing = Drawing(510, 142)
        left, bottom, width, height = 50, 29, 440, 94
        low, high = min(losses), max(losses)
        span = max(high - low, 1e-9)
        drawing.add(Line(left, bottom, left + width, bottom, strokeColor=colors.black))
        drawing.add(Line(left, bottom, left, bottom + height, strokeColor=colors.black))
        for index in range(1, len(losses)):
            x1 = left + (index - 1) * width / max(len(losses) - 1, 1)
            x2 = left + index * width / max(len(losses) - 1, 1)
            y1 = bottom + (losses[index - 1] - low) * height / span
            y2 = bottom + (losses[index] - low) * height / span
            drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#1A6DF0"), strokeWidth=1.3))
        drawing.add(String(left, bottom - 13, "1", fontName="Arial", fontSize=9))
        drawing.add(String(left + width - 12, bottom - 13, str(len(losses)), fontName="Arial", fontSize=9))
        drawing.add(String(230, 3, "Training epoch", fontName="Arial", fontSize=10))
        drawing.add(String(0, bottom + height - 2, f"{high:.3g}", fontName="Arial", fontSize=9))
        drawing.add(String(0, bottom, f"{low:.3g}", fontName="Arial", fontSize=9))
        drawing.add(String(170, 130, "Hinge objective plus regularization", fontName="Arial", fontSize=10))
        self.story.append(drawing)
        self.story.append(Spacer(1, 8))

    def build_content(self) -> None:
        d = self.data
        results, audit = d["comparison"]["methods"], d["audit"]
        tests, runner, verification = d["tests"], d["runner"], d["verification"]
        recommendation = d["comparison"]["recommendation"]["method"]
        local_date = datetime.now(timezone(timedelta(hours=8))).strftime("%d %B %Y")

        # 1
        self.p("VisAge Phase 2 to 6 Implementation Report", "title")
        self.p("Completed model development and local verification", "subheading")
        self.p(f"Prepared {local_date} | Academic research prototype", "small")
        self.p(
            f"VisAge now has a reproducible three-method age-group classification pipeline. "
            f"The completed work covers environment setup, a frozen {audit['included_samples']:,}-image "
            f"research dataset, a classical model, a randomly initialized neural model, and an adapted "
            f"pretrained model. Each predicts under 18, ages 18 to 59, or 60 and above."
        )
        self.table(["Method", "Test accuracy", "Test macro F1"], [
            (NAMES[method], percent(results[method]["metrics"]["accuracy"]),
             number(results[method]["metrics"]["macro_f1"], 4)) for method in METHODS
        ], [290, 112, 112])
        self.p(
            f"The default recommendation is <b>{escape(NAMES[recommendation])}</b>, chosen using "
            f"validation macro F1. The held-out test results above are reporting evidence, not a "
            f"selection criterion. The comparison uses all {audit['split_counts']['test']:,} frozen test images."
        )
        self.p(
            f"Automated phase checks recorded {tests['passed']} passing tests, {tests['failures']} failures, "
            f"{tests['errors']} errors, and {tests['skipped']} skipped tests. An independent CPU check reloaded "
            f"all three artifacts, and all {runner['code_cells']} notebook code cells completed locally "
            f"in artifact mode with zero cell errors."
        )
        self.h("Scope of completion")
        self.p(
            "These results complete the requested model-development phases. They do not establish a "
            "working web application, completion of every course experiment, accuracy on independent "
            "local participants, or a successful Colab run. Those tasks remain visible in the next-phase plan."
        )
        self.p(
            "Predictions describe the dataset's provided age-group annotations. They do not verify actual "
            "age, identity, sale eligibility, or senior-discount entitlement. No numerical age or calibrated "
            "age interval is claimed."
        )

        # 2
        self.page("Phase coverage and verified environment")
        env = d["environment"]
        self.p(
            f"The recorded training platform is {escape(env['platform'])}. CUDA was available through "
            f"torch's {escape(str(env['torch_cuda_build']))} runtime build. GPU: "
            f"{escape(env['gpu']['name'])}, {number(env['gpu']['vram_bytes'] / 1024**3, 2)} GiB reported VRAM. "
            f"The driver record is {escape(env['nvidia_smi'])}."
        )
        self.table(["Component", "Recorded version"], [
            ("Python", escape(env["python"].split(" (")[0])),
            *[(package, escape(str(version))) for package, version in env["packages"].items()],
            ("CPU inference threads", str(d["comparison"]["environment"]["torch_threads"])),
            ("Deterministic algorithms", str(env["deterministic_algorithms"])),
        ], [240, 274])
        self.p(
            "The project runs sequential neural training on the laptop GPU. Framework installation alone "
            "was not treated as proof of GPU execution: synthetic forward/backward pilots and the actual "
            "training runs supply that evidence. Exact environment JSON files accompany the artifacts.",
            "small",
        )

        self.h("GPU feasibility measurements")
        pilot_rows = []
        for pilot in d["pilots"]:
            for record in pilot["results"]:
                pilot_rows.append((
                    record["method"], str(record["batch_size"]),
                    number(record["seconds_per_step"] * 1000, 1),
                    number(record["images_per_second"], 1),
                    number(record["peak_allocated_mib"], 1),
                ))
        self.table(["Method", "Batch", "Step ms", "Images per s", "Peak MiB"], pilot_rows,
                   [172, 50, 86, 98, 108])
        self.p(
            "These synthetic 224-pixel forward/backward diagnostics use untrained weights and "
            "provide no age-classification accuracy evidence. Step times are synchronized after "
            "warmup. Peak allocated memory is a PyTorch counter, not total GPU residency. Actual "
            "measured epoch times include data loading, augmentation and validation, but exclude checkpoint serialization.",
            "small",
        )

        # 4
        self.page("Dataset provenance and frozen partitions")
        provenance = d["provenance"]
        self.p(
            "The dataset is UTKFace aligned and cropped faces. The authors' download links were unavailable "
            "during setup, so the explicitly documented Kaggle mirror was used. The downloader selects "
            "the full UTKFace directory, records its version and archive hash, and refuses an unexpected "
            "partial layout. The source images are research material, not public website assets."
        )
        self.table(["Provenance item", "Actual record"], [
            ("Original source", escape(provenance["original_source"])),
            ("Selected mirror", escape(provenance["mirror"])),
            ("Mirror version", str(provenance["mirror_version"])),
            ("Archive bytes", f"{provenance['archive_bytes']:,}"),
            ("Selected archive prefix", escape(provenance["selected_archive_prefix"])),
            ("Extracted source images", f"{provenance['extracted_images']:,}"),
            ("Retained after audit", f"{audit['included_samples']:,}"),
        ], [164, 350])
        self.table(["Partition", "Under 18", "18 to 59", "60 plus", "Total"], [
            (split.title(), *[f"{audit['split_class_counts'][split][str(label)]:,}" for label in range(3)],
             f"{audit['split_counts'][split]:,}") for split in ("train", "validation", "test")
        ], [126, 97, 97, 97, 97])
        self.p(
            "A single seed-42 manifest supplies the same source images, labels, and partitions to every "
            "method. The approximate 70/15/15 allocation keeps perceptual groups together. Training alone "
            "supplies scaler statistics, class weighting and augmentation; validation supplies selection. "
            "The held-out test partition supplies the final comparison."
        )
        self.p(
            "UTKFace ages originated from DEX estimates checked by a human annotator. They are not verified "
            "chronological ages. The 18-to-59 class dominates the data, so macro F1 and class-specific "
            "errors accompany overall accuracy. Gender and ethnicity are not prediction inputs."
        )

        # 5
        self.page("Data audit and independent leakage checks")
        review = d["data_review"]
        self.table(["Audit outcome", "Count"], [
            ("Raw files reconciled", f"{review['raw_files']:,}"),
            ("Retained canonical images", f"{review['retained_images']:,}"),
            *[(reason.replace("_", " ").title(), f"{count:,}")
              for reason, count in review["exclusion_reasons"].items()],
            ("Perceptual groups", f"{review['groups']:,}"),
            ("Multi-image perceptual groups", f"{review['multi_image_groups']:,}"),
            ("Largest group", str(review["largest_group"])),
        ], [350, 164])
        neighbors = review["perceptual_neighbor_check"]
        self.p(
            f"The primary audit fully decoded EXIF-corrected RGB images and enforced the 20-million-pixel "
            f"limit. Exact pixel duplicates retain a canonical sample; conflicting age-group labels exclude "
            f"the entire duplicate group. The independent review reconciled every retained/excluded path, "
            f"rechecked file and archive hashes, and used a separate bit-bucket search for perceptual neighbors."
        )
        self.p(
            f"That independent search checked {neighbors['candidate_pairs_checked']:,} candidate pairs and "
            f"found {neighbors['qualifying_neighbor_pairs']:,} pairs within Hamming distance four, with "
            f"<b>{neighbors['cross_split_neighbors']} crossing splits</b>. Its connected components matched "
            f"the recorded groups. This checks the implemented similarity rule without reusing the production BK-tree."
        )
        self.h("Frozen evidence")
        self.p("Manifest SHA-256: " + escape(d["manifest_sha256"]), "small")
        self.p("Archive SHA-256: " + escape(provenance["archive_sha256"]), "small")
        self.p(
            "<b>Identity limitation:</b> UTKFace has no subject IDs. Perceptual groups do not prove that "
            "all images of the same person stay in one partition. No subject-independent performance "
            "claim is made. The recorded mirror hash identifies this download; no author-provided archive "
            "hash was available for original-byte equivalence."
        )

        # 6
        self.page("Method A classical implementation and training")
        selection, config = d["classical_selection"], d["classical_config"]
        features = selection["feature_config"]
        self.p(
            "The group-reviewable classical module implements image gradients, HOG histograms/block "
            "normalization, rotation-invariant uniform LBP at multiple radii, feature scaling, and a "
            "class-weighted multiclass linear SVM. It uses no sklearn SVC, OpenCV HOG, or library LBP "
            "call for its submitted core."
        )
        selected = selection["selected_configuration"]
        self.table(["Recorded setting", "Value"], [
            ("Input region and size", f"Whole shared aligned crop; {features['image_size']} by {features['image_size']} grayscale"),
            ("Descriptors", f"HOG enabled {features['include_hog']}; LBP enabled {features['include_lbp']}"),
            ("LBP radii and grid", escape(str(features["lbp_radii"])) + "; " + str(features["lbp_grid"]) + " by " + str(features["lbp_grid"])),
            ("Training epochs and batch", f"{config['epochs']} epochs; {config['batch_size']} examples"),
            ("Declared learning rates", escape(str(config["learning_rates"]))),
            ("Declared regularization values", escape(str(config["regularizations"]))),
            ("Selected configuration", f"Learning rate {selected['learning_rate']}; L2 {selected['regularization']}; balanced weights; train standardization"),
            ("Validation macro F1", number(selection["validation_metrics"]["macro_f1"], 4)),
        ], [190, 324])
        self.objective_plot(d["classical_history"]["loss"])
        self.p(
            f"The selected fit took {number(selection['selected_fit_seconds'] / 60)} minutes; the recorded "
            f"search took {number(selection['search_seconds'] / 60)} minutes. Cached descriptor preparation "
            f"took {number(config['feature_extraction_seconds'] / 60)} minutes separately. Feature arrays "
            f"are bound to the manifest, configuration and implementation hash."
        )
        self.p(
            "The SVM fits scaling and class weights on training rows and selects by validation macro F1, "
            "then accuracy, then declared grid order. Test rows never enter fitting or model selection. "
            "The course table records zero classical parameters; learned coefficients and artifact size "
            "remain real storage costs and are disclosed separately.",
            "small",
        )

        # 7
        self.page("Method B custom ordinal neural network")
        custom = d["neural"]["visagenet"]
        self.p(
            "VisAgeNet combines a convolutional stem with four residual stages at widths 32, 64, 128, "
            "and 256, two blocks per stage, squeeze-and-excitation channel gates, global pooling, "
            "dropout, and a two-logit CORN head. These established components form the project's "
            "custom architecture; this work does not claim to invent residual connections, SE, or CORN."
        )
        self.neural_settings("visagenet")
        self.image("reports/evaluation/visagenet_training_curves.png", max_height=205)
        self.p(
            f"The network was trained from scratch without pretrained weights, using random convolution/linear "
            f"initialization and conventional bias/normalization initialization. The completed run contains "
            f"{custom['summary']['epochs_completed']} epochs; its selected checkpoint is epoch "
            f"{custom['summary']['best_epoch']}, with validation macro F1 "
            f"{number(custom['summary']['best_validation_macro_f1'], 4)}. Recorded training time is "
            f"{number(custom['summary']['training_seconds'] / 60)} minutes."
        )
        self.p(
            "Training accuracy uses augmented, class-balanced sampled batches. The validation series "
            "uses the unchanged validation partition. Their distributions differ, so their apparent "
            "gap must not be interpreted as a clean training-versus-validation generalization gap.",
            "small",
        )

        # 8
        self.page("Method C pretrained ordinal adaptation")
        modern = d["neural"]["convnext_ordinal"]
        weights = d["weights"]
        self.p(
            "The modern method uses ConvNeXt V2 Pico, published in 2023, with an FCMAE-pretrained and "
            "ImageNet-1k-fine-tuned backbone. The ImageNet head is discarded and a new two-output CORN "
            "head is initialized. Recorded head-only training is followed by backbone fine-tuning."
        )
        self.neural_settings("convnext_ordinal")
        self.image("reports/evaluation/convnext_ordinal_training_curves.png", max_height=205)
        self.p(
            f"The selected checkpoint is epoch {modern['summary']['best_epoch']} from "
            f"{modern['summary']['epochs_completed']} completed epochs. Validation macro F1 is "
            f"{number(modern['summary']['best_validation_macro_f1'], 4)}; recorded training time is "
            f"{number(modern['summary']['training_seconds'] / 60)} minutes."
        )
        self.p(
            f"Starting weights: {escape(weights['repository'])}; revision {escape(weights['revision'])}. "
            f"Terms: {escape(str(weights['license']))}. The model's published ImageNet performance is "
            f"not an age-group accuracy claim. The project deliberately resizes the full shared crop to "
            f"224 pixels, rather than using the published 288-pixel test crop.",
            "small",
        )

        # 9
        self.page("Common outputs and held out test comparison")
        self.p(
            "CORN learns the ordered boundaries conditionally. Let q18 be its estimate for class 18 or "
            "above. Let q60 be its estimate for class 60 or above conditional on being in class 18 or "
            "above. The second task trains only on examples in classes 18 to 59 and 60 plus."
        )
        self.table(["Class", "Derived neural class value"], [
            ("Under 18", "1 minus q18"),
            ("18 to 59", "q18 multiplied by (1 minus q60)"),
            ("60 and above", "q18 multiplied by q60"),
        ], [180, 334])
        self.p(
            "The largest of these three values selects the class. SVM margins and neural outputs are "
            "not calibrated confidence percentages. All methods share the class vocabulary, not necessarily "
            "the same prediction for each image."
        )
        self.table(["Method", "Accuracy", "Macro F1", "Parameters M"], [
            (NAMES[method], percent(results[method]["metrics"]["accuracy"]),
             number(results[method]["metrics"]["macro_f1"], 4),
             number(results[method].get("parameters", 0) / 1e6, 3)) for method in METHODS
        ], [236, 92, 92, 94])
        self.h("Boundary errors against provided annotations")
        boundary_rows = []
        for method in METHODS:
            cm = results[method]["metrics"]["confusion_matrix"]
            under_total, senior_total = sum(cm[0]), sum(cm[2])
            non_senior_total = sum(cm[0]) + sum(cm[1])
            under_wrong, senior_wrong, false_senior = sum(cm[0][1:]), sum(cm[2][:2]), cm[0][2] + cm[1][2]
            boundary_rows.append((
                NAMES[method], f"{under_wrong}/{under_total}<br/>{percent(under_wrong / under_total)}",
                f"{senior_wrong}/{senior_total}<br/>{percent(senior_wrong / senior_total)}",
                f"{false_senior}/{non_senior_total}<br/>{percent(false_senior / non_senior_total)}",
            ))
        self.table(["Method", "Under 18 predicted older", "60 plus predicted younger", "Below 60 predicted 60 plus"],
                   boundary_rows, [196, 106, 106, 106])
        self.p(
            "These are classification mistakes relative to the source annotations, not measured "
            "illegal-sale or discount decisions. The independent local-data and actual application "
            "decision studies have not been conducted.",
            "small",
        )

        # Held-out confusion plots are kept large enough to inspect labels.
        self.page("Classical and custom model confusion matrices")
        self.p(
            "Rows are provided age groups and columns are predicted groups. Diagonal cells are "
            "correct classifications. Both matrices cover the same held-out test images.",
            "small",
        )
        for method in ("classical", "visagenet"):
            self.image(f"reports/evaluation/{method}_confusion.png", width=350, max_height=260)
        self.p(
            "Overall accuracy is sensitive to the dominant middle class. Compare macro F1 and "
            "the boundary-error counts alongside these matrices.", "small",
        )

        self.page("Modern model confusion matrix and error analysis")
        self.image("reports/evaluation/convnext_ordinal_confusion.png", width=415, max_height=350)
        self.table(["Method", "Under 18 recall", "18 to 59 recall", "60 plus recall"], [
            (NAMES[method], *[percent(item["recall"]) for item in results[method]["metrics"]["per_class"]])
            for method in METHODS
        ], [214, 100, 100, 100])
        self.p(
            "The matrices and per-class recall show which annotated groups each method misses. "
            "They do not establish chronological-age verification or real transaction outcomes. "
            "Differences may reflect representation, optimization, data imbalance and label noise. "
            "Controlled ablations and an independent local pilot are still needed to test those explanations."
        )

        # 11
        self.page("Inference efficiency and independent reload verification")
        timing_rows = []
        for method in METHODS:
            value = results[method]
            network = value["cpu_batch1"]
            inclusive = value.get("cpu_preprocess_and_inference_batch1", network)
            gpu = value.get("gpu_batch1")
            timing_rows.append((
                NAMES[method], number(inclusive["median_ms"], 2),
                number(network["p95_ms"], 2),
                number(gpu["median_ms"], 2) if gpu else "Not measured",
                number(value["artifact_bytes"] / 1e6, 2),
            ))
        self.table(["Method", "CPU pipeline median ms", "CPU core p95 ms", "GPU core median ms", "Artifact MB"],
                   timing_rows, [166, 96, 90, 88, 74])
        self.p(
            "The comparable CPU pipeline measurement includes PIL-aligned-face preprocessing plus "
            "prediction. The core measurement for neural methods starts from a tensor; Method A's "
            "measurement includes descriptor extraction and SVM prediction. Each uses one repeated "
            "image after five warmups. Disk decoding, face detection, network transfer, and browser "
            "work are excluded. These values are not end-to-end web response times."
        )
        self.h("Independent validation artifact check")
        self.table(["Method", "Saved validation rows", "CPU reload samples", "Maximum score difference"], [
            (NAMES[method],
             f"{verification['methods'][method]['validation_rows_checked']:,}",
             str(verification["methods"][method]["reload_inference_samples"]),
             f"{verification['methods'][method]['maximum_score_absolute_error']:.3g}")
            for method in METHODS
        ], [224, 100, 90, 100])
        self.p(
            "The checker verifies artifact and CSV hashes, recomputes metrics from every saved validation "
            "row, and repeats CPU inference on fixed validation samples. It compares selected epochs, "
            "parameter counts, predictions, and score values. It performs no training and opens no test images."
        )
        tolerance = verification["score_tolerance"]
        self.p(
            f"Score comparisons use relative tolerance {tolerance['rtol']} and absolute tolerance "
            f"{tolerance['atol']}. All recorded method checks passed. The check applies to the saved "
            f"artifacts and selected cases, not all possible runtime behavior.",
            "small",
        )

        # 12
        self.page("Automated checks and executed notebook")
        self.table(["Check", "Recorded outcome"], [
            ("Automated tests", f"{tests['tests']} tests; {tests['passed']} passed; {tests['skipped']} skipped"),
            ("Failures and errors", f"{tests['failures']} failures; {tests['errors']} errors"),
            ("Test suite duration", number(tests["seconds"], 2) + " seconds"),
            ("Independent data review", "Passed; exact path reconciliation and no close pHash neighbors crossing splits"),
            ("Independent artifact check", "All three models passed saved-validation and CPU reload checks"),
            ("Notebook host and mode", escape(runner["execution_host"]) + "; " + escape(runner["mode"])),
            ("Notebook execution", f"{runner['executed_code_cells']}/{runner['code_cells']} code cells executed; {runner['cell_errors']} cell errors"),
            ("Notebook elapsed time", number(runner["elapsed_seconds"] / 60) + " minutes"),
            ("Colab execution", "Not verified; no Colab execution claim is made"),
        ], [205, 309])
        self.p(
            "The checks cover classical descriptor patterns and gradients, SVM sanity fitting, age-boundary "
            "labels, data corruption and freeze safety, ordinal conditional masks and probabilities, "
            "neural forward/backward shapes, checkpoint round-trips, and tiny training/resume behavior. "
            "The machine-readable JUnit file retains individual case names and runtimes."
        )
        self.p(
            "The notebook was executed top to bottom locally after training. Artifact mode verifies "
            "the data, runs checks, reloads models, renders actual histories, and runs validation "
            "evaluation without retraining. The actual local execution engine is "
            f"<b>{escape(runner['execution_engine'])}</b>. "
            + ("Real streams and PNG figures are captured; no Jupyter or Colab run is claimed."
               if runner["execution_engine"] == "python_cells" else
               "The project-local Jupyter kernel points to the project virtual environment.")
        )
        self.h("Implementation issues and their limits")
        self.p(
            "An initial two-worker Windows job caused severe RAM pressure and was restarted before "
            "a completed epoch with zero workers. A detached shell launch did not persist, so managed "
            "Python launch/logging was used. An earlier slow PIL diagnostic was archived separately; "
            "its epoch is not a final-model result. A kernel startup probe hung, so the explicit "
            "socket-free local cell executor is available. Training times exclude those attempts, "
            "cache preparation, checkpoint serialization, installation, downloads and diagnostics. Current "
            "checks do not guarantee error-free future operation.", "small",
        )

        # 13
        self.page("Reproduction next phases and limitations")
        self.table(["Evidence or entry point", "Purpose"], [
            ("README.md and requirements files", "Environment setup, commands, provenance restrictions and execution modes."),
            ("scripts/prepare_data.py", "Audit the documented raw source; verify the immutable data freeze."),
            ("scripts/train_classical.py<br/>scripts/train_neural.py", "Reproduce declared training using shared manifests and versioned configurations."),
            ("scripts/evaluate_models.py<br/>scripts/check_artifacts.py", "Held-out comparison, timing, plots, and independent artifact validation."),
            ("scripts/execute_notebook.py", "Local artifact-only executed notebook; no automatic Colab claim."),
            ("reports/ and artifacts/", "Machine-readable evidence, plots, selected models, configurations and hashes."),
        ], [220, 294])
        self.p(
            "From the project root, use the project virtual environment. The README contains the full "
            "training sequence. Key verification commands are:",
            "small",
        )
        for command in (
            "python -m pytest -q",
            "python scripts/check_artifacts.py",
            "python scripts/execute_notebook.py --engine python --split validation",
        ):
            self.p(escape(command), "code")
        self.p(
            "Next, integrate upload and controlled face selection, expose all three methods and their "
            "comparison, add camera capture and advisory wording, then test accessibility and failure "
            "states. Complete the approved data-efficiency, robustness, component-ablation and independent "
            "local-data experiments. Finish the report, slides, actual contributions and clean Colab run."
        )
        self.p(
            "Current limits: one recorded training seed; annotated ages and class imbalance; unproven "
            "subject independence; non-commercial dataset/weight terms; no independent local participant "
            "pilot; no calibrated certainty or chronological-age validation; no completed web application "
            "or end-to-end response benchmark. A pretrained-versus-custom architecture comparison alone "
            "does not isolate the causal effect of pretraining.",
            "small",
        )
        self.page("Primary references and evidence provenance")
        self.h("Recorded preprocessing and cache policy")
        policy = d["neural"]["visagenet"]["config"]["transform"]["train_augmentation"]
        cache = d["tensor_cache"]
        self.p(
            f"The successful neural runs use {escape(policy['name'])}: independently sampled horizontal "
            f"flip probability {policy['horizontal_flip_probability']}, rotations from "
            f"{policy['rotation_degrees'][0]} to {policy['rotation_degrees'][1]} degrees with bilinear "
            f"tensor sampling and zero padding, then fixed-order brightness, contrast and saturation "
            f"factors. RGB values are clamped after each color operation and normalized with each "
            f"model's recorded mean/std. This differs deliberately from the archived PIL policy.", "small",
        )
        self.p(
            f"The train/validation-only uint8 NCHW cache holds {cache['rows']:,} full faces at "
            f"{cache['image_size']} pixels, {number(cache['array_bytes'] / 1024**3, 2)} GiB; build time "
            f"was {number(cache['build_seconds'] / 60)} minutes. Source/order/preprocessing and chunk "
            f"hashes bind the cache. Held-out test faces are excluded. Cache preparation and the "
            f"archived startup diagnostics are outside measured successful-epoch training time.", "small",
        )
        self.p(
            "Classical descriptor branches are normalized then train-standardized per feature. "
            "Initial branch multipliers therefore do not establish equal effective HOG/LBP SVM "
            "weights. Frozen-data and prediction records remain the basis for comparing the methods.", "small",
        )
        references = [
            ("Course brief", COURSE_SOURCE),
            ("UTKFace authors", "https://susanqq.github.io/UTKFace/"),
            ("ConvNeXt V2", "https://arxiv.org/abs/2301.00808"),
            ("Selected weights", "https://huggingface.co/timm/convnextv2_pico.fcmae_ft_in1k"),
            ("CORN", "https://arxiv.org/abs/2111.08851"),
            ("HOG", "https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf"),
            ("SE attention", "https://arxiv.org/abs/1709.01507"),
        ]
        for label, location in references:
            self.p(f"<b>{escape(label)}:</b> {escape(location)}", "small")
        self.p("Revision: " + escape(d["weights"]["revision"]), "small")
        self.p("Starting weights SHA-256: " + escape(d["weights"]["sha256"]), "small")
        self.p(
            "UTKFace permits non-commercial research; original image owners retain copyright. "
            "The selected ConvNeXt V2 weights carry CC BY-NC 4.0 terms. Research photos must not "
            "be republished as website samples without suitable separate rights. The report includes "
            "plots and metrics, not face photographs.", "small",
        )
        self.h("Implementation assistance and review")
        self.p(
            "Substantial code and documentation were drafted with Codex assistance. Group members "
            "must review, understand, test, and disclose this contribution, and confirm instructor "
            "expectations beyond the brief's explicitly allowed debugging and plotting. Established "
            "residual, SE, CORN, HOG and ConvNeXt components are credited as prior work.", "small",
        )
        self.p(
            "The machine-readable reports/phase_report_build.json binds source checksums, the PDF "
            "hash, and page counts. Required completed records are checked before generation.", "small",
        )

    def neural_settings(self, method: str) -> None:
        item = self.data["neural"][method]
        config, summary = item["config"], item["summary"]
        rows = [
            ("Parameters", f"{summary['parameters']:,}"),
            ("Seed and head", f"{config['seed']}; {escape(config['model_config']['head_type'])}"),
            ("Batch and workers", f"{config['batch_size']}; {config['workers']} data-loader workers"),
            ("Optimizer and learning rate", "AdamW; " + str(config["learning_rate"])),
            ("Weight decay and dropout", str(config["weight_decay"]) + "; " + str(config["model_config"]["dropout"])),
            ("Early stopping", f"Budget {config['epochs']}; minimum {config['min_epochs']}; patience {config['patience']}"),
        ]
        if method == "convnext_ordinal":
            rows.append(("Head stage", f"{config['head_epochs']} epochs; head rate {config['head_learning_rate']}"))
        rows.append(("Preprocessing", escape(config["transform"]["geometry"])))
        self.table(["Actual setting", "Value"], rows, [190, 324])


def build_report(root: Path, output: Path, font_dir: Path) -> dict:
    evidence = load_evidence(root)
    add_font(font_dir)
    report = Report(root, evidence)
    report.build_content()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.pdf")
    generated_date = datetime.now(timezone(timedelta(hours=8))).strftime("%d %B %Y")

    def furniture(canvas, document):
        canvas.saveState()
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#444444"))
        if document.page > 1:
            canvas.drawString(49, 766, "VisAge model development and local verification")
        canvas.drawString(49, 27, f"Academic research | Actual completed evidence | {generated_date}")
        canvas.drawRightString(563, 27, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(temporary), pagesize=letter, leftMargin=49, rightMargin=49,
        topMargin=47, bottomMargin=44,
        title="VisAge Phase 2 to 6 Implementation Report", author="VisAge project",
        subject="Completed model-development evidence and limitations",
    )
    document.build(report.story, onFirstPage=furniture, onLaterPages=furniture)
    reader = PdfReader(temporary)
    if len(reader.pages) != EXPECTED_PAGES:
        raise ValueError(
            f"Report layout produced {len(reader.pages)} pages, expected {EXPECTED_PAGES}. "
            f"Inspect {temporary} and adjust layout before delivery."
        )
    for name, original_hash in evidence["source_checksums"].items():
        if sha256(root / name) != original_hash:
            raise ValueError(f"Evidence changed during report generation: {name}")
    temporary.replace(output)
    summary = {
        "output": output.relative_to(root).as_posix(), "pages": len(reader.pages),
        "sha256": sha256(output), "manifest_sha256": evidence["manifest_sha256"],
        "evaluation_split": evidence["comparison"]["split"],
        "all_required_evidence_validated": True,
        "page_word_counts": [len((page.extract_text() or "").split()) for page in reader.pages],
        "evidence_checksums": evidence["source_checksums"],
        "visual_qa_required": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (root / "reports/phase_report_build.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/pdf/VisAge_Phase_2_to_6_Implementation_Report.pdf")
    parser.add_argument("--font-dir", type=Path, default=Path(r"C:\Windows\Fonts"))
    parser.add_argument("--check", action="store_true", help="Validate required evidence without generating a PDF.")
    arguments = parser.parse_args()
    root, output = arguments.root.resolve(), arguments.output.resolve()
    if not output.is_relative_to(root):
        parser.error("Output must stay inside the project workspace.")
    if arguments.check:
        evidence = load_evidence(root)
        print(json.dumps({"ready": True, "manifest_sha256": evidence["manifest_sha256"],
                          "test_samples": evidence["audit"]["split_counts"]["test"]}, indent=2))
    else:
        print(json.dumps(build_report(root, output, arguments.font_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

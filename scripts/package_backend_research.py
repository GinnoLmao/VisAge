"""Package verified backend, detector, research evidence and trained checkpoints.

Run only after final independent review and report visual QA. Raw photos,
virtual environments, research optimizer state and descriptor/tensor caches are
omitted. Two original baseline resume records are included because the frozen
study's preservation checks require their exact bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.faces import DETECTOR_FILENAME, DETECTOR_SHA256, DETECTOR_LICENSE_SHA256
from visage.research import verify_protected_files
from visage.runtime import sha256_file, utc_now, write_json


def main():
    def read(relative):
        return json.loads((ROOT / relative).read_text(encoding="utf-8"))

    protocol = read("reports/research/protocol.json")
    summary = read("reports/research/study_summary.json")
    review = read("reports/research/independent_review.json")
    benchmark = read("reports/backend_benchmark.json")
    report = read("reports/backend_research_report_build.json")
    visual = read("reports/backend_research_visual_qa.json")
    manifest_sha = sha256_file(ROOT / "data/processed/manifest.csv")
    if not review.get("study_complete") or summary["new_test_evaluations"] != 0:
        raise ValueError("All predeclared research stages must complete without new test-set evaluation.")
    if summary["protocol_sha256"] != sha256_file(ROOT / "reports/research/protocol.json"):
        raise ValueError("Study summary belongs to another protocol.")
    if not review.get("ok") or not benchmark.get("ok"):
        raise ValueError("Independent research verification and backend benchmark must pass.")
    if not read("reports/extended_evaluation/independent_review.json").get("ok"):
        raise ValueError("Detection AP and age-group ROC must pass independent verification.")
    if any(record["manifest_sha256"] != manifest_sha for record in (protocol, summary, report, review)):
        raise ValueError("Package records belong to different dataset freezes.")
    verify_protected_files(ROOT, protocol["protected_original_files"])
    if not report.get("all_required_evidence_validated") or not visual.get("ok") or visual["pdf_sha256"] != report["sha256"]:
        raise ValueError("Report evidence and final visual QA must pass.")
    if sorted(visual["pages_reviewed"]) != list(range(1, report["pages"] + 1)):
        raise ValueError("Every final report page must be inspected.")
    for relative, expected in report["evidence_checksums"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"Evidence changed after report generation: {relative}")
    cases = ET.parse(ROOT / "reports/tests_backend_research.xml").findall(".//testcase")
    if not cases or any(any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in cases):
        raise ValueError("The complete local test suite must pass with no skips.")
    if not read("reports/api_http_smoke.json").get("ok"):
        raise ValueError("Real HTTP smoke must pass.")
    detector_dir = ROOT / ".cache/face_detector"
    if sha256_file(detector_dir / DETECTOR_FILENAME) != DETECTOR_SHA256 or sha256_file(detector_dir / "LICENSE") != DETECTOR_LICENSE_SHA256:
        raise ValueError("The packaged detector/license changed.")
    candidates = []
    for name in ("src", "scripts", "tests", "configs", "notebooks", "reports"):
        candidates.extend(path for path in (ROOT / name).rglob("*") if path.is_file()
                          and "__pycache__" not in path.parts and path.suffix not in {".pid", ".err"}
                          and path.name != "pipeline_check.log")
    for name in ("README.md", "pyproject.toml", ".gitignore", "requirements-colab.txt",
                 "requirements-lock-windows-cu128.txt", "requirements-lock-backend-windows-cu128.txt"):
        candidates.append(ROOT / name)
    candidates.append(ROOT / "data/provenance.json")
    candidates.extend(ROOT / name for name in (
        "data/detection_benchmark/provenance.json",
        "data/detection_benchmark/extracted/wider_face_split/wider_face_val_bbx_gt.txt",
    ))
    candidates.extend(path for path in (ROOT / "data/processed").glob("*") if path.is_file())
    for name in ("pretrained_provenance.json", "convnextv2_pico_config.json"):
        candidates.append(ROOT / ".cache/weights" / name)
    candidates.extend(detector_dir / name for name in (DETECTOR_FILENAME, "LICENSE", "provenance.json"))
    for path in (ROOT / "artifacts").rglob("*"):
        if path.is_file() and "diagnostics" not in path.parts and (path.name != "last.pt" or path.relative_to(ROOT).as_posix() in protocol["protected_original_files"]):
            candidates.append(path)
    candidates.append(ROOT / report["output"])
    candidates.extend(ROOT / "output/pdf" / name for name in (
        "VisAge_Project_Documentation_and_Execution_Plan.pdf",
        "VisAge_Phase_2_to_6_Implementation_Report.pdf",
    ))
    candidates = sorted(set(path.resolve() for path in candidates))
    entries = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)} for path in candidates]
    manifest = {"created_utc": utc_now(), "files": entries,
                "scope": "Local face preprocessing and backend; completed validation-only research; local pilot awaits participant data; UI/deployment not included.",
                "excluded": ["raw face photos", "download archives", "virtual environment", "tensor/descriptor caches", "research optimizer state last.pt", "diagnostic model attempts"],
                "baseline_resume_records": "Two original baseline last.pt files are included solely to reproduce frozen preservation checks; research last.pt files are omitted.",
                "colab_required": False}
    destination = ROOT / "output/VisAge_Backend_and_Research.zip"
    write_json(ROOT / "output/backend_research_package_manifest.json", manifest)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in candidates:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("backend_research_package_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed.")
        for entry in entries:
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise RuntimeError("Packaged file differs from verified bytes.")
    write_json(ROOT / "output/backend_research_package_verification.json", {
        "ok": True, "files": len(entries), "zip_bytes": destination.stat().st_size,
        "zip_sha256": sha256_file(destination), "path": str(destination), "verified_utc": utc_now(),
    })
    print(f"Verified {len(entries)} files: {destination}")


if __name__ == "__main__":
    main()

"""Package source, trained inference artifacts, and verification evidence.

Raw research images, virtual environments, pretrained downloads and optimizer
state are intentionally excluded. Download/source references remain in the pack.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import sha256_file, utc_now, write_json


def main():
    required_records = {
        "classical": ROOT / "artifacts/classical/selection.json",
        "visagenet": ROOT / "artifacts/visagenet/training_summary.json",
        "convnext_ordinal": ROOT / "artifacts/convnext_ordinal/training_summary.json",
        "comparison": ROOT / "reports/evaluation/comparison.json",
        "verification": ROOT / "reports/artifact_verification.json",
        "notebook": ROOT / "reports/notebook_runner.json",
        "report": ROOT / "reports/phase_report_build.json",
        "visual_qa": ROOT / "reports/phase_report_visual_qa.json",
    }
    for name, path in required_records.items():
        if not path.is_file():
            raise FileNotFoundError(f"Completed {name} evidence is required before packaging: {path}")
    records = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in required_records.items()}
    manifest_sha = sha256_file(ROOT / "data/processed/manifest.csv")
    if any(records[name]["manifest_sha256"] != manifest_sha for name in ("classical", "visagenet", "convnext_ordinal", "comparison", "verification", "report")):
        raise ValueError("Packaging evidence belongs to different dataset freezes.")
    if records["comparison"]["split"] != "test" or not records["verification"].get("ok"):
        raise ValueError("Held-out evaluation and successful independent artifact verification are required.")
    if records["notebook"].get("status") != "complete" or records["notebook"].get("cell_errors") != 0:
        raise ValueError("The reproducibility notebook did not complete successfully.")
    if not records["visual_qa"].get("ok") or records["visual_qa"].get("pdf_sha256") != records["report"]["sha256"]:
        raise ValueError("The final report must pass visual inspection before packaging.")
    for relative, expected in records["report"]["evidence_checksums"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"Report evidence changed after generation: {relative}")
    for method in ("visagenet", "convnext_ordinal"):
        if sha256_file(ROOT / "artifacts" / method / "best.pt") != records[method]["artifact_sha256"]:
            raise ValueError("A trained neural artifact changed before packaging.")
    if sha256_file(ROOT / "artifacts/classical/svm.npz") != records["classical"]["svm_sha256"]:
        raise ValueError("The trained classical artifact changed before packaging.")
    candidates = []
    for name in ("src", "scripts", "tests", "configs", "notebooks", "reports"):
        candidates.extend(p for p in (ROOT / name).rglob("*") if p.is_file()
                          and "__pycache__" not in p.parts and not p.suffix in {".pid", ".err"}
                          and p.name != "pipeline_check.log")
    for name in ("README.md", "pyproject.toml", ".gitignore", "requirements-colab.txt", "requirements-lock-windows-cu128.txt"):
        candidates.append(ROOT / name)
    candidates.extend(ROOT / "data" / name for name in ("provenance.json",))
    candidates.extend(p for p in (ROOT / "data/processed").glob("*") if p.is_file())
    candidates.extend(ROOT / ".cache/weights" / name for name in ("pretrained_provenance.json", "convnextv2_pico_config.json"))
    for method in ("classical", "visagenet", "convnext_ordinal"):
        candidates.extend(p for p in (ROOT / "artifacts" / method).rglob("*") if p.is_file() and p.name != "last.pt")
    pdf = ROOT / "output/pdf/VisAge_Phase_2_to_6_Implementation_Report.pdf"
    if not pdf.exists():
        raise FileNotFoundError("Create and verify the final phase report before packaging.")
    candidates.append(pdf)
    if sha256_file(pdf) != records["report"]["sha256"]:
        raise ValueError("The report changed after its evidence validation.")
    plan = ROOT / "output/pdf/VisAge_Project_Documentation_and_Execution_Plan.pdf"
    if not plan.is_file():
        raise FileNotFoundError("The project plan is required in this phase package.")
    candidates.append(plan)
    candidates = sorted(set(p.resolve() for p in candidates))
    entries = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                "sha256": sha256_file(p)} for p in candidates]
    manifest_path = ROOT / "output/phase_package_manifest.json"
    write_json(manifest_path, {"created_utc": utc_now(), "files": entries,
                "excluded": ["raw research photos", "download archives", "virtual environment", "initial pretrained checkpoint", "optimizer/resume checkpoint last.pt"],
                "scope": "Completed steps 2-6 and their evidence; not the final course submission or web application."})
    target = ROOT / "output/VisAge_Model_Phase.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for p in candidates:
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.write(manifest_path, "phase_package_manifest.json")
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Bad packaged file: {bad}")
        embedded = json.loads(archive.read("phase_package_manifest.json"))
        if len(embedded["files"]) != len(entries):
            raise RuntimeError("Package manifest count mismatch.")
        import hashlib
        for entry in embedded["files"]:
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise RuntimeError("Packaged bytes differ from the recorded manifest.")
    write_json(ROOT / "output/phase_package_verification.json", {
        "ok": True, "files": len(entries), "zip_bytes": target.stat().st_size,
        "zip_sha256": sha256_file(target), "path": str(target), "verified_utc": utc_now(),
    })
    print(f"Verified {len(entries)} files: {target}")


if __name__ == "__main__":
    main()

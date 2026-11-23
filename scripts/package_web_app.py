"""Create a verified, inference-ready web-app delivery without raw photos."""
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
from visage.runtime import sha256_file, utc_now, write_json


def main():
    from visage.model_registry import ARTIFACTS
    if (ARTIFACTS / "bundle.json").is_file():
        from package_pretrained_app import main as package_pretrained
        return package_pretrained()

    def read(relative):
        return json.loads((ROOT / relative).read_text(encoding="utf-8"))

    for name in ("http_verification", "browser_qa", "preservation_checks", "software_checks", "wheel_assets"):
        if not read(f"reports/web_app/{name}.json")["ok"]:
            raise ValueError(f"Web verification is incomplete: {name}")
    for name in ("tests_full.xml", "tests_javascript.xml"):
        cases = ET.parse(ROOT / "reports/web_app" / name).findall(".//testcase")
        if not cases or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
            raise ValueError(f"Tests did not pass: {name}")
    for name in ("preview", "camera_qa"):
        cleanup = read(f"reports/web_app/{name}_cleanup.json")
        if not cleanup["port_closed"] or not cleanup["server_process_stopped"]:
            raise ValueError("Owned preview servers must be stopped before delivery.")
    final_browser = read("reports/web_app/headless_cleanup.json")
    if not final_browser["browser_process_stopped"] or final_browser["remaining_browser_processes"] or not final_browser["server"]["port_closed"]:
        raise ValueError("Final browser QA processes must be stopped before delivery.")
    protocol = read("reports/research/protocol.json")
    for relative, digest in {**protocol["implementation"]["source_files"], **protocol["protected_original_files"]}.items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Frozen research evidence changed: {relative}")
    detector = ROOT / ".cache/face_detector"
    if sha256_file(detector / DETECTOR_FILENAME) != DETECTOR_SHA256 or sha256_file(detector / "LICENSE") != DETECTOR_LICENSE_SHA256:
        raise ValueError("Detector or license checksum changed.")

    candidates = []
    for directory in ("src/visage", "scripts", "tests", "configs", "reports", "notebooks"):
        candidates.extend(path for path in (ROOT / directory).rglob("*") if path.is_file()
                          and "__pycache__" not in path.parts and path.suffix not in {".log", ".pid", ".err"})
    candidates.extend(ROOT / name for name in ("README.md", "pyproject.toml", "package.json", ".gitignore",
        "requirements-colab.txt", "requirements-lock-windows-cu128.txt", "requirements-lock-backend-windows-cu128.txt", "data/provenance.json"))
    candidates.extend((ROOT / "data/processed").glob("*"))
    candidates.extend(ROOT / name for name in ("data/detection_benchmark/provenance.json",
        "data/detection_benchmark/extracted/wider_face_split/wider_face_val_bbx_gt.txt"))
    candidates.extend(detector / name for name in (DETECTOR_FILENAME, "LICENSE", "provenance.json"))
    candidates.extend(ROOT / ".cache/weights" / name for name in ("pretrained_provenance.json", "convnextv2_pico_config.json"))
    for method in ("classical", "visagenet", "convnext_ordinal"):
        candidates.extend(path for path in (ROOT / "artifacts" / method).glob("*") if path.is_file() and path.name != "last.pt")
    candidates.extend(path for path in (ROOT / "output/pdf").glob("*.pdf") if path.is_file())
    candidates = sorted(set(path.resolve() for path in candidates if path.is_file()))
    # No research face photos, fixture photos, environments or optimizer states.
    for path in candidates:
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("data/raw/", ".cache/web_qa/", ".venv/")) or path.suffix.lower() in {".jpg", ".jpeg", ".webp"} or path.name == "last.pt":
            raise ValueError(f"Forbidden delivery entry: {relative}")
    entries = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in candidates]
    manifest = {"created_utc": utc_now(), "scope": "Local inference-ready browser app, three original trained classifiers, pinned detector, evaluation/research reports and test sources.",
        "limits": ["Public hosting is not included.", "Camera browser QA uses simulated video, not physical hardware.", "Consented local pilot is pending.",
                   "Research training checkpoints and optimizer state are in the earlier research delivery; this app package serves the original models."],
        "excluded": ["raw research photos", "private uploads", "local QA fixtures", "virtual environment", "research weights", "optimizer/resume state"], "files": entries}
    destination = ROOT / "output/VisAge_Web_App.zip"
    write_json(ROOT / "output/web_app_package_manifest.json", manifest)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in candidates:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("web_app_package_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise RuntimeError("ZIP CRC validation failed.")
        for entry in entries:
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Archive content mismatch: {entry['path']}")
    record = {"ok": True, "files": len(entries), "zip_bytes": destination.stat().st_size,
              "zip_sha256": sha256_file(destination), "path": str(destination), "verified_utc": utc_now()}
    write_json(ROOT / "output/web_app_package_verification.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

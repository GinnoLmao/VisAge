"""Retrieve the pinned official YuNet model and its MIT license, verifying bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.faces import (
    DETECTOR_BYTES, DETECTOR_FILENAME, DETECTOR_LICENSE_SHA256, DETECTOR_REVISION, DETECTOR_SHA256,
)


def retrieve(url: str, maximum: int) -> bytes:
    # Use the operating system trust store, retaining TLS verification on Windows.
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    request = urllib.request.Request(url, headers={"User-Agent": "VisAge-Academic-Research/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Detector source exceeded its expected size.")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/face_detector")
    args = parser.parse_args()
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{DETECTOR_REVISION}/models/face_detection_yunet"
    media = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{DETECTOR_REVISION}/models/face_detection_yunet/{DETECTOR_FILENAME}"
    pointer = retrieve(f"{prefix}/{DETECTOR_FILENAME}", 1024).decode("utf-8")
    if f"oid sha256:{DETECTOR_SHA256}" not in pointer or f"size {DETECTOR_BYTES}" not in pointer:
        raise ValueError("Official pinned LFS pointer does not match the expected model.")
    target = directory / DETECTOR_FILENAME
    model = target.read_bytes() if target.is_file() else retrieve(media, DETECTOR_BYTES)
    if len(model) != DETECTOR_BYTES or hashlib.sha256(model).hexdigest() != DETECTOR_SHA256:
        raise ValueError("Detector checksum or byte size mismatch; refusing substitution.")
    license_bytes = retrieve(f"{prefix}/LICENSE", 16_384)
    if hashlib.sha256(license_bytes).hexdigest() != DETECTOR_LICENSE_SHA256:
        raise ValueError("Unexpected detector licensing evidence.")
    # Only publish fully verified model bytes and small provenance files.
    partial = target.with_suffix(".partial")
    partial.write_bytes(model)
    partial.replace(target)
    (directory / "LICENSE").write_bytes(license_bytes)
    record = {"repository": "opencv/opencv_zoo", "revision": DETECTOR_REVISION,
              "filename": DETECTOR_FILENAME, "sha256": DETECTOR_SHA256,
              "bytes": DETECTOR_BYTES, "license": "MIT", "source_url": media,
              "license_url": f"{prefix}/LICENSE", "lfs_pointer_url": f"{prefix}/{DETECTOR_FILENAME}",
              "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
              "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    (directory / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

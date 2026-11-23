"""Download one explicitly documented UTKFace mirror, retaining the original restrictions.

Uses only standard-library networking. Does not extract executable archive content.
The original UTKFace author links were unavailable when this mirror was selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

MIRROR = "jangedoo/utkface-new"
BASE = "https://www.kaggle.com/api/v1"


def fetch_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "VisAge-Academic-Research/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--version", type=int, default=1, help="Pinned Kaggle mirror version, default 1.")
    args = parser.parse_args()
    root = args.root.resolve()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    metadata_url = f"{BASE}/datasets/list?search=utkface-new"
    candidates = fetch_json(metadata_url)
    metadata = next((x for x in candidates if x.get("ref") == MIRROR), None)
    if metadata is None:
        raise RuntimeError("The selected public mirror was not found; refusing substitution.")
    version = args.version
    if version < 1:
        raise ValueError("Mirror version must be positive.")
    (downloads / "mirror_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"mirror": MIRROR, "version": version, "metadata": metadata}, indent=2), flush=True)
    if args.metadata_only:
        return
    url = f"{BASE}/datasets/download/{MIRROR}?datasetVersionNumber={version}"
    archive = downloads / f"utkface-new-v{version}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        request = urllib.request.Request(url, headers={"User-Agent": "VisAge-Academic-Research/0.1"})
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as target:
            last_log = 0
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if count > 2 * 1024**3:
                    raise ValueError("Mirror archive unexpectedly exceeds 2 GiB.")
                if count - last_log >= 16 * 1024**2:
                    print(f"Downloaded {count / 1024**2:.1f} MiB", flush=True)
                    last_log = count
        if not zipfile.is_zipfile(partial):
            raise ValueError("The mirror did not return a readable ZIP archive.")
        partial.replace(archive)
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    existing_provenance = root / "provenance.json"
    if existing_provenance.exists():
        previous = json.loads(existing_provenance.read_text(encoding="utf-8"))
        if previous.get("mirror_version") == version and previous.get("archive_sha256") != digest:
            raise ValueError("Archive differs from the recorded mirror version checksum; refusing to replace the dataset.")
    raw = root / "raw" / "UTKFace"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        prefixes = ("UTKFace/", "utkface_aligned_cropped/UTKFace/")
        selected_prefix = next(
            (prefix for prefix in prefixes if sum(i.filename.startswith(prefix) for i in members) >= 20000),
            None,
        )
        if selected_prefix is None:
            raise ValueError("Archive lacks the expected full UTKFace directory; refusing a partial dataset.")
        chosen = [i for i in members if i.filename.startswith(selected_prefix)]
        total = 0
        names = set()
        for item in chosen:
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or item.file_size > 20 * 1024**2:
                raise ValueError(f"Unsafe or unexpected archive member: {item.filename}")
            relative = PurePosixPath(item.filename[len(selected_prefix):])
            if len(relative.parts) != 1 or relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if relative.name in names:
                raise ValueError("Duplicate filenames within the selected archive directory.")
            names.add(relative.name)
            total += item.file_size
            if total > 2 * 1024**3:
                raise ValueError("Unexpected uncompressed image size.")
            target = raw / relative.name
            if not target.exists() or target.stat().st_size != item.file_size:
                with zf.open(item) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    provenance = {
        "dataset": "UTKFace aligned and cropped faces",
        "original_source": "https://susanqq.github.io/UTKFace/",
        "original_archive_name": "UTKFace.tar.gz",
        "mirror": f"https://www.kaggle.com/datasets/{MIRROR}",
        "download_url": url,
        "mirror_version": version,
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "selected_archive_prefix": selected_prefix,
        "extracted_images": len(names),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "terms": "Non-commercial research only; original image owners retain copyrights.",
        "label_limitations": "Ages were estimated by DEX and human checked, not verified birth-record ages.",
        "identity_limitations": "No subject IDs; duplicate grouping does not prove identity-independent splits.",
        "verification_limit": "Mirror archive hash and contents recorded; no author-published hash was available for byte-for-byte original verification.",
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Download pinned WIDER FACE validation images/annotations from the CUHK HF host.

Research-only data stays outside distribution packages. No training split is
downloaded and no inference result is used to create ground-truth annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import sha256_file, utc_now, write_json

REPOSITORY = "CUHK-CSE/wider_face"
FILES = ("data/WIDER_val.zip", "data/wider_face_split.zip")


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "VisAge-Academic-Research/0.2"})
    return urllib.request.urlopen(request, timeout=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    import truststore
    truststore.inject_into_ssl()
    directory = ROOT / "data/detection_benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "provenance.json"
    if source.exists():
        provenance = json.loads(source.read_text())
    else:
        with get(f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/main?blobs=true") as response:
            metadata = json.load(response)
        revision = metadata["sha"]
        siblings = {item["rfilename"]: item for item in metadata["siblings"]}
        records = {}
        for name in FILES:
            record = siblings[name]
            records[name] = {"bytes": record["lfs"]["size"], "sha256": record["lfs"]["sha256"],
                             "url": f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{name}"}
        provenance = {"repository": REPOSITORY, "revision": revision, "files": records,
            "recorded_utc": utc_now(), "split": "validation", "license": "CC-BY-NC-ND-4.0",
            "dataset_card": f"https://huggingface.co/datasets/{REPOSITORY}/blob/{revision}/README.md",
            "scope": "Research detector evaluation only; no face photos are redistributed. Source metadata supplies archive digests; no detector-derived labels."}
        write_json(source, provenance)
    print(json.dumps(provenance), flush=True)
    if args.metadata_only:
        return
    for name, record in provenance["files"].items():
        destination = directory / Path(name).name
        if not destination.exists() or sha256_file(destination) != record["sha256"]:
            partial = destination.with_suffix(destination.suffix + ".partial")
            completed = 0
            with get(record["url"]) as response, partial.open("wb") as stream:
                while block := response.read(1024**2):
                    completed += len(block)
                    if completed > record["bytes"]:
                        raise ValueError("Download exceeded the pinned source byte size.")
                    stream.write(block)
                    if completed // (32 * 1024**2) != (completed - len(block)) // (32 * 1024**2):
                        print(json.dumps({"file": name, "downloaded_bytes": completed, "total_bytes": record["bytes"]}), flush=True)
            if completed != record["bytes"] or sha256_file(partial) != record["sha256"]:
                raise ValueError("Benchmark archive checksum or size differs from pinned source.")
            partial.replace(destination)
        extracted = directory / "extracted"
        with zipfile.ZipFile(destination) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename.replace("\\", "/"))
                # Only validation photos and validation box annotations are needed.
                keep = (path.parts and path.parts[0] == "WIDER_val") or path.name == "wider_face_val_bbx_gt.txt"
                if not keep or member.is_dir():
                    continue
                target = (extracted / str(path)).resolve()
                if path.is_absolute() or ".." in path.parts or not target.is_relative_to(extracted.resolve()):
                    raise ValueError("Unsafe archive path.")
                if member.file_size > 30 * 1024**2 or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unexpected archive member type or size.")
                if target.is_file() and target.stat().st_size == member.file_size:
                    crc = 0
                    with target.open("rb") as existing:
                        while block := existing.read(1024**2):
                            crc = zlib.crc32(block, crc)
                    if crc == member.CRC:
                        continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    while block := src.read(1024**2):
                        dst.write(block)
        print(f"Verified and extracted {destination.name}", flush=True)
    write_json(directory / "download_complete.json", {"ok": True, "provenance_sha256": sha256_file(source), "completed_utc": utc_now()})


if __name__ == "__main__":
    main()

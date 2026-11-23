"""Recreate local browser QA images from three existing frozen validation files.

Requires the original research dataset locally. Nothing is downloaded or copied
into public assets. The resulting .cache directory must not be distributed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = {
    0: "62ab562d1e3d9e34184e76ac068b9b910b40d09d1564023e31b0d4dfd429e0ce",
    1: "cdcace4d9fcb55d67ca7fa148b15ac6202705866e01d8f00f81115d90b5a7816",
    2: "3aa9506e690bbe588500e864de77c58ab00ffd8fc15436255b7f9a85e0a14ddd",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/web_qa")
    target = parser.parse_args().output
    target.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/processed/manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = {row["sample_id"]: row for row in csv.DictReader(file) if row["sample_id"] in SAMPLES.values()}
    for label, sample_id in SAMPLES.items():
        row = rows[sample_id]
        assert row["split"] == "validation" and int(row["label"]) == label
        data = (ROOT / "data/raw/UTKFace" / row["relative_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sample_id
        (target / f"class_{label}.jpg").write_bytes(data)
    (target / "single.jpg").write_bytes((target / "class_2.jpg").read_bytes())
    composite = Image.new("RGB", (600, 280), "#888888")
    composite.paste(Image.open(target / "class_0.jpg"), (25, 40))
    composite.paste(Image.open(target / "class_2.jpg"), (375, 40))
    composite.save(target / "multiple.png")
    Image.new("RGB", (640, 400), "#888888").save(target / "blank.png")
    (target / "invalid.jpg").write_bytes(b"not an image")
    (target / "oversized.jpg").write_bytes(bytes(8 * 1024 ** 2 + 1))
    upright = Image.new("RGB", (320, 260), "#888888")
    upright.paste(Image.open(target / "class_1.jpg"), (60, 30))
    exif = Image.Exif(); exif[274] = 6
    upright.transpose(Image.Transpose.ROTATE_90).save(target / "rotated_exif.jpg", quality=95, exif=exif)
    # A new reproduction gets its own record; preserve the original observed QA hashes.
    record = {"scope": "Frozen validation fixtures only; not new research evidence.", "sources": SAMPLES,
              "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(target.iterdir()) if file.is_file()}}
    (target / "reproduction.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Prepared local-only fixtures in {target}")


if __name__ == "__main__":
    main()

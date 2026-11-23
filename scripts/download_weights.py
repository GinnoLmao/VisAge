"""Fetch a revision-pinned safetensors checkpoint and record provenance."""
from pathlib import Path
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
import argparse

ROOT = Path(__file__).resolve().parents[1]
REPO = "timm/convnextv2_pico.fcmae_ft_in1k"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "VisAge-Academic-Research/0.1"})
    return urllib.request.urlopen(request, timeout=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Exact repository commit; defaults to an existing recorded pin, otherwise resolves once.")
    args = parser.parse_args()
    directory = ROOT / ".cache" / "weights"
    directory.mkdir(parents=True, exist_ok=True)
    previous = directory / "pretrained_provenance.json"
    recorded = json.loads(previous.read_text()) if previous.exists() else {}
    requested_revision = args.revision or recorded.get("revision")
    metadata_url = f"https://huggingface.co/api/models/{REPO}"
    if requested_revision:
        metadata_url += f"/revision/{requested_revision}"
    with fetch(metadata_url) as response:
        metadata = json.load(response)
    revision = requested_revision or metadata["sha"]
    filenames = {entry["rfilename"] for entry in metadata["siblings"]}
    if "model.safetensors" not in filenames:
        raise RuntimeError("Expected safe checkpoint not present; refusing a different artifact.")
    output = directory / f"convnextv2_pico-{revision}.safetensors"
    if not output.exists():
        partial = output.with_suffix(".partial")
        with fetch(f"https://huggingface.co/{REPO}/resolve/{revision}/model.safetensors") as source, partial.open("wb") as target:
            size = 0
            while block := source.read(1024 * 1024):
                target.write(block)
                size += len(block)
                if size > 200 * 1024**2:
                    raise ValueError("Unexpected checkpoint size.")
            print(f"Downloaded checkpoint: {size / 1024**2:.1f} MiB", flush=True)
        partial.replace(output)
    with fetch(f"https://huggingface.co/{REPO}/resolve/{revision}/config.json") as response:
        config = json.load(response)
    (directory / "convnextv2_pico_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if recorded.get("revision") == revision and recorded.get("sha256") and digest != recorded["sha256"]:
        raise ValueError("Cached checkpoint differs from its recorded SHA-256.")
    provenance = {"repository": REPO, "revision": revision, "filename": output.name,
                  "relative_path": output.relative_to(ROOT).as_posix(), "sha256": digest,
                  "bytes": output.stat().st_size, "license": metadata.get("cardData", {}).get("license"),
                  "retrieved_utc": datetime.now(timezone.utc).isoformat(), "configuration": config}
    (directory / "pretrained_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()

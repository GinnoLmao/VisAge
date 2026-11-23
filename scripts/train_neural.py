import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visage.runtime import configure_caches
configure_caches(ROOT)


class TeeStream:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()


def main():
    print('{"event":"training_cli_started"}', flush=True)
    parser = argparse.ArgumentParser(description="Train one neural method from a versioned JSON config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from visage.training import train_neural
    print('{"event":"training_modules_loaded"}', flush=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["method"] == "convnext_ordinal" and not config.get("pretrained_path"):
        provenance = json.loads((ROOT / ".cache/weights/pretrained_provenance.json").read_text(encoding="utf-8"))
        config["pretrained_path"] = provenance["relative_path"]
        config["pretrained_provenance"] = provenance
    train_neural(config, resume=args.resume)


if __name__ == "__main__":
    # Log inside the supervised Python process instead of relying on a shell pipeline.
    import contextlib
    method = Path(sys.argv[1]).stem if len(sys.argv) > 1 else "neural"
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    with (report_dir / f"{method}_training.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(TeeStream(sys.stdout, log)), contextlib.redirect_stderr(TeeStream(sys.stderr, log)):
            main()

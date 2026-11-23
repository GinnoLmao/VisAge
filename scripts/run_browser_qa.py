"""Run the isolated browser integration suite in a fresh headless Chrome profile.

Uses existing local validation fixtures. Never uses a personal browser profile
or physical webcam. Starts/stops only the processes it owns.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_backend import stop_owned_server, wait_for_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/pretrained/web_app")
    args = parser.parse_args()
    if not args.browser.is_file() or not (ROOT / ".cache/web_qa/single.jpg").is_file():
        raise FileNotFoundError("Chrome and prepared local QA fixtures are required.")
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    profile = ROOT / ".cache" / f"web_qa_chrome_{uuid.uuid4().hex}"
    browser = None
    cleanup = {}
    with (folder / "headless_server.log").open("w", encoding="utf-8") as server_log, (folder / "headless_browser.log").open("w", encoding="utf-8") as browser_log:
        server = subprocess.Popen([sys.executable, "-u", str(ROOT / "tests/web/serve_camera_qa.py")], cwd=ROOT, env=environment,
                                  stdout=server_log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            with httpx.Client(base_url="http://127.0.0.1:8001", trust_env=False, timeout=30) as client:
                wait_for_server(server, client, folder / "headless_server.log", 8001, 120)
                browser = subprocess.Popen([str(args.browser), "--headless", "--no-first-run", "--disable-background-networking",
                    f"--user-data-dir={profile}", "--window-size=1280,900", "http://127.0.0.1:8001/?qa_camera=normal&qa_upload=1"],
                    stdout=browser_log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    events = client.get("/__qa__/events").json()
                    suites = [event for event in events if event.get("event") == "upload-integration"]
                    if suites:
                        (folder / "browser_events_final.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
                        if not suites[-1]["ok"]:
                            raise AssertionError(suites[-1])
                        if len(suites[-1]["passed"]) < 25:
                            raise AssertionError("Every browser integration assertion must complete.")
                        if any(event.get("event") in {"browser-error", "browser-rejection"} for event in events):
                            raise AssertionError("Uncaught browser application error.")
                        print(json.dumps(suites[-1]), flush=True)
                        break
                    if browser.poll() is not None:
                        raise RuntimeError("Headless browser exited before completing QA.")
                    time.sleep(.1)
                else:
                    raise TimeoutError("Browser QA did not finish in 120 seconds.")
        finally:
            if browser is not None:
                try:
                    parent = psutil.Process(browser.pid)
                    processes = parent.children(recursive=True) + [parent]
                except psutil.NoSuchProcess:
                    processes = []
                for process in processes:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(processes, timeout=10)
                for process in alive:
                    process.kill()
                if alive:
                    psutil.wait_procs(alive, timeout=5)
                browser.wait(timeout=5)
                cleanup["browser_process_stopped"] = browser.poll() is not None
                cleanup["remaining_browser_processes"] = [p.pid for p in processes if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
            cleanup["server"] = stop_owned_server(server, 8001)
            (folder / "headless_cleanup.json").write_text(json.dumps(cleanup, indent=2), encoding="utf-8")
            print(json.dumps(cleanup), flush=True)


if __name__ == "__main__":
    main()

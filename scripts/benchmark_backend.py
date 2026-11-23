"""Measure the real CPU backend through a temporary localhost Uvicorn process.

One frozen validation image is repeated after warm-up. This is a local latency
measurement, not camera-domain accuracy or hosted/concurrent-user performance.
No image bytes or face photos are written to the benchmark output.
"""
from __future__ import annotations

import argparse
import csv
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visage.runtime import sha256_file, utc_now, write_json

from visage.model_registry import METHODS, ARTIFACTS


def distribution_ms(values):
    """Aggregate finite durations using an explicit linear p95 interpolation."""
    values = sorted(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Timing measurements must be nonempty, finite and nonnegative.")
    position = 0.95 * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"median_ms": statistics.median(values), "p95_ms": p95,
            "mean_ms": statistics.mean(values), "measurements": len(values)}


def artifact_hashes(artifacts_dir):
    return {method: sha256_file(artifacts_dir / method / filename) for method, filename in ((method, "best.pt") for method in METHODS)}


def validation_input(sample_id=None):
    manifest = ROOT / "data/processed/manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        row = next((item for item in csv.DictReader(stream)
                    if item["split"] == "validation" and (sample_id is None or item["sample_id"] == sample_id)), None)
    if row is None:
        raise ValueError("The requested sample must belong to the frozen validation split.")
    raw = (ROOT / "data/raw/UTKFace").resolve()
    path = (raw / row["relative_path"]).resolve(strict=True)
    if not path.is_relative_to(raw) or not 0 < path.stat().st_size <= 8 * 1024**2:
        raise ValueError("Validation sample path or encoded size is invalid.")
    if sha256_file(path) != row["file_sha256"]:
        raise ValueError("The frozen validation image changed.")
    return row, path.read_bytes(), sha256_file(manifest)


def validate_prediction(result, manifest_sha, expected=None):
    if result.get("manifest_sha256") != manifest_sha:
        raise ValueError("HTTP prediction belongs to another frozen dataset.")
    predictions = result.get("predictions", [])
    if [item.get("method") for item in predictions] != list(METHODS):
        raise ValueError("The HTTP response must contain all three methods in declared order.")
    if any(type(item.get("class_id")) is not int or item["class_id"] not in {0, 1, 2}
           or item.get("manifest_sha256") != manifest_sha for item in predictions):
        raise ValueError("The backend returned invalid prediction metadata.")
    crop_sha = result["face_processing"]["pixel_sha256"]
    if not isinstance(crop_sha, str) or len(crop_sha) != 64:
        raise ValueError("A shared crop checksum is required.")
    signature = {"crop_sha256": crop_sha,
                 "labels": {item["method"]: {"class_id": item["class_id"], "age_group": item["age_group"]}
                            for item in predictions},
                 "selected_face_index": result["selected_face"]["index"]}
    if expected is not None and signature != expected:
        raise ValueError("Repeated HTTP calls produced a changed crop, selection or label.")
    return signature


def wait_for_server(process, client, log_path, port, deadline):
    started = time.perf_counter()
    while time.perf_counter() - started < deadline:
        if process.poll() is not None:
            raise RuntimeError("The temporary backend exited before becoming ready; inspect its server log.")
        # Confirm our exact child bound the requested address before accepting a
        # health response. Do not accidentally benchmark another local service.
        text = log_path.read_text(encoding="utf-8", errors="replace")[-32768:]
        import psutil
        try:
            owned = {process.pid, *[child.pid for child in psutil.Process(process.pid).children(recursive=True)]}
        except psutil.NoSuchProcess:
            owned = set()
        announced = [int(value) for value in re.findall(r"Started server process \[(\d+)\]", text)]
        bound = bool(owned.intersection(announced)) and f"http://127.0.0.1:{port}" in text
        if bound and "Application startup complete." in text:
            try:
                response = client.get("/health")
                if response.status_code == 200:
                    health = response.json()
                    if health != {"status": "ready", "models_loaded": 3, "device": "cpu"}:
                        raise ValueError("Unexpected backend readiness metadata.")
                    return health
            except (OSError, ValueError):
                raise
            except Exception as exc:
                # HTTP connection exceptions during the short bind/start window
                # are retryable; an unexpected error type is not swallowed.
                import httpx
                if not isinstance(exc, httpx.TransportError):
                    raise
        time.sleep(0.1)
    raise TimeoutError("The temporary backend did not become ready before its startup deadline.")


def stop_owned_server(process, port):
    """Stop only this Popen tree, including Windows venv launcher children."""
    import psutil
    owned = []
    if process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            owned = parent.children(recursive=True) + [parent]
        except psutil.NoSuchProcess:
            pass
    ids = [item.pid for item in owned]
    for item in owned:
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=15)
    for item in alive:
        item.kill()
    if alive:
        psutil.wait_procs(alive, timeout=5)
    process.wait(timeout=5)
    remaining = []
    for item in owned:
        try:
            if item.is_running() and item.status() != psutil.STATUS_ZOMBIE:
                remaining.append(item.pid)
        except psutil.NoSuchProcess:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(.5)
        port_closed = probe.connect_ex(("127.0.0.1", port)) != 0
    result = {"launched_pid": process.pid, "owned_process_ids": ids, "server_process_stopped": not remaining and process.poll() is not None,
              "remaining_owned_process_ids": remaining, "port_closed": port_closed,
              "exit_code_after_stop": process.returncode, "forced_kill": bool(alive)}
    if remaining or not port_closed:
        raise RuntimeError("The owned temporary server tree or listening port did not close.")
    return result


def benchmark(args):
    import httpx

    row, data, manifest_sha = validation_input(args.sample_id)
    artifacts_dir = args.artifacts_dir.resolve(strict=True)
    initial_hashes = artifact_hashes(artifacts_dir)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new benchmark output path; prior measurements are preserved.")
    if not output.is_relative_to(ROOT):
        raise ValueError("Benchmark reports must stay inside the project workspace.")
    log_path = output.parent / "backend_benchmark_server.log"
    if log_path.exists():
        raise FileExistsError("Use a new output directory to preserve the earlier server log.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fail before spawning if the requested loopback port is already occupied.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", args.port))
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "2",
                        "PYTHONUNBUFFERED": "1"})
    command = [sys.executable, "-u", str(ROOT / "scripts/serve_api.py"), "--host", "127.0.0.1",
               "--port", str(args.port), "--device", "cpu", "--artifacts-dir", str(artifacts_dir)]
    process = None
    cleanup = {}
    samples = []
    reference = None
    startup_started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=60,
                              trust_env=False, follow_redirects=False) as client:
                health = wait_for_server(process, client, log_path, args.port, args.startup_timeout)
                startup_seconds = time.perf_counter() - startup_started
                model_response = client.get("/v1/models")
                model_response.raise_for_status()
                models = model_response.json()
                if models["manifest_sha256"] != manifest_sha or models["detector"]["device"] != "cpu":
                    raise ValueError("The temporary backend loaded unexpected artifacts or detector configuration.")
                evaluation_response = client.get("/v1/evaluation")
                evaluation_response.raise_for_status()
                evaluation = evaluation_response.json()
                if any(evaluation[key]["status"] != "complete" for key in ("classification", "detection")):
                    raise ValueError("Measured backend must expose completed AP/ROC evaluation records.")
                for index in range(args.warmups + args.iterations):
                    if process.poll() is not None:
                        raise RuntimeError("The temporary backend exited during measurement.")
                    started = time.perf_counter()
                    response = client.post("/v1/predict?method=all", content=data,
                                           headers={"Content-Type": "application/octet-stream"})
                    round_trip = (time.perf_counter() - started) * 1000
                    response.raise_for_status()
                    if response.headers.get("cache-control") != "no-store":
                        raise ValueError("Prediction responses must disable storage in HTTP caches.")
                    result = response.json()
                    reference = validate_prediction(result, manifest_sha, reference)
                    if index >= args.warmups:
                        samples.append({"http_round_trip_ms": round_trip,
                                        "server_timings_ms": result["timings_ms"],
                                        "method_preprocessing_and_inference_ms": {
                                            item["method"]: item["inference_and_preprocessing_ms"] for item in result["predictions"]}})
    finally:
        if process is not None:
            cleanup = stop_owned_server(process, args.port)
    if len(samples) != args.iterations or artifact_hashes(artifacts_dir) != initial_hashes:
        raise ValueError("Measurements are incomplete or a trained artifact changed.")
    result = {
        "ok": True, "completed_utc": utc_now(), "device": "cpu", "host": "127.0.0.1", "port": args.port,
        "transport": "Real sequential HTTP requests through a temporary one-process Uvicorn backend",
        "startup_seconds": startup_seconds, "warmups": args.warmups, "iterations": args.iterations,
        "sample_id": row["sample_id"], "input_sha256": row["file_sha256"], "input_bytes": len(data),
        "input_partition": "validation", "manifest_sha256": manifest_sha,
        "artifact_sha256": initial_hashes, "detector": models["detector"], "health": health,
        "consistent_result": reference,
        "http_round_trip": distribution_ms([item["http_round_trip_ms"] for item in samples]),
        "server_timings": {key: distribution_ms([item["server_timings_ms"][key] for item in samples])
                           for key in samples[0]["server_timings_ms"]},
        "method_preprocessing_and_inference": {
            method: distribution_ms([item["method_preprocessing_and_inference_ms"][method] for item in samples])
            for method in METHODS},
        "samples": samples,
        "environment": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                        "logical_cpu_count": os.cpu_count(), "api_torch_threads": 4,
                        "child_omp_threads": 4, "child_mkl_threads": 4, "child_openblas_threads": 2,
                        "packages": {name: metadata.version(name) for name in ("torch", "fastapi", "uvicorn", "httpx", "opencv-python-headless")}},
        "no_concurrent_training": args.training_idle_confirmed,
        "evaluation_endpoint_check": {"status_code": evaluation_response.status_code,
            "classification_status": evaluation["classification"]["status"], "detection_status": evaluation["detection"]["status"],
            "methods": list(evaluation["classification"]["methods"]),
            "detection_ap_percent": {name: value["ap_percent"] for name, value in evaluation["detection"]["metrics"].items()}},
        "idle_verification": "Operator confirmation before launch; this script does not inspect operating-system training processes.",
        "cleanup": cleanup,
        "server_log": {"path": log_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(log_path)},
        "scope": "One existing pre-aligned validation face image repeated. Measures raw-byte admission, detection, shared crop/quality and all three classifiers. Not a real-camera accuracy, concurrent-load, cold-inference or hosted-network benchmark.",
        "privacy": "No uploaded bytes or face photos written; benchmark keeps only checksums, timing and structured outputs.",
    }
    if not cleanup.get("server_process_stopped"):
        raise RuntimeError("The temporary server process did not stop.")
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", default=30, type=int)
    parser.add_argument("--warmups", default=5, type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pretrained/benchmark/backend_benchmark.json")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--sample-id")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--training-idle-confirmed", action="store_true",
                        help="Confirm training has finished before measuring latency; this is an operator assertion, not an OS process check.")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0 or not 1 <= args.port <= 65535:
        parser.error("Use positive iterations, nonnegative warmups and a valid TCP port.")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("Startup timeout must be finite and positive.")
    if not args.training_idle_confirmed:
        parser.error("Finish training first, then pass --training-idle-confirmed to record the idle-context assertion.")
    result = benchmark(args)
    print(json.dumps({"ok": result["ok"], "output": str(args.output),
                      "http_round_trip": result["http_round_trip"], "startup_seconds": result["startup_seconds"],
                      "server_stopped": result["cleanup"]["server_process_stopped"]}, indent=2))


if __name__ == "__main__":
    main()

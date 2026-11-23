"""Benchmark aggregation and response identity checks without starting a server."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("backend_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_backend.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def prediction():
    return {"manifest_sha256": "a" * 64, "selected_face": {"index": 0},
            "face_processing": {"pixel_sha256": "b" * 64},
            "predictions": [{"method": method, "class_id": 1, "age_group": "18-59", "manifest_sha256": "a" * 64}
                            for method in benchmark.METHODS]}


def test_aggregation_has_known_interpolated_p95_and_singleton_behavior():
    result = benchmark.distribution_ms([100, 0, 30, 10])
    assert result["median_ms"] == 20 and result["mean_ms"] == 35
    assert result["p95_ms"] == pytest.approx(89.5)
    assert result["measurements"] == 4
    assert benchmark.distribution_ms([7]) == {"median_ms": 7, "p95_ms": 7, "mean_ms": 7, "measurements": 1}


@pytest.mark.parametrize("values", [[], [-1], [float("nan")], [float("inf")]])
def test_nonfinite_or_invalid_measurements_are_not_published(values):
    with pytest.raises(ValueError):
        benchmark.distribution_ms(values)


def test_repeated_response_identity_ignores_timings_but_checks_crop_and_labels():
    original = prediction()
    reference = benchmark.validate_prediction(original, "a" * 64)
    assert benchmark.validate_prediction(deepcopy(original), "a" * 64, reference) == reference
    for change in ("crop", "label", "selection", "manifest"):
        changed = deepcopy(original)
        if change == "crop": changed["face_processing"]["pixel_sha256"] = "c" * 64
        elif change == "label": changed["predictions"][1]["class_id"] = 2
        elif change == "selection": changed["selected_face"]["index"] = 1
        else: changed["manifest_sha256"] = "c" * 64
        with pytest.raises(ValueError):
            benchmark.validate_prediction(changed, "a" * 64, reference)


def test_all_three_prediction_metadata_records_are_required():
    for change in ("missing_method", "bad_class", "wrong_artifact_freeze"):
        changed = prediction()
        if change == "missing_method": changed["predictions"].pop()
        elif change == "bad_class": changed["predictions"][0]["class_id"] = 3
        else: changed["predictions"][0]["manifest_sha256"] = "c" * 64
        with pytest.raises(ValueError):
            benchmark.validate_prediction(changed, "a" * 64)
def test_owned_python_tree_and_port_are_cleaned_up():
    import os
    import socket
    import subprocess
    import sys
    import time
    import psutil
    from pathlib import Path
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/benchmark_backend.py'
    spec=importlib.util.spec_from_file_location('backend_cleanup_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0))
        port=probe.getsockname()[1]
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        deadline=time.monotonic()+3
        while os.name=='nt' and time.monotonic()<deadline and not psutil.Process(process.pid).children():
            time.sleep(.02)
        result=module.stop_owned_server(process,port)
        assert result['server_process_stopped'] and result['port_closed']
        assert process.pid in result['owned_process_ids'] and not result['remaining_owned_process_ids']
    finally:
        if process.poll() is None:
            module.stop_owned_server(process,port)


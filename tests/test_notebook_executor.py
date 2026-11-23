"""Socket-free executor probes use real cell, child-process and figure output."""

import base64
import importlib.util
from pathlib import Path
import subprocess

import nbformat
import pytest


spec = importlib.util.spec_from_file_location(
    "visage_notebook_executor", Path(__file__).resolve().parents[1] / "scripts/execute_notebook.py"
)
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)


def notebook(*sources):
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source) for source in sources])


def test_python_cells_share_state_capture_real_subprocess_and_figure():
    document = notebook(
        "shared = 7\nprint('first actual stream')",
        "import subprocess, sys\n"
        "print('shared value', shared)\n"
        "subprocess.run([sys.executable, '-c', \"print('actual child output')\"], check=True)\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2], [shared, shared + 1])\nplt.show()",
    )
    original_run = subprocess.run
    state = executor.execute_python_cells(document, source_name="two_cell_probe.ipynb")
    assert state["shared"] == 7
    assert [cell.execution_count for cell in document.cells] == [1, 2]
    streams = "".join(output.text for cell in document.cells for output in cell.outputs
                      if output.output_type == "stream")
    assert "first actual stream" in streams
    assert "shared value 7" in streams
    assert "actual child output" in streams
    figures = [output for output in document.cells[1].outputs if output.output_type == "display_data"]
    assert len(figures) == 1
    assert base64.b64decode(figures[0].data["image/png"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert document.metadata["visage"]["execution_engine"] == "python_cells"
    assert document.metadata["visage"]["colab_verified"] is False
    assert subprocess.run is original_run
    nbformat.validate(document)


def test_python_cells_preserve_actual_failure_and_do_not_continue():
    document = notebook(
        "shared = 13\nprint('completed first cell')",
        "print('before actual failure', shared)\nraise ValueError('probe failure')",
        "print('must not execute')",
    )
    original_run = subprocess.run
    with pytest.raises(ValueError, match="probe failure"):
        executor.execute_python_cells(document, source_name="failure_probe.ipynb")
    assert [cell.execution_count for cell in document.cells] == [1, 2, None]
    outputs = document.cells[1].outputs
    assert outputs[0].output_type == "stream" and "before actual failure 13" in outputs[0].text
    assert outputs[-1].output_type == "error"
    assert outputs[-1].ename == "ValueError" and outputs[-1].evalue == "probe failure"
    assert "failure_probe.ipynb:cell-2" in "".join(outputs[-1].traceback)
    assert not document.cells[2].outputs
    assert subprocess.run is original_run
    nbformat.validate(document)


def test_python_engine_rejects_magics_before_any_execution():
    document = notebook("print('not run')", "%time 2 + 2")
    with pytest.raises(SyntaxError):
        executor.execute_python_cells(document)
    assert all(cell.execution_count is None for cell in document.cells)


def test_python_cells_capture_subprocess_failure_streams_and_restore_process_state(tmp_path):
    original_directory = Path.cwd()
    document = notebook(
        "import os, subprocess, sys\n"
        f"os.chdir({str(tmp_path)!r})\n"
        "subprocess.run([sys.executable, '-c', "
        "\"import sys; print('real failure stdout'); print('real failure stderr', file=sys.stderr); sys.exit(3)\"], check=True)",
        "print('unreachable')",
    )
    with pytest.raises(subprocess.CalledProcessError) as captured:
        executor.execute_python_cells(document)
    assert captured.value.returncode == 3
    assert Path.cwd() == original_directory
    streams = {name: "".join(output.text for output in document.cells[0].outputs
                              if output.output_type == "stream" and output.name == name)
               for name in ("stdout", "stderr")}
    assert "real failure stdout" in streams["stdout"]
    assert "real failure stderr" in streams["stderr"]
    assert document.cells[0].outputs[-1].ename == "CalledProcessError"
    assert document.cells[1].execution_count is None

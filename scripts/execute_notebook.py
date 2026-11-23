"""Execute the model notebook in artifact-only mode with truthful local evidence.

The nbclient engine uses a project-local Jupyter kernel. The python engine runs
plain Python code cells directly in a shared namespace without sockets. Neither
engine starts full model training or installs dependencies. Neither establishes
a Colab run. Kernel metadata, when used, stays under .cache/jupyter.
"""

from __future__ import annotations

import argparse
import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import linecache
import locale
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_project_environment(*, kernel_required: bool = False) -> Path:
    """Require the project interpreter without installing or launching anything."""
    environment_root = (ROOT / ".venv").resolve()
    expected = environment_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # sys.executable may resolve a POSIX venv symlink to the base interpreter;
    # sys.prefix retains the actual active virtual environment identity.
    if Path(sys.prefix).resolve() != environment_root or not expected.is_file():
        raise RuntimeError(
            "Run this script with the project's .venv interpreter; refusing a "
            "kernel bound to an unrelated Python environment."
        )
    packages = ("nbformat", "nbclient", "ipykernel", "jupyter_client") if kernel_required else ("nbformat", "matplotlib")
    missing = [name for name in packages
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Install project dev dependencies first. Missing: " + ", ".join(missing))
    return expected


def configure_artifact_environment() -> None:
    os.environ.update({
        "VISAGE_PROJECT_ROOT": str(ROOT), "VISAGE_NOTEBOOK_MODE": "artifacts",
        "VISAGE_NOTEBOOK_INSTALL": "0", "VISAGE_NOTEBOOK_RESUME": "0",
    })


def configure_local_kernel() -> tuple[Path, Path]:
    """Write a local kernelspec pointing to this exact project interpreter."""
    expected = check_project_environment(kernel_required=True)
    jupyter_root = ROOT / ".cache/jupyter"
    directory = jupyter_root / "kernels/visage"
    directory.mkdir(parents=True, exist_ok=True)
    specification = {
        "argv": [str(expected), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "VisAge project virtual environment",
        "language": "python",
        "metadata": {"debugger": False},
        "env": {
            "VISAGE_PROJECT_ROOT": str(ROOT),
            "VISAGE_NOTEBOOK_MODE": "artifacts",
            "VISAGE_NOTEBOOK_INSTALL": "0",
            "VISAGE_NOTEBOOK_RESUME": "0",
        },
    }
    path = directory / "kernel.json"
    path.write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    existing = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(jupyter_root) + (os.pathsep + existing if existing else "")
    for name, subdirectory in (
        ("JUPYTER_CONFIG_DIR", "config"),
        ("JUPYTER_DATA_DIR", "data"),
        ("JUPYTER_RUNTIME_DIR", "runtime"),
        ("IPYTHONDIR", "ipython"),
    ):
        target = jupyter_root / subdirectory
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    os.environ.update(specification["env"])
    return jupyter_root, path


class _NotebookStream(io.TextIOBase):
    """Append actual writes in stream/display order to a notebook cell."""

    def __init__(self, outputs: list, name: str):
        self.outputs, self.name = outputs, name

    @property
    def encoding(self):
        return "utf-8"

    def write(self, text):
        import nbformat
        if not isinstance(text, str):
            raise TypeError("Notebook stream writes must be text.")
        if text:
            if self.outputs and self.outputs[-1].output_type == "stream" and self.outputs[-1].name == self.name:
                self.outputs[-1].text += text
            else:
                self.outputs.append(nbformat.v4.new_output("stream", name=self.name, text=text))
        return len(text)

    def flush(self):
        return None

    def isatty(self):
        return False


def execute_python_cells(notebook, *, source_name: str = "notebook.ipynb",
                         on_cell_start=None, on_cell_executed=None) -> dict:
    """Execute plain Python cells in order, preserving real streams and failures.

    There is no Jupyter kernel, socket, IPython magic expansion, or training
    shortcut. Exceptions stop execution after the actual failing cell is saved.
    The caller writes the notebook in its own finally block.
    """
    import nbformat
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{source_name}:cell-{index + 1}")
            cell.outputs = []
            cell.execution_count = None
    namespace = {"__name__": "__main__", "__file__": str(source_name)}
    original_run, original_show = subprocess.run, plt.show
    current_outputs = None
    executed = 0

    def captured_show(*args, **kwargs):
        for figure_number in list(plt.get_fignums()):
            figure = plt.figure(figure_number)
            stream = io.BytesIO()
            figure.savefig(stream, format="png", dpi=130)
            if current_outputs is not None:
                current_outputs.append(nbformat.v4.new_output(
                    "display_data",
                    data={"image/png": base64.b64encode(stream.getvalue()).decode("ascii"),
                          "text/plain": f"Matplotlib figure {figure_number}"},
                    metadata={},
                ))
            plt.close(figure)

    def emit(value, *, error=False):
        if value is None:
            return
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                value = value.decode(locale.getpreferredencoding(False), errors="replace")
        target = sys.stderr if error else sys.stdout
        target.write(value)

    def captured_run(*args, **kwargs):
        # Respect callers that explicitly own output (capture_output/PIPE/file).
        if kwargs.get("capture_output", False):
            return original_run(*args, **kwargs)
        capture_stdout = kwargs.get("stdout") is None
        capture_stderr = kwargs.get("stderr") is None and not (
            kwargs.get("stdout") is not None and kwargs.get("stderr") == subprocess.STDOUT
        )
        if capture_stdout:
            kwargs["stdout"] = subprocess.PIPE
        if capture_stderr:
            kwargs["stderr"] = subprocess.PIPE
        try:
            result = original_run(*args, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            if capture_stdout:
                emit(error.stdout)
                error.stdout = None
            if capture_stderr:
                emit(error.stderr, error=True)
                error.stderr = None
            raise
        if capture_stdout:
            emit(result.stdout)
            result.stdout = None
        if capture_stderr:
            emit(result.stderr, error=True)
            result.stderr = None
        return result

    previous_cwd = Path.cwd()
    subprocess.run, plt.show = captured_run, captured_show
    notebook.metadata["visage"] = nbformat.from_dict(dict(notebook.metadata.get("visage", {})))
    notebook.metadata["visage"].update(
        execution_engine="python_cells", execution_host="local", colab_verified=False,
    )
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if on_cell_start is not None:
                on_cell_start(cell=cell, cell_index=index)
            executed += 1
            cell.execution_count = executed
            current_outputs = cell.outputs
            filename = f"{source_name}:cell-{index + 1}"
            linecache.cache[filename] = (len(cell.source), None, cell.source.splitlines(True), filename)
            cell.metadata["visage_execution"] = {
                "engine": "python_cells", "started_utc": utc_now(),
            }
            try:
                with redirect_stdout(_NotebookStream(cell.outputs, "stdout")), redirect_stderr(_NotebookStream(cell.outputs, "stderr")):
                    exec(compile(cell.source, filename, "exec"), namespace, namespace)
                    if plt.get_fignums():
                        captured_show()
            except BaseException as error:
                cell.outputs.append(nbformat.v4.new_output(
                    "error", ename=type(error).__name__, evalue=str(error),
                    traceback=traceback.format_exception(type(error), error, error.__traceback__),
                ))
                raise
            finally:
                cell.metadata["visage_execution"]["completed_utc"] = utc_now()
            if on_cell_executed is not None:
                on_cell_executed(cell=cell, cell_index=index)
    finally:
        subprocess.run, plt.show = original_run, original_show
        plt.close("all")
        os.chdir(previous_cwd)
    return namespace


def require_artifacts() -> None:
    required = [
        "data/provenance.json", "data/processed/manifest.csv",
        "data/processed/manifest.freeze.json", "data/processed/audit.json",
        "artifacts/classical/selection.json", "artifacts/classical/svm.npz",
        "artifacts/classical/svm.json", "artifacts/classical/features.json",
    ]
    for method in ("visagenet", "convnext_ordinal"):
        required.extend(f"artifacts/{method}/{name}" for name in (
            "best.pt", "training_summary.json", "validation_metrics.json", "history.json"
        ))
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Artifact-only execution requires completed runs. Missing files:\n" + "\n".join(missing)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "notebooks/visage_phase_models.ipynb")
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/visage_phase_models_executed.ipynb")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/notebook_runner.json")
    parser.add_argument("--cell-timeout", type=int, default=1800)
    parser.add_argument("--engine", choices=("nbclient", "python"), default="nbclient",
                        help="Use python for plain code-cell execution without Jupyter sockets.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--prepare-only", action="store_true", help="Write/check the local kernel without executing cells.")
    arguments = parser.parse_args(argv)
    if arguments.cell_timeout < 1:
        parser.error("--cell-timeout must be positive.")
    source, output, report_path = arguments.source.resolve(), arguments.output.resolve(), arguments.report.resolve()
    if source == output:
        parser.error("Output must differ from the source notebook.")
    for path in (source, output, report_path):
        if not path.is_relative_to(ROOT):
            parser.error("Source, output and report must stay inside the project workspace.")
    kernel_path = None
    if arguments.engine == "nbclient":
        _, kernel_path = configure_local_kernel()
    else:
        check_project_environment()
        configure_artifact_environment()
    os.environ["VISAGE_EVALUATION_SPLIT"] = arguments.split
    import nbformat

    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    if arguments.engine == "nbclient":
        from jupyter_client.kernelspec import KernelSpecManager
        notebook.metadata["kernelspec"] = {
            "name": "visage", "display_name": "VisAge project virtual environment", "language": "python",
        }
        resolved_kernel = KernelSpecManager().get_kernel_spec("visage")
        if Path(resolved_kernel.resource_dir).resolve() != kernel_path.parent:
            raise RuntimeError("Kernel lookup did not resolve to the project-local kernelspec.")
    else:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{source}:cell-{index + 1}")
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    if arguments.prepare_only:
        print(json.dumps({"prepared": True, "executed": False,
                          "kernel": str(kernel_path) if kernel_path else None, "engine": arguments.engine,
                          "source": str(source), "code_cells": code_cells, "mode": "artifacts"}, indent=2))
        return 0

    require_artifacts()
    started_utc, started = utc_now(), time.perf_counter()
    progress = {"current_cell": None, "completed_cells": 0}
    report = {
        "started_utc": started_utc, "execution_host": "local", "mode": "artifacts",
        "training_started": False, "dependency_installation": False,
        "evaluation_split": arguments.split, "source": source.relative_to(ROOT).as_posix(),
        "output": output.relative_to(ROOT).as_posix(),
        "kernel": kernel_path.relative_to(ROOT).as_posix() if kernel_path else None,
        "execution_engine": "python_cells" if arguments.engine == "python" else "nbclient",
        "jupyter_kernel_executed": arguments.engine == "nbclient",
        "cell_timeout_enforced": arguments.engine == "nbclient",
        "python_executable": str(sys.executable), "code_cells": code_cells,
        "status": "running", "colab_verified": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def write_report() -> None:
        report_path.write_text(json.dumps({**report, **progress}, indent=2) + "\n", encoding="utf-8")

    def on_cell_start(cell, cell_index, **kwargs) -> None:
        progress["current_cell"] = cell_index + 1
        print(json.dumps({"event": "cell_started", "cell": cell_index + 1,
                          "kind": cell.cell_type}), flush=True)
        write_report()

    def on_cell_executed(cell, cell_index, **kwargs) -> None:
        progress["completed_cells"] += 1
        print(json.dumps({"event": "cell_executed", "cell": cell_index + 1}), flush=True)
        write_report()

    write_report()
    failure = None
    try:
        if arguments.engine == "python":
            execute_python_cells(notebook, source_name=str(source),
                                 on_cell_start=on_cell_start, on_cell_executed=on_cell_executed)
        else:
            from nbclient import NotebookClient
            client = NotebookClient(
                notebook, timeout=arguments.cell_timeout, kernel_name="visage",
                startup_timeout=30,
                resources={"metadata": {"path": str(ROOT)}}, allow_errors=False,
                record_timing=True, on_cell_start=on_cell_start, on_cell_executed=on_cell_executed,
            )
            client.execute()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        failure = error
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        report.update(completed_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
        report["executed_code_cells"] = sum(
            cell.cell_type == "code" and cell.execution_count is not None for cell in notebook.cells
        )
        report["cell_errors"] = sum(
            output_item.output_type == "error" for cell in notebook.cells
            if cell.cell_type == "code" for output_item in cell.get("outputs", [])
        )
        write_report()
        print(json.dumps(report, indent=2), flush=True)
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

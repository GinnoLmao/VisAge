"""Write active-model guidance from completed evidence; archive the original README."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.model_registry import METHODS, REGISTRY, ARTIFACTS


def main():
    reports = ROOT / 'reports/pretrained'
    selection = json.loads((reports / 'selection.json').read_text())
    default = selection['default_method']
    backup = reports / 'v1_sources/README.md'
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes((ROOT / 'README.md').read_bytes())
    models = '\n'.join(f'| `{m}` | {REGISTRY[m].name} | `{REGISTRY[m].checkpoint}` |' for m in METHODS)
    tests = len(ET.parse(reports / 'web_app/tests_full.xml').findall('.//testcase'))
    javascript = len(ET.parse(reports / 'web_app/tests_javascript.xml').findall('.//testcase'))
    feature_path = reports / 'ui_refinement/acceptance.json'
    if not feature_path.exists():
        feature_path = reports / 'score_feature/acceptance.json'
    feature = json.loads(feature_path.read_text()) if feature_path.exists() else None
    if feature:
        if not feature['ok']:
            raise ValueError('The score-display extension must pass before documenting it as complete.')
        tests, javascript = feature['counts']['tests_full.xml'], feature['counts']['tests_javascript.xml']
    hardware_path = reports / 'hardware_acceptance.json'
    hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else None
    hardware_statement = 'Physical webcam and real-phone acceptance remain pending.'
    if hardware and hardware['webcam']['status'] == 'user_confirmed_working' and hardware['phone']['status'] == 'not_required_by_user':
        hardware_statement = 'The user has confirmed that the physical webcam works correctly. Phone testing is not required by the user.'
    text = f'''# VisAge pretrained age group web application

VisAge estimates **Under 18**, **18–59**, or **60+** from one selected face and compares three published pretrained vision backbones. It supports staff review of age-related ID checks for purchases or senior-discount requests. Actual age and eligibility require the applicable identification; the app does not approve transactions or recognize identity.

| API ID | Model | Published timm checkpoint |
| --- | --- | --- |
{models}

The active default is **{REGISTRY[default].name}** (`{default}`). Selection uses mean validation macro F1 across **seeds 42 and 43**, with the original near-tie boundary-error rules. The approved scope is six completed runs. Seed44 is excluded and the nine frozen-backbone/data-efficiency jobs are deferred. Each family serves its **seed42 best-validation checkpoint**. Nano's selected checkpoint is from head-only warm-up; the EfficientNet and Swin serving checkpoints are from fine-tuning. Subsequent training never replaces a better validation checkpoint merely because it ran later.

## Start the local app

From the project directory in PowerShell:

```powershell
.\\.venv\\Scripts\\python.exe scripts/serve_api.py --device cpu
```

Open **http://127.0.0.1:8000**. To use the available NVIDIA GPU for inference, choose `--device cuda`. The active workspace contains the three selected age classifiers, pinned YuNet detector and offline dashboard evidence. From a source-only GitHub clone, first install `requirements.txt` and run `python scripts/download_runtime.py` to restore the checksum-pinned runtime release. It requires a Python environment with the declared dependencies. Raw research images and original published weight downloads are needed to reproduce research, but are unnecessary for ordinary app inference. The versioned local delivery ZIP includes the serving models, detector, active source and offline dashboard evidence.

The frontend uses native HTML, CSS, JavaScript ES modules and SVG on the same FastAPI origin. It has no frontend build step, external JavaScript CDN or external font service. Node is needed only for JavaScript tests.

- Upload or drag and drop JPEG, PNG, WebP, HEIC/HEIF, AVIF, BMP or a single-page TIFF up to 8 MiB and 20 megapixels. A bounded JPEG preview makes additional formats visible in browsers while inference uses the original oriented pixels. Select a face when several are detected.
- Run one model or **Compare all three**. Every model receives the identical selected face. Cards display estimated age groups and measured processing times. The selected overlay shows the estimate and flags disagreement in a comparison.
- Camera capture uses the same API path. Capture stops video tracks; retake restarts preview. Switching, device/permission errors and automatic shutdown on route changes are supported. Browser camera use requires localhost or HTTPS.
- Changing the photo, face or method clears stale results and overlay estimates. Retry, reset, cancellation and bounded server admission handle interrupted work.
- Every result shows all three age-group softmax scores in fixed class order, as percentage labels and bars. The chosen group has the highest score; there is no next-highest callout. These are uncalibrated model scores, not verified probabilities of actual age. Values below 0.01% remain visibly nonzero instead of being rounded to zero. The API and both inference commands preserve the full numeric scores.
- The Evaluation view shows aligned-validation accuracy, macro F1, weighted F1, mean AUC, runtime (ms), parameters (millions), complete seed-42 training-run time (minutes), per-class metrics, confusion matrices, ROC/AUC, detector AP/PR/FROC and raw-pipeline coverage. Runtime is the recorded warm CPU median per classifier; shared face processing and HTTP overhead are excluded. The rejected-input reasons table is omitted from the UI, while denominator accounting remains in the evidence.

Images are processed in memory and are not saved or added to training. {hardware_statement} The user excluded pilot photo testing and further model improvements from the current delivery scope; no independent pilot result is claimed. Webcam functionality and simulated camera checks do not establish real-camera age-classification accuracy.

## Environment

The recorded environment uses Python 3.12, PyTorch with CUDA12.8, timm1.0.31 and an RTX4050 Laptop GPU with about6 GiB VRAM. Exact versions and pilot measurements are in `reports/pretrained/gpu_pilot.json` and the frozen protocol. The Windows CUDA pair below matches the verified local installation. Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) for a different platform or CPU-only build.

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\\.venv\\Scripts\\python.exe -m pip install -e ".[dev]"
.\\.venv\\Scripts\\python.exe -m pip check
.\\.venv\\Scripts\\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); d='cuda' if torch.cuda.is_available() else 'cpu'; x=torch.randn(32,32,device=d); print((x@x).isfinite().all().item())"
```

`reports/pretrained/requirements-lock-pretrained-windows-cu128.txt` records the migrated dependencies and excludes the local project itself. Install the CUDA framework pair first when restoring that lock, then install the project with `--no-deps --no-build-isolation -e .`. The two earlier lock files remain historical records. Training verifies CUDA/BF16 availability before running. This milestone verifies the existing local environment; a new delivery archive, extraction check and clean-machine dependency restoration remain separate work. Colab is not required.

## Common input and transfer learning

The fixed class order is 0 Under18, 1 ages18–59, 2 ages60+. All three heads emit three logits and use the same class-weighted cross-entropy objective. Weights are normalized inverse-square-root training class counts, with no second balancing sampler. Softmax is used for classification scores and ROC, not as calibrated certainty for a person.

Inputs follow EXIF orientation, RGB conversion, whole-face Pillow bicubic resizing to224 and then256 pixels. Raw photos first use the unchanged YuNet five-point alignment to a224-pixel canonical crop. No extra center crop changes the compared region. ConvNeXt/Swin retain ImageNet mean/std; the selected TensorFlow EfficientNet checkpoint uses mean/std0.5. This256-pixel recipe adapts the published ImageNet preprocessing and does not reproduce its leaderboard setting.

Training uses a physical batch32, channels-last layout, CUDA BF16, deterministic settings, AdamW, two head-only warm-up epochs, then unfreezing, and a fixed cosine schedule. FP32 validation chooses the best epoch, with minimum8/maximum24 epochs and patience5. Frozen backbones remain in evaluation mode. Checkpoints contain optimizer and RNG state for exact epoch-boundary continuation.

## Dataset and evidence

The frozen UTKFace mirror has **16,300 training / 3,492 validation / 3,493 test** retained images. `data/provenance.json` records the jangedoo UTKFace Kaggle mirror, version, archive checksum and extraction scope. The unavailable original download has no author checksum proving byte-for-byte mirror identity. Ages are dataset annotations, not verified birth-record ages. Perceptual duplicate groups do not establish subject identity because UTKFace has no participant IDs.

The user approved **two seeds per model**, yielding **six completed full-data runs**. The original18-run protocol remains preserved; `configs/pretrained/two_seed_amendment.json` records the reduced scope after validation results were observed. The unfinished Nano seed44 run and two unstarted seed44 runs are excluded. Three frozen-backbone and six data-efficiency jobs are deferred, with no results claimed. Report both seeds, their mean and sample standard deviation; two seeds provide limited replication. Robustness covers seven conditions on the full validation split for each serving family. Detector-only AP is reused with pinned bytes, benchmark/configuration/annotation provenance and twelve exact reload checks; it is not new detector-training evidence or the official WIDER difficulty leaderboard protocol.

The existing test partition was observed earlier in the project. The new models were evaluated after freezing the validation-only default decision; these are results on an existing benchmark, not independent confirmation. No test-guided tuning is performed. The user excluded independent pilot photo testing and further model improvement from the delivery scope. No independent real-camera accuracy result is claimed.

Key files:

- `configs/pretrained/protocol.json`: fixed run matrix, source/input hashes and decision rules.
- `configs/pretrained/two_seed_amendment.json`: approved six-run scope, exclusions, deferred jobs and preserved original evidence hashes.
- `reports/pretrained/verified_runs.json`: six verified completed runs; all validation labels reproduced after checkpoint reload.
- `reports/pretrained/selection.json`: the validation-only default decision.
- `reports/pretrained/test_summary.json`, `robustness.json`, `face_pipeline_validation.json`: separate evaluation scopes.
- `reports/pretrained/evaluation/classification_roc.json`: current serving-model dashboard evidence.
- `reports/pretrained/evaluation/detection_lineage.json`: reused detector AP provenance.
- `artifacts/pretrained/serving/bundle.json`: default and three serving artifact hashes.
- `reports/pretrained/figures_two_seed`: the six real training curves, confusion matrices, ROC and robustness plots.
- `reports/pretrained/two_seed_report.html`: self-contained supervision report, with metric CSVs beside it.
- `reports/pretrained/score_feature`: score-display changes, unchanged-weight evidence, regression results and browser checks.
- `reports/pretrained/hardware_acceptance.json`: user-confirmed webcam status and exclusion of phone testing.

## Reproduce the experiments

Training is stopped at the user's request. Do not run the historical18-job training controller to reproduce the approved six-run analysis. The inference-only command below verifies the amended scope and saved artifacts, reproduces validation predictions, selects the default, and resumes existing evaluation evidence. It requires the complete research workspace and frozen dataset; it does not train. Keep original source/data bytes and partitions unchanged.

```powershell
.\\.venv\\Scripts\\python.exe scripts/finalize_pretrained_two_seed.py
```

Completed evidence is checksum-verified. The existing final test pass is retained rather than rerun for model selection; durable progress supports interrupted evaluation. Config/source/input changes require an explicit amendment and output version. No further training is authorized by these commands.

## API and command line

`GET /health`, `/v1/models`, `/v1/evaluation`; `POST /v1/faces` and `/v1/predict` accept raw supported still-image bytes rather than multipart form data. Use `?method=all&face_index=0` for a comparison. Omitting the method uses the bundle default. The old classical/custom IDs are not aliases for these models.

```powershell
.\\.venv\\Scripts\\python.exe scripts/predict_image.py photo.jpg --method all
.\\.venv\\Scripts\\python.exe scripts/predict.py aligned_face.png --method all
```

Both inference adapters, the API and pilot runner use the active three-model bundle. The aligned-face command assumes an already selected face; the raw-image command performs detection, selection and quality checks.

## Verification and delivery

The current code passed **{tests} Python tests** and **{javascript} JavaScript tests**, with real-artifact HTTP/CLI and browser integration records. The latest interface and image-format checks are in `reports/pretrained/ui_refinement`; the score-display checks are in `reports/pretrained/score_feature`. Earlier activation, device consistency and latency evidence remain under `reports/pretrained/web_app`, `device_verification.json` and `benchmark/backend_benchmark.json`. User-confirmed webcam operation is recorded separately from automated tests. Pilot photo testing was excluded by the user and is not counted as completed.

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
node --test tests/web/core.test.mjs
.\\.venv\\Scripts\\python.exe scripts/verify_web_app.py
```

The current local delivery is `output/VisAge_Complete_Web_App_v0.4.0.zip`. Its checksum inventory and isolated-extraction verification are recorded under `reports/pretrained/delivery`. The full system report is delivered alongside the ZIP and published under `docs/VisAge_System_Report.html` and `docs/VisAge_System_Report.pdf`. The historical `output/VisAge_Web_App.zip` remains the separate prior-version rollback. Inspect the recorded verification scope: fresh extraction uses the verified existing dependencies and does not claim a separate clean-machine install.

Historical artifacts/reports and `output/VisAge_Web_App.zip` remain preserved as the v1 rollback. Original research modules/commands are retained for their historical experiments; they do not select the active API models. Restore the separate rollback archive to an isolated folder to run that version.

## Public source and hosting

The public repository is [GinnoLmao/VisAge](https://github.com/GinnoLmao/VisAge). The [GitHub Pages portal](https://ginnolmao.github.io/VisAge/) hosts documentation and results, and the versioned release provides large inference weights separately from Git history. See [Deployment.md](docs/Deployment.md) for Vercel, local and container setup. GitHub Pages does not execute Python inference. The Vercel adapter has a 4 MiB upload cap; the local app retains 8 MiB. Hosting completion is reported only after real endpoint checks.

## Attribution and permissions

`licenses/pretrained` contains the exact publisher model cards, immutable revisions, provenance and archived license texts: CC BY-NC4.0 for the selected ConvNeXt weights, Apache2.0 for EfficientNet weights and MIT for Swin weights. YuNet's MIT license is next to its ONNX artifact. Dataset and original-photo rights remain separate. The academic prototype does not grant permission for commercial deployment or public redistribution of research photos.
'''
    (ROOT / 'README.md').write_text(text, encoding='utf-8')
    print('Updated the active README from completed evidence.', flush=True)


if __name__ == '__main__':
    main()

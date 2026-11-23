"""Prepare the app switch; activate only after verified serving evidence exists."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.model_registry import METHODS, REGISTRY, ARTIFACTS
from visage.runtime import sha256_file, write_json, utc_now

STAGE = ROOT / '.cache/pretrained_app_stage'
BACKUP = ROOT / 'reports/pretrained/v1_sources'


def prepare():
    originals = {}
    changed = {}
    def source(relative):
        path = ROOT / relative
        backup = BACKUP / relative
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(path.read_bytes())
        originals[relative] = sha256_file(backup)
        return backup.read_text(encoding='utf-8')
    def put(relative, text):
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        changed[relative] = sha256_file(target)
    def replace(text, old, new):
        if old not in text:
            raise ValueError(f'Expected original source fragment is missing: {old[:100]}')
        return text.replace(old, new)

    text = source('src/visage/service.py')
    text = replace(text, 'from .inference import AgeGroupPredictor\nfrom .deployment import default_artifacts_dir',
        'from .pretrained_inference import PretrainedPredictor\nfrom .model_registry import METHODS, REGISTRY, ARTIFACTS, serving_default')
    text = replace(text, 'METHODS = ("classical", "visagenet", "convnext_ordinal")\nDEFAULT_METHOD = "visagenet"',
        'DEFAULT_METHOD = serving_default()')
    text = replace(text, 'predictor_factory=AgeGroupPredictor', 'predictor_factory=PretrainedPredictor')
    text = replace(text, 'else default_artifacts_dir()', 'else ARTIFACTS')
    text = replace(text, 'self.device = str(device)',
        'self.device = str(device)\n        self.default_method = serving_default(artifacts_dir)')
    text = replace(text, 'self.manifest_sha256 = freezes.pop()',
        'self.manifest_sha256 = freezes.pop()\n            if predictor_factory is PretrainedPredictor:\n                import json\n                bundle = json.loads((artifacts_dir / "bundle.json").read_text(encoding="utf-8"))\n                if set(bundle["methods"]) != set(METHODS):\n                    raise ValueError("Incomplete serving bundle.")\n                for method, predictor in self.predictors.items():\n                    if predictor.artifact_sha256 != bundle["methods"][method]["artifact_sha256"] or predictor.metadata["protocol_sha256"] != bundle["protocol_sha256"]:\n                        raise ValueError("Serving bundle does not match the loaded models.")')
    text = replace(text, '"default_method": DEFAULT_METHOD', '"default_method": self.default_method')
    text = replace(text, '"methods": [{"id": method, "device": "cpu" if method == "classical" else self.device}',
        '"methods": [{**REGISTRY[method].to_dict(), "device": self.device,\n                "seed": self.predictors[method].metadata.get("seed"),\n                "selected_epoch": self.predictors[method].metadata.get("epoch"),\n                "run_id": self.predictors[method].metadata.get("run_id"),\n                "parameters": self.predictors[method].metadata.get("parameters"),\n                "feature_dimension": self.predictors[method].metadata.get("pretrained", {}).get("configuration", {}).get("num_features"),\n                "artifact_sha256": getattr(self.predictors[method], "artifact_sha256", None)}')
    text = replace(text, 'method=DEFAULT_METHOD', 'method=None')
    text = replace(text, 'self._ensure_ready()\n        if method not in',
        'self._ensure_ready()\n        method = self.default_method if method is None else method\n        if method not in')
    text = replace(text, '"Choose classical, visagenet, convnext_ordinal, or all."',
        '"Choose convnextv2_nano, efficientnetv2_s, swinv2_tiny, or all."')
    put('src/visage/service.py', text)

    text = source('src/visage/api.py')
    text = replace(text, 'from .deployment import default_artifacts_dir', 'from .model_registry import ARTIFACTS, EVALUATION')
    text = replace(text, 'field(default_factory=default_artifacts_dir)', 'field(default_factory=lambda: ARTIFACTS)')
    text = replace(text, 'ROOT / "reports/extended_evaluation"', 'EVALUATION')
    text = replace(text, 'version="0.3.0"', 'version="0.4.0"')
    text = replace(text, 'method: Literal["all", "classical", "visagenet", "convnext_ordinal"] = DEFAULT_METHOD',
        'method: Literal["all", "convnextv2_nano", "efficientnetv2_s", "swinv2_tiny"] | None = None')
    text = replace(text, 'return await execute(request, "predict", method=method, face_index=face_index)',
        'selected = method if method is not None else service_or_error().describe()["default_method"]\n        return await execute(request, "predict", method=selected, face_index=face_index)')
    text = replace(text, 'from .service import DEFAULT_METHOD, InferenceService, ROOT', 'from .service import InferenceService, ROOT')
    put('src/visage/api.py', text)

    text = source('scripts/predict.py')
    text = replace(text, 'choices=("all", "classical", "visagenet", "convnext_ordinal")', 'choices=("all", *METHODS)')
    text = replace(text, 'configure_caches(ROOT)', 'configure_caches(ROOT)\nfrom visage.model_registry import METHODS, ARTIFACTS, serving_default')
    text = replace(text, 'from visage.inference import AgeGroupPredictor, read_face_crop\n    from visage.deployment import default_artifacts_dir',
        'from visage.inference import read_face_crop\n    from visage.pretrained_inference import PretrainedPredictor')
    text = replace(text, 'args.artifacts_dir or default_artifacts_dir()', 'args.artifacts_dir or ARTIFACTS')
    text = replace(text, '("classical", "visagenet", "convnext_ordinal") if', 'METHODS if')
    text = replace(text, 'choices=("all", *METHODS), default="all"', 'choices=("all", *METHODS), default=None')
    text = replace(text, 'else (args.method,)', 'else (args.method or serving_default(args.artifacts_dir),)')
    text = replace(text, 'AgeGroupPredictor(method,', 'PretrainedPredictor(method,')
    put('scripts/predict.py', text)
    text = source('scripts/predict_image.py')
    text = replace(text, 'sys.path.insert(0, str(ROOT / "src"))',
        'sys.path.insert(0, str(ROOT / "src"))\nfrom visage.model_registry import METHODS')
    text = replace(text, 'choices=("all", "classical", "visagenet", "convnext_ordinal"), default="visagenet"',
        'choices=("all", *METHODS), default=None')
    put('scripts/predict_image.py', text)
    for relative in ('scripts/serve_api.py', 'scripts/run_local_pilot.py'):
        text = source(relative)
        text = replace(text, 'from visage.deployment import default_artifacts_dir', 'from visage.model_registry import ARTIFACTS, METHODS')
        text = replace(text, 'args.artifacts_dir or default_artifacts_dir()', 'args.artifacts_dir or ARTIFACTS')
        if relative.endswith('run_local_pilot.py'):
            text = replace(text, 'for name, filename in (\n            ("classical", "svm.npz"), ("visagenet", "best.pt"), ("convnext_ordinal", "best.pt"))',
                'for name, filename in ((method, "best.pt") for method in METHODS)')
        put(relative, text)
    text = source('src/visage/pilot.py')
    text = replace(text, 'methods = ("classical", "visagenet", "convnext_ordinal")',
        'from .model_registry import METHODS\n    methods = METHODS')
    put('src/visage/pilot.py', text)

    tones = ['convnext-tone', 'efficient-tone', 'swin-tone']
    icons = ['layers', 'grid', 'spark']
    model_info = {method: {'name': REGISTRY[method].name, 'kind': REGISTRY[method].family,
        'tone': tones[i], 'icon': icons[i]} for i, method in enumerate(METHODS)}
    put('src/visage/web/models.js', '// Generated from the Python model registry.\nexport const METHODS = '
        + json.dumps(METHODS) + ';\nexport const MODEL_INFO = ' + json.dumps(model_info, ensure_ascii=False) + ';\n')
    text = source('src/visage/web/core.js')
    start, end = text.index('export const METHODS'), text.index('export class AppError')
    text = text[:start] + 'import { METHODS, MODEL_INFO } from "./models.js";\nexport { METHODS, MODEL_INFO };\nexport const GROUPS = ["Under 18", "18–59", "60+"];\n' + text[end:]
    text = replace(text, 'method = "visagenet"', 'method = METHODS[0]')
    put('src/visage/web/core.js', text)
    text = source('src/visage/web/evaluation.js')
    text = replace(text, '"visagenet"', 'METHODS[0]')
    text = replace(text, 'Selected from validation evidence', 'Chosen across two training seeds')
    text = replace(text, 'Accuracy, macro F1 and mean one-vs-rest ROC AUC.',
        'Accuracy, macro F1 and mean one-vs-rest ROC AUC for the current checkpoints.')
    text = replace(text, 'Macro F1 weights the three age groups equally. AUC measures score ranking; it is not a confidence percentage.',
        'Each row shows the app’s seed 42 checkpoint. The recommended family is chosen across seeds 42 and 43. Macro F1 weights groups equally; AUC measures ranking, not certainty for a person.')
    text = replace(text, '["Model", "Accuracy", "Macro F1", "Mean AUC"]',
        '["Model", "Accuracy", "Macro F1", "Weighted F1", "Mean AUC"]')
    text = replace(text, 'formatNumber(value.macro_f1, 4), formatNumber(value.macro_auc, 4)',
        'formatNumber(value.macro_f1, 4), formatNumber(value.weighted_f1, 4), formatNumber(value.macro_auc, 4)')
    text = replace(text, 'charts.append(rocCard, matrixCard); root.replaceChildren(charts);',
        'charts.append(rocCard, matrixCard); root.replaceChildren(charts);\n    if (value.per_class) { const perClass = panel("Per age group metrics", "Precision, recall and F1 for the selected serving checkpoint."); perClass.classList.add("table-panel"); perClass.append(table(["Age group", "Support", "Precision", "Recall", "F1"], value.per_class.map(c => [names[c.label], formatNumber(c.support), formatNumber(c.precision, 4), formatNumber(c.recall, 4), formatNumber(c.f1, 4)]), "Per age group classification metrics")); root.append(perClass); }')
    put('src/visage/web/evaluation.js', text)
    text = source('src/visage/web/app.js')
    text = replace(text, 'defaultMethod = "visagenet"', 'defaultMethod = METHODS[0]')
    text = replace(text, 'models.default_method : "visagenet"', 'models.default_method : METHODS[0]')
    text = replace(text, 'let evaluationLoaded = false', 'let methodChosen = false;\nlet evaluationLoaded = false')
    text = replace(text, '$("service-status").dataset.state = "ready";',
        'if (!methodChosen) session.setMethod(defaultMethod);\n    METHODS.forEach(method => { const option = [...$("method-select").options].find(value => value.value === method); if (option) option.textContent = MODEL_INFO[method].name + (method === defaultMethod ? " · Recommended" : ""); });\n    $("service-status").dataset.state = "ready";')
    text = replace(text, 'event => { session.setMethod(event.target.value);', 'event => { methodChosen = true; session.setMethod(event.target.value);')
    text = replace(text, '() => { session.setMethod("all");', '() => { methodChosen = true; session.setMethod("all");')
    text = replace(text, 'lastResult = state.result; renderResults(state);', 'lastResult = state.result; renderResults(state); renderOverlay();')
    text = replace(text, 'const labelY = Math.max(0, y - labelHeight), labelX = Math.max(0, Math.min(x, width - textSize * 4.8));',
        'const result = selected && state.result ? (state.result.predictions.find(value => value.method === defaultMethod) || state.result.predictions[0]) : null;\n    const differs = selected && state.result && new Set(state.result.predictions.map(value => value.class_id)).size > 1;\n    const label = result ? `Face ${face.index + 1} · ${GROUPS[result.class_id]}${differs ? " · differs" : ""}` : `Face ${face.index + 1}`;\n    const labelWidth = Math.min(width, textSize * (label.length * .62 + 1.2));\n    const labelY = Math.max(0, y - labelHeight), labelX = Math.max(0, Math.min(x, width - labelWidth));')
    text = replace(text, 'width: textSize * 4.8, height: labelHeight', 'width: labelWidth, height: labelHeight')
    text = replace(text, 'class: "box-label" }, `Face ${face.index + 1}`)', 'class: "box-label" }, label)')
    put('src/visage/web/app.js', text)
    text = source('src/visage/web/index.html')
    start = text.index('<select id="method-select">')
    end = text.index('</select>', start) + len('</select>')
    options = ''.join(f'<option value="{m}">{REGISTRY[m].name}</option>' for m in METHODS)
    text = text[:start] + '<select id="method-select">' + options + '<option value="all">Compare all three</option></select>' + text[end:]
    text = text.replace('Classical, custom CNN and transfer learning', 'Two CNNs and a hierarchical Transformer')
    start = text.index('<div class="empty-models">')
    end = text.index('</div></div>', start) + len('</div></div>')
    # Match the whole nested placeholder with the following results-loading anchor.
    end = text.index('\n              <div id="results-loading"', start)
    placeholders = ''.join(f'<div><span class="model-mini {tones[i]}"><svg class="icon"><use href="#i-{icons[i]}"/></svg></span><span>{REGISTRY[m].name}</span><span class="placeholder-dash">—</span></div>' for i, m in enumerate(METHODS))
    text = text[:start] + '<div class="empty-models">' + placeholders + '</div></div>' + text[end:]
    start = text.index('<div class="about-models">')
    end = text.index('\n        <div class="about-bottom">', start)
    descriptions = [
        ('MODERN CONVOLUTION', 'A published ConvNeXt V2 Nano backbone with FCMAE pretraining and ImageNet adaptation. A replacement classifier estimates three age groups.'),
        ('EFFICIENT CONVOLUTION', 'Published EfficientNetV2-S weights adapted with a three-class head, evaluated using the same data partitions and training budget.'),
        ('HIERARCHICAL TRANSFORMER', 'Published Swin Transformer V2 Tiny weights using shifted-window attention, adapted for the same three age groups.')]
    cards = ''.join(f'<article class="panel"><span class="model-mini {tones[i]}"><svg class="icon"><use href="#i-{icons[i]}"/></svg></span><span class="eyebrow">{descriptions[i][0]}</span><h2>{REGISTRY[m].name}</h2><p>{descriptions[i][1]}</p><span class="model-kind">Pretrained backbone · Three-class softmax head</span></article>' for i, m in enumerate(METHODS))
    text = text[:start] + '<div class="about-models">' + cards + '</div>' + text[end:]
    put('src/visage/web/index.html', text)
    text = source('src/visage/web/styles.css')
    text += '\n.efficient-tone{background:#e5f3ec;color:#287657}.swin-tone{background:#eee9fa;color:#7353ad}\n'
    put('src/visage/web/styles.css', text)

    text = source('scripts/benchmark_backend.py')
    text = replace(text, 'METHODS = ("classical", "visagenet", "convnext_ordinal")',
        'from visage.model_registry import METHODS, ARTIFACTS')
    text = replace(text, 'for method, filename in (\n        ("classical", "svm.npz"), ("visagenet", "best.pt"), ("convnext_ordinal", "best.pt"))',
        'for method, filename in ((method, "best.pt") for method in METHODS)')
    text = replace(text, 'default=ROOT / "artifacts"', 'default=ARTIFACTS')
    text = replace(text, 'reports/backend_benchmark.json', 'reports/pretrained/benchmark/backend_benchmark.json')
    put('scripts/benchmark_backend.py', text)
    text = source('scripts/verify_web_app.py')
    text = replace(text, 'METHODS = ["classical", "visagenet", "convnext_ordinal"]',
        'from visage.model_registry import METHODS as ACTIVE_METHODS\nMETHODS = list(ACTIVE_METHODS)')
    text = replace(text, '"/v1/predict?method=visagenet&face_index=0"', 'f"/v1/predict?method={METHODS[0]}&face_index=0"')
    text = replace(text, 'result["predictions"][1]["class_id"]', 'result["predictions"][0]["class_id"]')
    text = replace(text, 'reports/web_app/', 'reports/pretrained/web_app/')
    put('scripts/verify_web_app.py', text)
    text = source('scripts/package_web_app.py')
    text = replace(text, 'def main():\n    def read(relative):',
        'def main():\n    from visage.model_registry import ARTIFACTS\n    if (ARTIFACTS / "bundle.json").is_file():\n        from package_pretrained_app import main as package_pretrained\n        return package_pretrained()\n\n    def read(relative):')
    put('scripts/package_web_app.py', text)
    text = source('src/visage/evaluation_registry.py')
    text = replace(text, 'currently served original artifacts', 'currently served model artifacts')
    text = replace(text, '"macro_f1": details["macro_f1"] if details else None,',
        '"macro_f1": details["macro_f1"] if details else None,\n                "weighted_f1": sum(v["support"] * v["f1"] for v in details["per_class"]) / details["samples"] if details else None,')
    text = replace(text, '"samples": record["samples"], "class_names": list(CLASS_NAMES), "dataset": "UTKFace",',
        '"samples": record["samples"], "class_names": list(CLASS_NAMES), "dataset": "UTKFace",\n            "model_scope": record.get("scope", "Metrics for the currently served checkpoints"),')
    text = replace(text,
        'if sha256_file(predictor.directory / filename) != value["artifact_sha256"]:',
        'disk_sha256 = sha256_file(predictor.directory / filename)\n            if getattr(predictor, "artifact_sha256", disk_sha256) != disk_sha256 or disk_sha256 != value["artifact_sha256"]:')
    text = replace(text,
        'if sha256_file(predictor.directory / filename) != record["artifact_sha256"][method]:',
        'disk_sha256 = sha256_file(predictor.directory / filename)\n            if getattr(predictor, "artifact_sha256", disk_sha256) != disk_sha256 or disk_sha256 != record["artifact_sha256"][method]:')
    put('src/visage/evaluation_registry.py', text)
    text = source('tests/test_evaluation_registry.py')
    text += '''\n\ndef test_replaced_disk_artifacts_cannot_relabel_metrics_for_a_loaded_model(tmp_path):
    method = "convnextv2_nano"
    directory = tmp_path / method
    directory.mkdir()
    path = directory / "best.pt"
    path.write_bytes(b"original loaded checkpoint")
    loaded_hash = sha256_file(path)
    predictor = SimpleNamespace(directory=directory, artifact_sha256=loaded_hash)
    service = SimpleNamespace(manifest_sha256="a" * 64, predictors={method: predictor})
    curve = {"auc": .5, "fpr": [0, 1], "tpr": [0, 1]}
    path.write_bytes(b"a new checkpoint on disk that was never loaded")
    record = {"ok": True, "manifest_sha256": service.manifest_sha256, "task": "age_group_classification",
        "split": "validation", "samples": 3, "methods": {method: {"artifact_sha256": sha256_file(path),
        "accuracy_percent": 50, "score_type": "softmax", "roc": {"macro_auc_mean": .5,
        "per_class": {str(i): curve for i in range(3)}}}}}
    (tmp_path / "classification_roc.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="artifact"):
        evaluation_snapshot(tmp_path, service)
'''
    put('tests/test_evaluation_registry.py', text)
    text = source('scripts/run_browser_qa.py')
    text = replace(text, 'reports/web_app', 'reports/pretrained/web_app')
    text = replace(text, 'if len(suites[-1]["passed"]) != 13:', 'if len(suites[-1]["passed"]) < 25:')
    text = replace(text, 'deadline = time.monotonic() + 60', 'deadline = time.monotonic() + 120')
    text = replace(text, 'did not finish in 60 seconds', 'did not finish in 120 seconds')
    put('scripts/run_browser_qa.py', text)
    text = source('pyproject.toml')
    text = replace(text, 'version = "0.1.0"', 'version = "0.4.0"')
    text = replace(text, 'Reproducible three-method age-group classification research',
        'VisAge three-pretrained-model age-group research and local web application')
    text = replace(text, '"timm", "matplotlib"', '"timm", "safetensors>=0.4", "matplotlib"')
    text = replace(text, 'dependencies = ["numpy", "scipy", "pillow"',
        'dependencies = ["numpy>=2.0", "scipy", "pillow==13.0.0"')
    text = replace(text, '"timm", "safetensors>=0.4"', '"timm==1.0.31", "safetensors>=0.4"')
    put('pyproject.toml', text)
    text = source('src/visage/__init__.py')
    text = replace(text, '__version__ = "0.1.0"', '__version__ = "0.4.0"')
    put('src/visage/__init__.py', text)

    # Tests follow the active IDs. Historical isolated training/evaluation tests
    # retain explicit legacy IDs and continue to verify their archived methods.
    replacements = {'classical': METHODS[1], 'visagenet': METHODS[0], 'convnext_ordinal': METHODS[2]}
    for relative in ('tests/test_service.py', 'tests/test_api.py', 'tests/test_web_ui.py',
        'tests/test_integration.py', 'tests/test_pilot.py', 'tests/web/core.test.mjs', 'tests/web/browser-upload.js'):
        text = source(relative)
        for old, new in replacements.items():
            text = text.replace(old, new)
        if relative.endswith('test_integration.py'):
            text = text.replace('artifacts/efficientnetv2_s/svm.npz', 'artifacts/pretrained/serving/efficientnetv2_s/best.pt')
            text = text.replace('artifacts/convnextv2_nano/best.pt', 'artifacts/pretrained/serving/convnextv2_nano/best.pt')
            text = text.replace('artifacts/swinv2_tiny/best.pt', 'artifacts/pretrained/serving/swinv2_tiny/best.pt')
            text += '''\n\ndef test_real_default_and_loaded_identity_agree_across_api_and_both_clis(real_application, tmp_path):
    client, service, path, data = real_application
    metadata = client.get("/v1/models").json()
    default = metadata["default_method"]
    assert default in METHODS and len(metadata["methods"]) == 3
    for value in metadata["methods"]:
        assert value["seed"] == 42 and value["selected_epoch"] >= 1
        assert len(value["artifact_sha256"]) == 64
    response = client.post("/v1/predict", content=data, headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 200, response.text
    expected = response.json()["predictions"]
    assert len(expected) == 1 and expected[0]["method"] == default
    image = decode_image(data)
    try:
        prepared = service.detector.prepare(image)
        try:
            crop = tmp_path / "selected.png"
            prepared.crop.save(crop)
        finally:
            prepared.crop.close()
    finally:
        image.close()
    for command, source in (("predict_image.py", path), ("predict.py", crop)):
        output = tmp_path / (command + ".json")
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / command), str(source), "--output", str(output)],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        values = json.loads(output.read_text(encoding="utf-8"))["predictions"]
        assert len(values) == 1 and values[0]["method"] == default
        assert values[0]["class_id"] == expected[0]["class_id"]
'''
        if relative.endswith('core.test.mjs'):
            text = text.replace('["efficientnetv2_s", "convnextv2_nano", "swinv2_tiny"]', '["convnextv2_nano", "efficientnetv2_s", "swinv2_tiny"]')
        if relative.endswith('test_api.py'):
            text += '''\n\n@pytest.mark.parametrize("legacy_method", ["classical", "visagenet", "convnext_ordinal"])
def test_legacy_method_ids_are_rejected_without_running_a_different_model(legacy_method):
    service = StubService()
    with TestClient(create_app(service_factory=lambda: service)) as client:
        response = client.post(f"/v1/predict?method={legacy_method}", content=b"valid", headers={"Content-Type": "image/png"})
        assert response.status_code == 422
        assert not service.calls
'''
        if relative.endswith('browser-upload.js'):
            text = 'import { METHODS, MODEL_INFO, GROUPS } from "/assets/core.js";\n' + text
            text = replace(text, 'byId("tab-upload").click();',
                'const metadata = await (await fetch("/v1/models")).json();\n    check(byId("method-select").value === metadata.default_method, "The menu starts on the validation-selected default");\n    check([...byId("method-select").options].map(v => v.value).filter(v => v !== "all").join() === METHODS.join(), "The active menu has exactly three pretrained IDs");\n    byId("tab-upload").click();')
            text = replace(text, 'check(byId("model-results").children.length === 3, "Compare all renders three real model results");',
                'check(byId("model-results").children.length === 3, "Compare all renders three real model results");\n    check(METHODS.every(m => byId("model-results").textContent.includes(MODEL_INFO[m].name)), "Result cards identify all three new architectures");\n    check(GROUPS.some(g => byId("face-overlay").textContent.includes(g)), "The selected face overlay shows its estimated age group");')
            text = replace(text, 'check(byId("prediction-results").hidden && !byId("analyze-photo").disabled, "Changing the method clears results and keeps the selected face");',
                'check(byId("prediction-results").hidden && !byId("analyze-photo").disabled, "Changing the method clears results and keeps the selected face");\n    check(!GROUPS.some(g => byId("face-overlay").textContent.includes(g)), "Changing the method also clears the stale overlay age group");')
            anchor = 'byId("remove-photo").click();'
            additions = '''byId("tab-camera").click(); byId("start-camera").click();
    await until(() => !byId("video-stage").hidden, "Simulated camera did not start");
    check(!byId("switch-camera").disabled, "Live camera exposes both simulated devices");
    byId("switch-camera").click();
    await until(() => !byId("video-stage").hidden && !byId("camera-preview").classList.contains("mirrored"), "Camera did not switch");
    check(!byId("camera-preview").classList.contains("mirrored"), "Camera switching selects the simulated rear device");
    byId("capture-photo").click();
    await until(() => !byId("analyze-photo").disabled && !byId("photo-stage").hidden, "Camera capture did not enter the shared face pipeline");
    check(byId("video-stage").hidden && byId("camera-preview").srcObject === null, "Capturing a photo shuts down the video stream");
    byId("method-select").value = "all"; byId("method-select").dispatchEvent(new Event("change")); byId("analyze-photo").click();
    await until(() => !byId("prediction-results").hidden, "Camera prediction did not complete");
    check(byId("model-results").children.length === 3, "Camera pixels reach all three trained classifiers");
    byId("replace-photo").click();
    await until(() => !byId("video-stage").hidden, "Retake did not restart the preview");
    check(byId("prediction-results").hidden && byId("photo-stage").hidden, "Retake clears the prior photo and all results");
    location.hash = "about";
    await until(() => !byId("view-about").hidden && byId("camera-preview").srcObject === null, "Leaving camera mode did not stop tracks");
    check(METHODS.every(m => byId("view-about").textContent.includes(MODEL_INFO[m].name)), "About explains the three pretrained architectures");
    check(!/VisAgeNet|LBP|SVM|random weights/.test(byId("view-about").textContent), "The active About view has no legacy model descriptions");
    location.hash = "evaluation";
    await until(() => document.querySelectorAll("#eval-classification .chart-select button").length === 3, "New model dashboard did not load", 30000);
    check(METHODS.every(m => byId("eval-classification").textContent.includes(MODEL_INFO[m].name)), "Dashboard comparison lists all three current models");
    for (const method of METHODS) {
      document.querySelector(`#eval-classification button[data-method="${method}"]`).click();
      check(byId("classification-model-details").textContent.includes(MODEL_INFO[method].name) && byId("classification-model-details").querySelector("svg"), `ROC and confusion details render for ${method}`);
    }
    check(document.documentElement.scrollWidth <= innerWidth + 1, "The dashboard fits the browser viewport");
    const events = await (await fetch("/__qa__/events")).json();
    check(events.filter(e => e.event === "camera-tracks").at(-1)?.active === 0, "Every simulated camera track was stopped");
    location.hash = "analyze";
    await until(() => !byId("view-analyze").hidden, "Could not return to analysis");
    byId("remove-photo").click();'''
            text = replace(text, anchor, additions)
        put(relative, text)
    manifest = {'status': 'staged', 'originals': originals, 'staged': changed, 'updated_utc': utc_now()}
    write_json(ROOT / 'reports/pretrained/app_stage.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    manifest = prepare()
    if args.apply:
        bundle = json.loads((ARTIFACTS / 'bundle.json').read_text())
        if set(bundle['methods']) != set(METHODS):
            raise ValueError('Complete serving bundle is required before activating the app.')
        for method in METHODS:
            if sha256_file(ARTIFACTS / method / 'best.pt') != bundle['methods'][method]['artifact_sha256']:
                raise ValueError('Serving bundle hash mismatch.')
        for relative, digest in manifest['staged'].items():
            target = ROOT / relative
            # Never discard a concurrent edit or an unrecognized active version.
            if target.exists() and sha256_file(target) not in {manifest['originals'].get(relative), digest}:
                raise ValueError(f'App file changed after staging: {relative}')
        for relative in manifest['staged']:
            (ROOT / relative).write_bytes((STAGE / relative).read_bytes())
        manifest['status'] = 'activated'
        manifest['activated_utc'] = utc_now()
        write_json(ROOT / 'reports/pretrained/app_activation.json', manifest)
    print(json.dumps({'status': manifest['status'], 'files': len(manifest['staged'])}))


if __name__ == '__main__':
    main()

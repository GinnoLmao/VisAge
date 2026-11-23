"""Export actual six-run comparison, figures and a self-contained HTML report."""
import csv
import html
import json
import base64
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import configure_caches, write_json, utc_now, sha256_file
configure_caches(ROOT)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visage.model_registry import METHODS, REGISTRY, ARTIFACTS

REPORTS = ROOT / 'reports/pretrained'


def main():
    def read(name):
        return json.loads((REPORTS / name).read_text(encoding='utf-8'))
    verified, selection = read('verified_runs.json'), read('selection.json')
    if not verified['ok'] or verified['count'] != 6 or selection['seeds'] != [42, 43]:
        raise ValueError('The report requires the six approved and verified runs.')
    records = verified['runs']
    rows = []
    per_class = []
    for r in records:
        m = r['verified_metrics']
        rows.append({'method': r['method'], 'seed': r['seed'], 'selected_epoch': r['best']['epoch'],
            'selected_stage': r['selected_stage'], 'epochs_run': r['epochs'], 'accuracy_percent': m['accuracy'] * 100,
            'macro_f1': m['macro_f1'], 'weighted_f1': m['weighted_f1'], 'macro_auc': r['roc']['macro_auc_mean'],
            'under18_predicted_older_percent': m['under18_predicted_older_rate'] * 100,
            'senior_recall': m['per_class'][2]['recall'], 'artifact_sha256': r['artifact_sha256']})
        for c in m['per_class']:
            per_class.append({'method': r['method'], 'seed': r['seed'], **c,
                'auc': r['roc']['per_class'][str(c['label'])]['auc']})
    for name, data in (('two_seed_comparison.csv', rows), ('two_seed_per_class.csv', per_class)):
        with (REPORTS / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    figures = REPORTS / 'figures_two_seed'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .2})
    plots = []
    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(figures / f'{name}.png', dpi=160, bbox_inches='tight', facecolor='white')
        fig.savefig(figures / f'{name}.pdf', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plots.append((name, caption))
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharey=True)
    for i, method in enumerate(METHODS):
        for j, seed in enumerate((42, 43)):
            r = next(v for v in records if v['method'] == method and v['seed'] == seed)
            history = json.loads((ROOT / f'artifacts/pretrained/runs/{r["run_id"]}/history.json').read_text())
            ax = axes[i, j]
            x = [e['epoch'] for e in history]
            ax.plot(x, [e['train']['macro_f1'] for e in history], color='#9aa6af', label='Online augmented train')
            ax.plot(x, [e['validation']['macro_f1'] for e in history], color='#267983', label='FP32 validation')
            ax.axvline(2.5, color='#888', ls=':', lw=1)
            ax.scatter([r['best']['epoch']], [r['verified_metrics']['macro_f1']], color='#b77a28', marker='*', s=85, zorder=4)
            ax.set(title=f'{REGISTRY[method].name} · seed {seed}', xlabel='Epoch', ylabel='Macro F1', ylim=(0, 1))
            ax.legend(loc='lower right', fontsize=8)
    save(fig, 'training_curves', 'Six completed runs. Stars show selected epochs; the dotted line marks unfreezing after two head-only epochs.')
    roc_record = read('evaluation/classification_roc.json')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, method in zip(axes, METHODS, strict=True):
        for label, name in enumerate(('Under 18', '18–59', '60+')):
            curve = roc_record['methods'][method]['roc']['per_class'][str(label)]
            ax.plot(curve['fpr'], curve['tpr'], label=f'{name}: {curve["auc"]:.4f}')
        ax.plot([0, 1], [0, 1], '--', color='#aaa')
        ax.set(title=REGISTRY[method].name, xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8, loc='lower right')
    save(fig, 'serving_validation_roc', 'Seed 42 serving checkpoints on all 3,492 aligned validation images. AUC is verified independently by average ranks.')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, method in zip(axes, METHODS, strict=True):
        m = next(r['verified_metrics'] for r in records if r['method'] == method and r['seed'] == 42)
        matrix = np.array(m['confusion_matrix']); norm = np.array(m['row_normalized_confusion_matrix'])
        ax.grid(False); ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{matrix[i,j]}\n{norm[i,j] * 100:.1f}%', ha='center', va='center',
                    color='white' if norm[i,j] > .5 else '#102a3c', fontsize=9)
        ax.set(title=REGISTRY[method].name, xticks=range(3), yticks=range(3),
            xticklabels=['Under18', '18–59', '60+'], yticklabels=['Under18', '18–59', '60+'], xlabel='Predicted group', ylabel='Provided group')
    save(fig, 'serving_validation_confusion', 'Seed 42 validation confusion counts and row percentages. Rows are provided age groups; columns are predictions.')
    robust, test, pipeline = read('robustness.json'), read('test_summary.json'), read('face_pipeline_validation.json')
    test_per_class = [{'method': method, **c, 'auc': value['roc']['per_class'][str(c['label'])]['auc']}
        for method, value in test['methods'].items() for c in value['metrics']['per_class']]
    means = [{'method': method, **value} for method, value in selection['validation_means'].items()]
    for name, data in (('two_seed_test_per_class.csv', test_per_class), ('two_seed_validation_means.csv', means)):
        with (REPORTS / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    conditions = list(robust['methods'][METHODS[0]]['conditions'])
    values = np.array([[robust['methods'][m]['conditions'][c]['metrics']['macro_f1'] for c in conditions] for m in METHODS])
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.grid(False)
    ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    for i in range(3):
        for j in range(len(conditions)):
            ax.text(j, i, f'{values[i,j]:.4f}', ha='center', va='center', color='white' if values[i,j] > .65 else '#102a3c')
    ax.set(xticks=range(len(conditions)), xticklabels=[c.replace('_', '\n') for c in conditions],
        yticks=range(3), yticklabels=[REGISTRY[m].name for m in METHODS], title='Aligned validation robustness macro F1')
    save(fig, 'validation_robustness', 'Seven fixed validation conditions per serving family. These do not establish real-camera accuracy.')
    e = lambda value: html.escape(str(value))
    def table(headers, values):
        return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{e(v)}</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{e(v)}</td>' for v in row) + '</tr>' for row in values) + '</tbody></table></div>'
    parts = [f'<h1>VisAge two-seed evaluation and activation</h1><p class="lead">Six completed runs use seeds 42 and 43. The default selected by the unchanged validation tie rules is <strong>{e(REGISTRY[selection["default_method"]].name)}</strong>. No additional training was performed.</p>',
        '<h2>Scope and protocol</h2><p>The original training protocol and artifacts are preserved. The approved amendment excludes all seed44 runs, including the unfinished Nano run, and defers nine frozen-backbone/data-efficiency jobs. Two seeds provide limited replication. The amendment was made after validation results were observed, at the user’s request, before evaluating these models on the existing test benchmark.</p>',
        '<h2>Active model architecture and preprocessing</h2>',
        table(['Model', 'Published checkpoint', 'Adapted parameters', 'Selected seed 42 stage'], [[REGISTRY[m].name, REGISTRY[m].checkpoint,
            f'{next(r["best"]["parameters"] for r in records if r["method"] == m):,}',
            next(r['selected_stage'] for r in records if r['method'] == m and r['seed'] == 42)] for m in METHODS]),
        '<p>All three are pretrained backbones with a replacement three-logit linear head, trained with the same train-only class-weighted cross-entropy. Inputs use EXIF orientation, RGB conversion, whole-face Pillow bicubic resizing to 224 then 256 pixels, with no additional center crop. Raw photos first pass through the unchanged YuNet five-landmark alignment. ConvNeXt and Swin use ImageNet mean/std; this EfficientNet checkpoint uses mean/std 0.5. Validation and inference use FP32. Scores used for ROC are not calibrated certainty about a person.</p>',
        '<h2>Completed validation runs</h2>', table(['Model', 'Seed', 'Best epoch', 'Stage', 'Accuracy %', 'Macro F1', 'Weighted F1', 'Macro AUC'],
            [[REGISTRY[r['method']].name, r['seed'], r['selected_epoch'], r['selected_stage'], f'{r["accuracy_percent"]:.2f}',
                f'{r["macro_f1"]:.4f}', f'{r["weighted_f1"]:.4f}', f'{r["macro_auc"]:.4f}'] for r in rows]),
        '<h2>Default selection</h2><p>Use the mean validation macro F1 across the two seeds. Families within 0.005 of the highest mean are near-tied; choose lower mean under-18-to-older error, then higher senior recall, lower measured CPU median, and registry order for an exact remaining tie. Test scores do not select the default. Every family serves its seed 42 checkpoint.</p>',
        table(['Model', 'Mean macro F1', 'Sample std', 'Under18 older error %', 'Senior recall %'], [[REGISTRY[m].name,
            f'{v["macro_f1_mean"]:.4f}', f'{v["macro_f1_std_sample"]:.4f}', f'{v["under18_older_mean"] * 100:.2f}', f'{v["senior_recall_mean"] * 100:.2f}']
            for m, v in selection['validation_means'].items()]),
        '<p>Nano’s selected seed42 checkpoint came from head-only warm-up. Its subsequent fine-tuning did not surpass that checkpoint. It remains a pretrained-backbone classifier; the report does not relabel this selected artifact as fine-tuned.</p>',
        '<p>EfficientNet and Swin fall within the fixed near-tie threshold. Swin wins the first tie-break through its lower mean under-18-to-older error; latency is not needed to break this tie. The standard deviation uses the sample convention (ddof=1).</p>',
        '<h2>Existing test benchmark</h2><p>The 3,493-image partition was observed earlier in the project. Each frozen serving family received one final resumable evaluation after the default decision was frozen. This is an existing benchmark, not independent confirmation; no test-guided tuning was performed. Accuracy here measures age-group classification, not identity recognition.</p>',
        table(['Model', 'Accuracy %', 'Macro F1', 'Weighted F1', 'Macro AUC'], [[REGISTRY[m].name, f'{v["metrics"]["accuracy"] * 100:.2f}',
            f'{v["metrics"]["macro_f1"]:.4f}', f'{v["metrics"]["weighted_f1"]:.4f}', f'{v["roc"]["macro_auc_mean"]:.4f}'] for m, v in test['methods'].items()]),
        '<h2>Shared face pipeline</h2>', f'<p>Attempted {pipeline["attempted"]}; accepted {pipeline["accepted"]}; rejected {pipeline["rejected"]}; coverage {pipeline["coverage"] * 100:.2f}%. Every rejected input remains in the all-attempted denominator. These existing aligned UTKFace crops were redetected/realigned and do not establish ordinary-photo accuracy. Rejection codes: {e(pipeline["rejection_codes"])}.</p>',
        table(['Model', 'Accepted accuracy %', 'Accepted macro F1', 'All-attempted correct %'], [[REGISTRY[m].name,
            f'{v["accepted_image_metrics"]["accuracy"] * 100:.2f}', f'{v["accepted_image_metrics"]["macro_f1"]:.4f}', f'{v["end_to_end_correct_rate"] * 100:.2f}'] for m, v in pipeline['methods'].items()])]
    ap = read('evaluation/detection_ap.json')
    parts += ['<h2>Detector AP provenance</h2>', f'<p>The unchanged YuNet detector retains its checksum-bound WIDER validation evidence: AP50 {ap["metrics"]["AP50"]["ap_percent"]:.2f}%, AP75 {ap["metrics"]["AP75"]["ap_percent"]:.2f}%. Detector bytes, revision, annotations and saved predictions were verified, with twelve exact reload checks. This reused custom all-valid-box AP protocol is not the official WIDER difficulty leaderboard. The sweep uses confidence 0.05; the app uses 0.9 before selection and quality checks.</p>',
        '<h2>Artifact and checkpoint audit</h2><p>All six saved checkpoints reloaded strictly and reproduced every one of the 3,492 validation predictions, with maximum absolute score difference 0.0. Histories, earliest best epochs, training-stage transitions, class weights, preprocessing, prediction order and metric calculations were verified. The serving checksums below bind the API and dashboard to the selected artifacts.</p>',
        table(['Serving model', 'Seed', 'Epoch', 'Checkpoint SHA-256'], [[REGISTRY[r['method']].name, r['seed'], r['selected_epoch'], r['artifact_sha256']] for r in rows if r['seed'] == 42])]
    for name, caption in plots:
        encoded = base64.b64encode((figures / f'{name}.png').read_bytes()).decode()
        parts.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{e(caption)}"><figcaption>{e(caption)}</figcaption></figure>')
    parts.append('<h2>Interpretation and limits</h2><p>Nano shows a substantial validation drop at the first unfrozen epoch in both seeds; its seed 42 best checkpoint remains the initial warm-up epoch. Several runs develop a large gap between online training and validation macro F1. The fixed best-validation rule preserves the selected checkpoint rather than serving a later, worse epoch. Swin and EfficientNet are close in two-seed mean macro F1, and two seeds do not establish statistical superiority. The robustness checks support comparison under the seven specified perturbations only. No additional tuning or training was performed to change these outcomes.</p>')
    acceptance = REPORTS / 'web_app/automated_acceptance.json'
    parts += ['<h2>Application activation and verification</h2>']
    if acceptance.exists():
        a = json.loads(acceptance.read_text())
        if not a['ok']:
            raise ValueError('Acceptance record must indicate success.')
        browser = next(v for v in read('web_app/browser_events_final.json') if v.get('event') == 'upload-integration')
        parts.append(f'<p>Automated acceptance passed: {a["counts"]["tests_full.xml"]} Python tests, {a["counts"]["tests_javascript.xml"]} JavaScript tests and {len(browser["passed"])} browser workflow assertions, with no failed or skipped regression cases. The active app exposes the three new model IDs, selected default, upload/camera/shared-face workflows, estimated-age overlays, Compare All, and new-artifact dashboard evidence. See the saved HTTP, browser, CPU/CUDA and regression records for actual test scope.</p>')
        parts.append(table(['Check', 'Recorded outcome'], [[check, 'Passed'] for check in a['checks']] + [['Visual inspection', a['visual_review']]]))
        devices = read('device_verification.json')
        parts.append('<p>CPU and CUDA matched labels on twelve fixed validation crops and six raw-photo pipeline fixtures. All three models were also loaded together on the GPU. These are functional consistency checks, not an independent accuracy study.</p>')
        parts.append(table(['Model', 'Maximum CPU/CUDA score difference', 'Warm CUDA median ms'], [[REGISTRY[m].name,
            f'{v["maximum_absolute_score_difference"]:.8f}', f'{v["cuda_batch_one_median_ms"]:.2f}'] for m, v in devices['methods'].items()]))
        benchmark = read('benchmark/backend_benchmark.json')
        timings = benchmark['http_round_trip']
        parts.append(f'<p>The local CPU backend completed {benchmark["iterations"]} repeated Compare All HTTP requests after {benchmark["warmups"]} warm-ups: median round-trip {timings["median_ms"]:.2f} ms; p95 {timings["p95_ms"]:.2f} ms. Startup took {benchmark["startup_seconds"]:.2f} seconds. This measures one fixed validation input in an isolated local process, including the shared image pipeline and all three models; it does not measure hosted concurrency or real-camera accuracy.</p>')
    else:
        parts.append('<p>Evaluation is complete. Application activation/acceptance is still pending; this report must be refreshed after those checks.</p>')
    parts.append('<h2>Evidence files and reproduction</h2><p>The adjacent comparison, validation-means, validation-per-class and test-per-class CSVs provide spreadsheet-ready metrics. Complete confusion matrices and ROC arrays remain in the JSON evidence. Research inputs and original training/evaluation protocols are preserved. The updated project README documents setup and inference commands; the historical delivery ZIP remains a separate rollback package.</p>')
    paths = ['configs/pretrained/two_seed_amendment.json', 'reports/pretrained/evaluation_protocol_two_seed.json',
        'reports/pretrained/verified_runs.json', 'reports/pretrained/reload_verification.json', 'reports/pretrained/selection.json',
        'reports/pretrained/test_summary.json', 'reports/pretrained/robustness.json', 'reports/pretrained/face_pipeline_validation.json',
        'reports/pretrained/evaluation/detection_lineage.json', 'artifacts/pretrained/serving/bundle.json']
    parts.append(table(['Evidence path', 'SHA-256'], [[path, sha256_file(ROOT / path)] for path in paths]))
    feature_path = REPORTS / 'score_feature/acceptance.json'
    if feature_path.exists():
        feature = read('score_feature/acceptance.json')
        if not feature['ok']:
            raise ValueError('Score-display acceptance must pass before publishing completion.')
        parts.append(f'<h2>All three class scores</h2><p>The application now displays Under 18, 18–59 and 60+ scores for each model, without a next-highest callout. The selected group uses the unchanged highest-score rule. The API and both command-line adapters retain full numeric scores; bars and percentages are a display of uncalibrated softmax values, not verified probabilities of actual age or an explanation of facial features.</p><p>The extension passed {feature["counts"]["tests_full.xml"]} Python tests, {feature["counts"]["tests_javascript.xml"]} JavaScript tests and {feature["browser_assertions"]} browser workflow assertions. Selected weights, preprocessing and research metrics remain unchanged. A browser-discovered difference between API hyphens and display en dashes was corrected and covered by regression checks.</p>')
    hardware_path = REPORTS / 'hardware_acceptance.json'
    hardware_note = 'Physical webcam/real-phone acceptance remains pending.'
    if hardware_path.exists():
        hardware = read('hardware_acceptance.json')
        if hardware['webcam']['status'] == 'user_confirmed_working' and hardware['phone']['status'] == 'not_required_by_user':
            hardware_note = 'The user confirmed that the physical webcam works correctly. Phone testing is not required by the user.'
    parts.append(f'<h2>Remaining scope</h2><p>{hardware_note} An independent consented-person pilot, public hosting and a fresh delivery package remain separate. Webcam functionality does not establish real-camera age-classification accuracy. No Colab run is required. The unfinished seed 44 run and nine deferred supporting experiments remain excluded. Age estimates support human ID checks and do not establish actual age or purchase/discount eligibility.</p>')
    styles = 'body{font:16px/1.65 system-ui,sans-serif;color:#1d3342;background:#f4f7f9;margin:0}main{max-width:1120px;margin:32px auto;padding:40px;background:white;border-radius:16px}h1{font-size:32px;line-height:1.2}h2{margin-top:36px;font-size:23px}.lead{font-size:18px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dce5eb;padding:10px;vertical-align:top;overflow-wrap:anywhere}th{background:#edf4f7}figure{margin:36px 0}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#536a7b}@media(max-width:700px){main{margin:0;padding:20px;border-radius:0}h1{font-size:26px}}'
    report = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisAge two seed evaluation</title><style>' + styles + '</style><main>' + ''.join(parts) + '</main></html>'
    (REPORTS / 'two_seed_report.html').write_text(report, encoding='utf-8')
    write_json(REPORTS / 'two_seed_summary.json', {'generated_utc': utc_now(), 'completed_runs': 6, 'seeds': [42, 43],
        'default_method': selection['default_method'], 'validation_runs': rows, 'selection': selection,
        'test_metrics': {m: r['metrics'] for m, r in test['methods'].items()}, 'pipeline': pipeline,
        'training_stopped': True, 'deferred_supporting_runs': 9, 'application_acceptance_complete': acceptance.exists()})
    print('Exported the six-run CSVs, four figure sets and self-contained HTML report.', flush=True)


if __name__ == '__main__':
    main()

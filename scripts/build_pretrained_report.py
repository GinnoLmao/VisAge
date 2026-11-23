"""Build the final supervision report using the bundled document Python runtime."""
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_project_documentation import inline, add_field

REPORTS = ROOT / 'reports/pretrained'
NAMES = {'convnextv2_nano': 'ConvNeXt V2 Nano', 'efficientnetv2_s': 'EfficientNetV2 S', 'swinv2_tiny': 'Swin Transformer V2 Tiny'}
METHODS = tuple(NAMES)


def main():
    def read(relative):
        return json.loads((REPORTS / relative).read_text(encoding='utf-8'))
    runs = read('verified_runs.json')['runs']
    selection = read('selection.json')
    final_test = read('test_summary.json')
    pipeline = read('face_pipeline_validation.json')
    robustness = read('robustness.json')
    devices = read('device_verification.json')
    benchmark = read('benchmark/backend_benchmark.json')
    acceptance = read('web_app/automated_acceptance.json')
    visual = read('web_app/visual_review.json')
    package = read('package.json')
    extracted = read('package_verification.json')
    if len(runs) != 18 or not all(record['ok'] for record in (final_test, pipeline, robustness, devices,
        benchmark, acceptance, visual, package, extracted)):
        raise ValueError('Final report requires all real completion and extracted-package evidence.')
    default = selection['default_method']
    protocol = json.loads((ROOT / 'configs/pretrained/protocol.json').read_text())
    records = {r['run_id']: r for r in runs}
    browser_events = read('web_app/browser_events_final.json')
    browser = next(e for e in reversed(browser_events) if e.get('event') == 'upload-integration')
    python_tests = len(ET.parse(REPORTS / 'web_app/tests_full.xml').findall('.//testcase'))
    javascript_tests = len(ET.parse(REPORTS / 'web_app/tests_javascript.xml').findall('.//testcase'))
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(.72)
    section.left_margin = section.right_margin = Inches(.8)
    section.header_distance = section.footer_distance = Inches(.3)
    for name in ('Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption'):
        style = doc.styles[name]
        style.font.name = 'Calibri'; style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.italic = False
        style.paragraph_format.widow_control = True
    doc.styles['Normal'].font.size = Pt(11)
    doc.styles['Normal'].paragraph_format.space_after = Pt(7)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.1
    doc.styles['Subtitle'].font.size = Pt(14)
    doc.styles['Subtitle'].paragraph_format.space_after = Pt(12)
    for name, size in (('Title', 28), ('Heading 1', 17), ('Heading 2', 12)):
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].font.bold = True
        doc.styles[name].paragraph_format.keep_with_next = True
        doc.styles[name].paragraph_format.space_before = Pt(14)
        doc.styles[name].paragraph_format.space_after = Pt(7)
    doc.styles['Caption'].font.size = Pt(9)
    footer = section.footer.paragraphs[0]
    footer.add_run('VisAge implementation and verification     ')
    add_field(footer, ' PAGE '); footer.add_run(' of '); add_field(footer, ' NUMPAGES ')
    for run in footer.runs:
        run.font.size = Pt(8)
    md = []
    title = 'VisAge Pretrained Model Implementation Report'
    doc.add_paragraph(title, style='Title'); md.append('# ' + title)
    date = datetime.now(timezone(timedelta(hours=8))).strftime('%B %d %Y')
    doc.add_paragraph('Execution evidence and supervision record', style='Subtitle')
    md.append('Execution evidence and supervision record')
    def prose(text):
        # Keep code paths/checksums literal while spacing prose quantities.
        tokens = re.split(r'(`[^`]*`)', text)
        for i, token in enumerate(tokens):
            if token.startswith('`'):
                continue
            token = re.sub(r'\b(all|minimum|maximum|patience|epoch|seed|batch|quality|radius|side|top|Under|ages|over|about|CUDA|ImageNet|LR|RTX)(?=\d)', r'\1 ', token)
            token = re.sub(r'(?<=\d)(?=(?:GB|GiB|MiB|ms|pixel|images|epochs|percent|requests)\b)', ' ', token)
            tokens[i] = token
        return ''.join(tokens)
    def p(text):
        text = prose(text)
        paragraph = doc.add_paragraph(); inline(paragraph, text); md.append(text)
    def h(text, level=1):
        doc.add_heading(text, level=level); md.append('#' * (level + 1) + ' ' + text)
    def table(headers, rows, ratios=None):
        ratios = ratios or ([.32] + [.68 / (len(headers) - 1)] * (len(headers) - 1))
        node = doc.add_table(rows=1, cols=len(headers)); node.autofit = False
        node.alignment = WD_TABLE_ALIGNMENT.CENTER
        for col, ratio in zip(node.columns, ratios, strict=True):
            col.width = Inches(6.9 * ratio)
        borders = OxmlElement('w:tblBorders')
        for edge in ('top', 'bottom', 'left', 'right', 'insideH', 'insideV'):
            item = OxmlElement('w:' + edge); item.set(qn('w:val'), 'single'); item.set(qn('w:sz'), '4'); item.set(qn('w:color'), 'D9D9D9'); borders.append(item)
        node._tbl.tblPr.append(borders)
        for index, words in enumerate([headers, *rows]):
            cells = node.rows[0].cells if index == 0 else node.add_row().cells
            row_props = cells[0]._tc.getparent().get_or_add_trPr()
            row_props.append(OxmlElement('w:cantSplit'))
            if index == 0:
                row_props.append(OxmlElement('w:tblHeader'))
            for j, (cell, value, ratio) in enumerate(zip(cells, words, ratios, strict=True)):
                cell.width = Inches(6.9 * ratio); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                props = cell._tc.get_or_add_tcPr()
                shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), '17365D' if index == 0 else ('EFF4F8' if index % 2 == 0 else 'FFFFFF')); props.append(shade)
                margins = OxmlElement('w:tcMar')
                for edge in ('top', 'bottom', 'left', 'right'):
                    item = OxmlElement('w:' + edge); item.set(qn('w:w'), '85' if edge in ('top', 'bottom') else '105'); item.set(qn('w:type'), 'dxa'); margins.append(item)
                props.append(margins)
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.line_spacing = 1.05
                paragraph.paragraph_format.keep_with_next = index == 0
                if j > 0 and len(str(value)) < 18:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = paragraph.add_run(prose(str(value))); run.font.size = Pt(10.5 if len(headers) <= 3 else 9.8)
                if index == 0:
                    run.bold = True; run.font.color.rgb = RGBColor(255, 255, 255)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        md.append('\n'.join(['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |',
            *['| ' + ' | '.join(str(v).replace('\n', ' ') for v in row) + ' |' for row in rows]]))
    def figure(name, caption, width=6.65):
        path = REPORTS / 'figures' / (name + '.png')
        with Image.open(path) as image:
            ratio = image.height / image.width
        width = min(width, 7.6 / ratio)
        paragraph = doc.add_paragraph(); paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(); run.add_picture(str(path), width=Inches(width))
        run._r.xpath('.//wp:docPr')[0].set('descr', caption)
        doc.add_paragraph(caption, style='Caption')
        md.append(f'![{caption}](figures/{name}.png)')
    pct = lambda value: f'{value * 100:.2f}%'
    p(f'Prepared {date} in Asia Manila time. The migration to three pretrained classifiers is implemented, evaluated and integrated into the local VisAge web application. The validation-selected default is **{NAMES[default]}**. All18 declared training jobs, new-model evaluation, software regression checks and isolated package startup checks are complete.')
    p('This report lets the project supervisor inspect what was built, how the models were selected, which evidence supports the application, and what remains outside the completed local milestone. Physical webcam and phone acceptance, public hosting, and an independent consented-person pilot remain pending. No pilot results or Colab execution are claimed.')
    table(['Completion area', 'Verified outcome'], [
        ['Training', '18 completed jobs with frozen partitions, schedule, seeds, checkpoints and independently checked prediction metrics'],
        ['Application', 'Three pretrained models available through upload, selected-face overlays, camera capture and Compare all'],
        ['Software checks', f'{python_tests} Python tests and {javascript_tests} JavaScript tests with no failures or skips; {len(browser["passed"])} actual browser assertions'],
        ['Delivery', f'Fresh extraction starts its own models, detector and dashboard; real HTTP predictions match the workspace app']])
    h('Project purpose and scope')
    p('VisAge estimates one of three age groups from a selected face: Under18, ages18–59, or60+. The intended demonstration supports staff review of ID checks for age-restricted purchases and senior-discount requests. The model does not verify a birth date, identify a person, establish liveness, or approve a transaction. The interface directs staff to the required identification.')
    p('The implementation compares pretrained systems under a common project recipe. It does not establish that architecture alone causes any performance difference: capacities and published pretraining recipes differ. A pretrained ImageNet backbone becomes an age-group estimator only after its task classifier is adapted and evaluated.')
    h('System architecture and implemented technology')
    p('The responsive browser interface uses HTML, CSS, native JavaScript modules and SVG. FastAPI and Uvicorn serve the page and JSON API from one origin. Images enter bounded memory-only admission, EXIF-aware decoding, YuNet face detection, explicit face selection and five-point alignment. A single224-pixel canonical crop then reaches every selected classifier through the same whole-face256-pixel input geometry. PyTorch and timm supply the three backbones and strict artifact loading.')
    p('The API loads the three selected models once per process. A bounded admission queue and a shared execution lock serialize detector/model work. Typed errors cover missing, multiple, tiny or unusable faces, invalid/oversize images, upload/queue timeouts and interrupted requests. Request-finally cleanup releases images and prepared crops. Browser state invalidation prevents earlier requests from restoring stale results.')
    table(['Active ID', 'Published architecture', 'Features', 'Total params', 'Head params'], [[m,
        protocol['models'][m]['checkpoint'],
        str(next(r for r in runs if r['method'] == m)['best']['pretrained']['configuration']['num_features']),
        f'{next(r for r in runs if r["method"] == m)["best"]["parameters"]:,}',
        f'{3 * (next(r for r in runs if r["method"] == m)["best"]["pretrained"]["configuration"]["num_features"] + 1):,}']
        for m in METHODS], [.23, .35, .12, .16, .14])
    h('Pretrained provenance and licensing')
    for method in METHODS:
        info = next(r for r in runs if r['method'] == method)['best']['pretrained']
        p(f'**{NAMES[method]}** uses publisher `{info["repository"]}`, immutable revision `{info["revision"]}`. Original weights have SHA256 `{info["sha256"]}` and the publisher license tag `{info["license"]}`. The [publisher model card](https://huggingface.co/{info["repository"]}/blob/{info["revision"]}/README.md), configuration, provenance and archived license text are included under `licenses/pretrained`.')
    p('The loader verifies the downloaded safetensors hash and every backbone parameter/persistent buffer against the published checkpoint, replacing only the ImageNet classifier. Older Swin exports also serialized five deterministic attention masks; each is compared in full against the library-generated mask before being excluded as a nonpersistent derived buffer. A failed pretrained load aborts a research run.')
    h('Dataset freeze and image preparation')
    p('The existing provenance-checked public UTKFace mirror and immutable manifest were retained:16,300 training,3,492 validation and3,493 test images. Labels, source bytes, duplicate grouping and membership are verified. Original mirror provenance is retained in `data/provenance.json`; no author-published checksum was available to prove byte identity with the unavailable original download. The dataset ages are annotations rather than verified birth-record ages. Perceptual groups do not establish subject-disjoint splits because UTKFace lacks participant IDs.')
    p('Training/validation images use EXIF transpose, RGB conversion, complete-face bicubic resizing to224 then256, and CHW uint8 storage. The separate mapped cache contains only training and validation pixels. Batch loading copies the current batch rather than retaining the full image dataset in RAM or VRAM. All families use256-pixel inputs without an additional center crop; ConvNeXt/Swin use ImageNet normalization and the selected TensorFlow EfficientNet weights use mean/std0.5. This adapts the published ImageNet input recipes.')
    p(f'The frozen training protocol SHA256 is `{selection["protocol_sha256"]}`. The original35 protected evidence files and16 original research sources remain checksum-verified. The prior app delivery ZIP remains the separate rollback snapshot; current dashboard numbers come from the new checkpoints.')
    h('Training design and run completion')
    p('Each adapted head emits three logits and uses class-weighted cross entropy. Per-run weights are normalized inverse-square-root counts computed only from that run’s training labels. No weighted sampler is combined with this loss. Accumulated gradients divide summed weighted loss by the weight sum of the whole effective batch; a dedicated test compares this with a physical full-batch update.')
    p('The GPU pilot selected a common physical batch32, channels-last layout and BF16 on the6GB RTX4050 Laptop GPU. All backbones passed finite forward/backward, frozen-backbone and unfrozen-update checks. AdamW uses head LR0.0005, backbone LR0.0001 and weight decay0.01, with two head-only epochs, then unfreezing and a cosine schedule. Validation stays FP32. The budget is minimum8/maximum24 total epochs with patience5 on validation macro F1. Frozen-feature baselines retain the backbone in evaluation mode for their entire budget.')
    table(['Study', 'Models and seeds', 'Completed jobs'], [
        ['Full data primary comparison', 'Three families with seeds42,43,44', '9'],
        ['Frozen backbone component ablation', 'Three families with seed42', '3'],
        ['Training data efficiency', 'Common nested whole-group25% and50% subsets; three families with seed42', '6']])
    table(['Model and experiment', 'Seed', 'Epochs run', 'Best epoch', 'Val macro F1'], [[NAMES[r['method']] + '\n' + r['experiment'], str(r['seed']), str(r['epochs']), str(r['best']['epoch']), f'{r["verified_metrics"]["macro_f1"]:.4f}'] for r in runs], [.40, .12, .15, .15, .18])
    p(f'Total recorded completed-run training duration is {sum(r["elapsed_seconds"] for r in runs) / 3600:.2f} hours. Each job retains best/resume checkpoints, optimizer and Python/NumPy/Torch/CUDA RNG state, stage transitions, environment/source/data/checkpoint identities and epoch curves. The mapped cache addresses the measured limited host RAM; one GPU training job runs at a time.')
    doc.add_page_break()
    h('Training curves for the three serving candidates')
    figure('serving_training_curves', 'Seed42 curves. Training F1 is measured online on augmented batches; validation F1 uses deterministic FP32 inputs. The star marks the selected checkpoint. The dotted line marks the transition after two head-only epochs.')
    h('Validation results and default selection')
    p('The default uses mean validation macro F1 across the three full-data seeds. Families within0.005 of the highest mean are near-tied. The fixed tie-break order is lower mean under18-predicted-older error, higher mean senior recall, lower isolated CPU median latency, then registry order for an exact remaining tie. Supporting studies and test scores do not select the default. Each app family uses its seed42 best-validation checkpoint rather than its best-performing seed.')
    table(['Model', 'Mean macro F1', 'Sample std', 'Under18 older error', 'Senior recall'], [[NAMES[m], f'{v["macro_f1_mean"]:.4f}', f'{v["macro_f1_std_sample"]:.4f}', pct(v['under18_older_mean']), pct(v['senior_recall_mean'])] for m, v in selection['validation_means'].items()], [.34, .18, .14, .18, .16])
    p(f'The eligible near-tie set was {", ".join(NAMES[m] for m in selection["near_tied_candidates"])}. The rule selected **{NAMES[default]}**. The immutable decision and primary evidence hashes are in `reports/pretrained/selection.json`. No test results were used for this decision.')
    for method in METHODS:
        record = records[f'{method}_full_seed42']
        stage = 'head only' if record['best']['epoch'] <= 2 else 'fine tuned'
        p(f'{NAMES[method]} serves epoch{record["best"]["epoch"]} from seed42, selected in the **{stage}** stage. A full-data run can retain a warm-up checkpoint when subsequent fine-tuning fails to surpass it; this outcome is reported rather than relabeled as a successful fine-tuned checkpoint.')
    h('Per class validation metrics and confusion counts')
    for method in METHODS:
        metrics = records[f'{method}_full_seed42']['verified_metrics']
        h(NAMES[method], 2)
        p(f'Seed42 aligned-validation accuracy is {pct(metrics["accuracy"])}; macro F1 is {metrics["macro_f1"]:.4f}; weighted F1 is {metrics["weighted_f1"]:.4f}.')
        table(['Age group', 'Support', 'Precision', 'Recall', 'F1'], [[v['name'], str(v['support']), f'{v["precision"]:.4f}', f'{v["recall"]:.4f}', f'{v["f1"]:.4f}'] for v in metrics['per_class']], [.28, .18, .18, .18, .18])
        table(['Provided group', 'Pred Under18', 'Pred 18 to 59', 'Pred 60 plus'], [[name, *map(str, metrics['confusion_matrix'][i])] for i, name in enumerate(['Under18', '18 to59', '60 plus'])], [.31, .23, .23, .23])
    h('ROC evaluation and independent metric verification')
    roc = read('evaluation/classification_roc.json')
    table(['Model', 'AUC Under18', 'AUC 18 to59', 'AUC 60 plus', 'Macro AUC'], [[NAMES[m], *[f'{roc["methods"][m]["roc"]["per_class"][str(k)]["auc"]:.4f}' for k in range(3)], f'{roc["methods"][m]["roc"]["macro_auc_mean"]:.4f}'] for m in METHODS], [.34, .165, .165, .165, .165])
    p('Each class is evaluated against the other two using softmax ranking scores. Macro AUC is the arithmetic mean of the three class AUCs. An independent Mann–Whitney average-rank calculation verifies the ROC integral, including ties. Saved CSVs are checked for exact frozen IDs/order/labels, finite normalized scores, argmax agreement and independently counted confusion matrices. Softmax values are not calibrated certainty for a visitor.')
    h('Frozen feature transfer and data efficiency')
    table(['Model', 'Frozen F1', '25 percent F1', '50 percent F1', 'Full data F1'], [[NAMES[m], *[f'{next(r for r in runs if r["method"] == m and r["experiment"] == label and r["seed"] == 42)["verified_metrics"]["macro_f1"]:.4f}' for label in ('frozen', 'subset25', 'subset50', 'full')]] for m in METHODS], [.34, .165, .165, .165, .165])
    p('The25% and50% subsets use identical nested whole-group membership across families and leave the full validation split unchanged. Their ratios are approximate because groups remain indivisible. These seed42 studies describe component and data-budget behavior; they are not additional opportunities to tune the serving model.')
    h('Bounded validation robustness')
    conditions = list(robustness['methods'][METHODS[0]]['conditions'])
    for condition in conditions:
        table([condition.replace('_', ' ').title(), *[NAMES[m] for m in METHODS]], [['Macro F1', *[f'{robustness["methods"][m]["conditions"][condition]["metrics"]["macro_f1"]:.4f}' for m in METHODS]]], [.25, .25, .25, .25])
    p('Every condition uses all3492 validation images: clean, brightness0.65, brightness1.35, Gaussian blur radius1.5, JPEG quality35/subsampling2,56-pixel down/up sampling, and a5% right shift with black fill. Clean predictions must reproduce the selected validation checkpoint. This bounded aligned-crop study does not establish ordinary-photo or population robustness.')
    h('Existing test benchmark after selection freeze')
    p('The3,493-image test partition had already been observed earlier in the project. The new seed42 candidates receive one final durable, resumable FP32 evaluation after the selection decision is frozen. It is an existing benchmark, not an untouched independent confirmation set. Its scores do not change model choice or subsequent training.')
    table(['Model', 'Accuracy', 'Macro F1', 'Weighted F1', 'Macro AUC'], [[NAMES[m], pct(final_test['methods'][m]['metrics']['accuracy']), f'{final_test["methods"][m]["metrics"]["macro_f1"]:.4f}', f'{final_test["methods"][m]["metrics"]["weighted_f1"]:.4f}', f'{final_test["methods"][m]["roc"]["macro_auc_mean"]:.4f}'] for m in METHODS], [.34, .165, .165, .165, .165])
    h('Face pipeline coverage and detection AP')
    p(f'The unchanged production face-processing path attempted all{pipeline["attempted"]} validation crops, accepted{pipeline["accepted"]} and rejected{pipeline["rejected"]}, yielding {pct(pipeline["coverage"])} coverage. Rejection codes were `{json.dumps(pipeline["rejection_codes"])}`. All models receive the same accepted face pixels. This redetection/realignment diagnostic uses existing research crops; it is not an independent local-camera pilot.')
    table(['Model', 'Accepted accuracy', 'Accepted macro F1', 'All attempted correct'], [[NAMES[m], pct(pipeline['methods'][m]['accepted_image_metrics']['accuracy']), f'{pipeline["methods"][m]["accepted_image_metrics"]["macro_f1"]:.4f}', pct(pipeline['methods'][m]['end_to_end_correct_rate'])] for m in METHODS], [.34, .22, .22, .22])
    detection = read('evaluation/detection_ap.json')
    p(f'The unchanged pinned YuNet detector retains its WIDER FACE validation evidence over{detection["images"]} images: AP50 {detection["metrics"]["AP50"]["ap_percent"]:.2f}% and AP75 {detection["metrics"]["AP75"]["ap_percent"]:.2f}%. Reuse is bound to detector bytes/revision, annotation/provenance/prediction hashes and the original benchmark configuration. Twelve fixed reload fixtures reproduced exact boxes and scores. This is reused detector-only evidence, not newly measured classifier performance.')
    p('The AP sweep uses a0.05 confidence floor, NMS0.3, maximum image side1280 and top5000 detections. The app uses0.9 before selection/quality checks. The declared all-valid-box ignore protocol is not the official WIDER easy/medium/hard leaderboard score. The dashboard shows these different operating points and scope notes.')
    h('Runtime and numerical verification')
    table(['Model', 'CPU median ms', 'CUDA median ms', 'Checkpoint MiB', 'CPU CUDA max score difference'], [[NAMES[m], f'{selection["validation_means"][m]["cpu_median_ms"]:.2f}', f'{devices["methods"][m]["cuda_batch_one_median_ms"]:.2f}', f'{devices["methods"][m]["artifact_bytes"] / 1024**2:.2f}', f'{devices["methods"][m]["maximum_absolute_score_difference"]:.3g}'] for m in METHODS], [.34, .165, .165, .165, .165])
    p('CPU timing includes crop resizing and normalization; CUDA timing uses already decoded/resized uint8 inputs and includes normalization/device transfer. These scopes differ and are not a direct speedup ratio. Both use thirty warm batch-one measurements after warm-up and no concurrent training. Full FP32 CPU/CUDA scores are checked on twelve fixed validation crops with absolute tolerance0.0002 and relative tolerance0.001, and labels must agree. Six raw-image checks also load all three models together and require identical selected-pixel hashes and labels across devices.')
    p(f'The isolated real HTTP CPU benchmark recorded {benchmark["http_round_trip"]["median_ms"]:.2f}ms median and {benchmark["http_round_trip"]["p95_ms"]:.2f}ms p95 over{benchmark["iterations"]} sequential Compare-all requests after{benchmark["warmups"]} warm-ups. It repeats one existing validation image through admission, detection, shared alignment/quality and all classifiers. It does not measure concurrent load, hosted networks or camera-domain accuracy. Owned server trees were stopped and their ports closed.')
    h('Application changes and acceptance evidence')
    p('The active API IDs, direct/aligned and raw-image CLIs, default resolution, local-pilot runner, menus, cards, About explanations, dashboard tables, ROC/confusion selectors and recommendation badge use the three pretrained families. The default comes from the serving bundle. Active evaluation records are checked against the served model hashes and common dataset freeze; historical classifier scores are not presented under the new names.')
    p(f'The final project run passed {python_tests} Python tests and {javascript_tests} JavaScript tests without skips, plus {len(browser["passed"])} browser integration assertions. Real-artifact tests compare identical prepared pixels across adapters, both CLIs and actual HTTP requests. Dependency, lint, wheel-asset and cleanup checks passed. Desktop/mobile visual review is recorded separately in `web_app/visual_review.json`.')
    table(['Acceptance scope', 'Evidence'], [
        ['Uploaded images and selection', 'Drop/picker, preview, explicit face selection, Compare-all, crop agreement, EXIF orientation and typed input/face errors'],
        ['State recovery', 'Photo/face/method changes clear prior output and overlay age; reset, cancellation, timeout and reconnect checked'],
        ['Browser camera', 'Simulated live stream, switching, capture through all three classifiers, retake and track shutdown checked'],
        ['Evaluation interface', 'Current names/default, three ROC/confusion selectors, checksum-bound metrics, AP and coverage with scope labels'],
        ['Packaging', 'Wheel assets match source; ZIP content hashes/CRC verified; fresh extraction starts its own source/models/detector and predicts all three methods']])
    h('Corrections and engineering decisions')
    p('The Swin checkpoint initially failed strict loading because its older export included regenerated attention masks. The loader now verifies their exact generated values and continues strict loading of every learned tensor/persistent buffer. CPU reload tests cover all three families. The initial staged API test invocation imported the active source directory; an absolute isolated source path corrected the harness, after which all46 staged API/service/pilot/shell tests passed. Those preflight counts are separate from the final full-project run.')
    p('The original host RAM measurement justified mapped batch loading. Channels-last layout was accepted only after full-precision output-equivalence and GPU-throughput/memory pilots. Pretrained files resolve to immutable publisher revisions and never silently fall back to random weights. Full model checkpoint/state and final CSV checks prevent substituting historical artifacts or mislabeling uncompleted work as research evidence.')
    h('Delivery files and local operation')
    p('Start the app with `.venv/Scripts/python.exe scripts/serve_api.py --device cpu` from the project directory and open `http://127.0.0.1:8000`. CUDA inference is available with `--device cuda`. The responsive page and API share one origin; there is no frontend build or external font/CDN dependency. The updated README contains environment, training continuation, API and rollback instructions.')
    p(f'The new archive is `output/VisAge_Pretrained_Web_App.zip`, with {package["files"]} checksum-listed files and {package["bytes"] / 1024**2:.2f}MiB. SHA256: `{package["sha256"]}`. It excludes raw research photos, visitor images, caches, environments and optimizer/resume state. Exact model/detector attribution is included. The final supervision report is delivered alongside the archive so it can include the archive’s completed extraction verification without a self-hash dependency.')
    p(f'Fresh-extraction verification used isolated Python mode and the extraction’s source, three serving checkpoints, detector and evaluation records. It checked {extracted["manifest_files_verified"]} manifest entries, loaded the page/dashboard and matched all three labels to the workspace app for the same existing fixture. Temporary verification servers were stopped. See `reports/pretrained/package_verification.json` for the extraction path, shared crop hash and cleanup record.')
    h('Remaining scope and interpretation limits')
    p('Physical webcam and real-phone browser acceptance remain to be performed. The browser camera evidence uses simulated video tracks and an existing validation fixture. A consented participant pilot still needs reliable age-group labels, participant-disjoint development/holdout assignment and permission records; its protocol/template/runner are prepared and no outcomes are fabricated. Public hosting remains a separate deployment phase.')
    p('Dataset annotation noise, age-boundary ambiguity, uncertain subject identity separation, convenience-sample bias and the gap between aligned research crops and real photos limit interpretation. Shared hyperparameters and image sizes support a practical system comparison but do not isolate architecture from pretraining/capacity. The ConvNeXt checkpoint’s noncommercial terms and dataset/photo permissions remain relevant to any future deployment beyond this academic prototype.')
    h('Evidence index')
    table(['File or directory', 'Purpose'], [
        ['configs/pretrained/protocol.json', 'Immutable18-run training design, source/input hashes and default-selection rule'],
        ['reports/pretrained/verified_runs.json', 'All completed runs and independently verified validation metrics'],
        ['reports/pretrained/selection.json', 'Frozen validation-only default decision and supporting hashes'],
        ['reports/pretrained/test_summary.json', 'Existing benchmark evaluation after decision freeze'],
        ['reports/pretrained/robustness.json', 'Seven full-validation perturbations for each serving family'],
        ['reports/pretrained/face_pipeline_validation.json', 'Accepted/rejected denominator accounting and new-model outcomes'],
        ['reports/pretrained/evaluation', 'Current classifier ROC and reused detector AP provenance'],
        ['reports/pretrained/device_verification.json', 'FP32 CPU/CUDA outputs and combined raw-pipeline consistency'],
        ['reports/pretrained/web_app', 'Regression, HTTP, browser, visual, wheel and process-cleanup records'],
        ['reports/pretrained/figures', 'Standalone high-resolution PNG and PDF research figures'],
        ['reports/pretrained/package_verification.json', 'Fresh extraction and real three-model HTTP prediction proof']])
    # Remove all title/heading border residue, including inherited theme borders.
    for root in (doc.styles.element, doc.element):
        for node in root.xpath('.//w:pBdr'):
            node.getparent().remove(node)
    doc.core_properties.title = title
    doc.core_properties.subject = 'Three pretrained age group models and local web application verification'
    doc.core_properties.author = 'VisAge Project'
    output = ROOT / 'output/docx/VisAge_Pretrained_Implementation_Report.docx'
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    (REPORTS / 'implementation_report.md').write_text('\n\n'.join(md) + '\n', encoding='utf-8')
    print(f'Built {output}', flush=True)


if __name__ == '__main__':
    main()

"""Build the complete evidence-led system report as PDF and self-contained HTML.

Run with the bundled reportlab Python runtime; model inference is not performed.
"""
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/pretrained'
OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
DOCS=ROOT/'docs';DOCS.mkdir(exist_ok=True)
METHODS=['swinv2_tiny','convnextv2_nano','efficientnetv2_s']
NAMES={'swinv2_tiny':'Swin Transformer V2 Tiny','convnextv2_nano':'ConvNeXt V2 Nano','efficientnetv2_s':'EfficientNetV2-S'}
GROUPS=['Under 18','18-59','60+']


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def pct(value):return f'{value*100:.2f}%'
def number(value):return f'{value:.4f}'
def clean(value):return str(value).replace('\u2013','-').replace('\u2014',' - ').replace('\u2011','-').replace('\u2019',"'").replace('\u00a0',' ')


def content():
    runs=read('reports/pretrained/verified_runs.json')['runs']
    selection=read('reports/pretrained/selection.json')
    test=read('reports/pretrained/test_summary.json')
    pipeline=read('reports/pretrained/face_pipeline_validation.json')
    robust=read('reports/pretrained/robustness.json')
    audit=read('data/processed/audit.json')
    provenance=read('data/provenance.json')
    profile=read('reports/pretrained/evaluation/application_profile.json')
    benchmark=read('reports/pretrained/benchmark/backend_benchmark.json')
    devices=read('reports/pretrained/device_verification.json')
    acceptance=read('reports/pretrained/ui_refinement/acceptance.json')
    delivery=read('reports/pretrained/delivery/package.json')
    verification=read('reports/pretrained/delivery/verification.json')
    hosting=read('reports/pretrained/deployment_status.json')
    detector=read('reports/pretrained/evaluation/detection_ap.json')
    config=read('configs/pretrained/protocol.json')['runs'][0]
    nodes=[]
    def h(title):nodes.append(('heading',title))
    def p(text):nodes.append(('paragraph',text))
    def table(headers,rows,widths=None):nodes.append(('table',{'headers':headers,'rows':rows,'widths':widths}))
    def figure(path,caption,height=350):nodes.append(('figure',{'path':path,'caption':caption,'max_height':height}))
    def page():nodes.append(('page',None))

    h('Executive overview')
    p('VisAge is an implemented computer-vision web application that estimates one of three age groups from a selected face and compares three published pretrained vision backbones. The current system uses ConvNeXt V2 Nano, EfficientNetV2-S and Swin Transformer V2 Tiny, with Swin as the validation-selected default. The app supports uploads, camera capture, explicit face selection, face overlays, individual inference, Compare All, processing times and an offline evaluation dashboard.')
    p('The approved study contains six completed full-data runs: seeds 42 and 43 for each model. Every family serves its seed-42 best-validation checkpoint. Training remains stopped. The unfinished seed-44 run is preserved and excluded; nine supporting experiments are deferred. Independent pilot photo testing and further model improvements were explicitly excluded by the user. Physical webcam functionality was confirmed by the user; phone hardware testing is not required.')
    table(['Deliverable or check','Recorded outcome'],[
        ['Current application','Functional local FastAPI + browser application; three model scores per model'],
        ['Regression checks',f'{acceptance["counts"]["tests_full.xml"]} Python tests; {acceptance["counts"]["tests_javascript.xml"]} JavaScript tests; {acceptance["browser_assertions"]} browser assertions'],
        ['Fresh delivery','240-file archive; isolated extraction, own-source import, predictions, curves and HEIC preview verified'],
        ['Public source','https://github.com/GinnoLmao/VisAge'],
        ['GitHub Pages','Live documentation/report portal; does not execute Python inference'],
        ['Vercel','Deployment files prepared; team authorization pending; hosted inference is not claimed']],[.28,.72])
    p('The selected Swin checkpoint achieved 92.70% age-group accuracy on the 3,492-image aligned validation split and 92.59% on the 3,493-image existing test benchmark. The shared redetection/alignment pipeline accepted 3,482 of 3,492 attempts (99.71% coverage). These measurements do not establish independent real-camera performance or actual-age verification.')

    h('1. Background and practical problem')
    p('Staff handling age-related checks must often decide when to request identification during restricted purchases or senior-discount requests. Facial appearance can provide an imperfect cue, but lighting, pose, image quality and individual variation can make visual judgments inconsistent. VisAge provides a comparative visual estimate and transparent model outputs to support a human review workflow. It does not determine legal age, recognize identity, approve a transaction or replace valid identification.')
    p('The project formulates the task as three-class image classification rather than numerical age regression. The groups are Under 18, ages 18-59 and 60+. A detector first finds candidate faces. A shared selection and alignment pipeline produces one canonical face image; each classifier then processes that same region. This formulation permits an interpretable comparison of two convolutional families and a hierarchical Transformer under common partitions and a recorded training budget.')

    h('2. Objectives, requirements and completed scope')
    table(['Objective','Implementation and evidence'],[
        ['Compare three pretrained models','Published checkpoints with replacement three-logit heads; common train/validation/test membership'],
        ['Use consistent face inputs','YuNet detection, explicit face selection, five-point alignment and shared crop checksum'],
        ['Provide a usable browser workflow','Drag/drop and picker, previews, camera capture/retake/switching, clear errors, retry/reset and stale-result clearing'],
        ['Expose model outputs','All three class scores in fixed group order, chosen-class marker, timing and recommended-first display'],
        ['Show evaluation evidence','Accuracy, macro/weighted F1, per-class precision/recall, confusion matrices, ROC/AUC, detector AP and pipeline coverage'],
        ['Show system cost','Recorded CPU runtime (ms), parameters (M) and complete seed-42 run time (min)'],
        ['Deliver reproducibly','Artifact hashes, six reload checks, regression records, runtime release and fresh-extraction verification']],[.28,.72])
    p('The requested cosmetic refinements are implemented: the prototype badge and research branding were removed; the header lists the three models; the recommended model is first in displayed results; percentage headings read Model scores; the expandable score explanation and rejected-input reasons table were removed. Rejected inputs remain included in the aggregate evidence and all-attempted denominator. JPEG, PNG, WebP, HEIC/HEIF, AVIF, BMP and single-page TIFF uploads are supported.')

    h('3. System architecture and data flow')
    figure('reports/pretrained/system_figures/architecture.png','Figure 1. The shared raw-image path and three-model inference architecture.',250)
    p('The browser sends raw image bytes to a same-origin FastAPI service. A bounded admission queue serializes detector/model use and limits simultaneous retained uploads. The service decodes and orients the image, detects candidate faces, enforces face selection and quality conditions, aligns the selected face and runs one or all loaded classifiers. The response contains age-group labels, class scores, processing times and geometry metadata. Image objects are closed after each request; visitor images are not written to disk or added to training.')
    p('The evaluation dashboard reads saved, checksum-bound aggregate evidence for the loaded checkpoint hashes. It does not evaluate an uploaded person against an unknown true age. Classification, detector and shared-pipeline results have different datasets and denominators, which remain explicit in the API and interface.')

    h('4. Technology stack and execution environment')
    table(['Layer','Technology and role'],[
        ['Browser','Native HTML, CSS, JavaScript ES modules, SVG charts, browser camera and Blob previews; no frontend build framework required'],
        ['API','Python, FastAPI, Uvicorn, bounded async admission and serialized inference'],
        ['Models','PyTorch, torchvision, timm 1.0.31, safetensors, CUDA BF16 training and FP32 serving'],
        ['Images','Pillow 13.0.0, OpenCV YuNet, pillow-heif 1.8.1; bounded in-memory format conversion'],
        ['Evaluation','NumPy, SciPy, custom checked metrics/ROC, Matplotlib and immutable CSV/JSON evidence'],
        ['Verification','pytest, Node test runner, httpx, isolated Chrome and fresh archive extraction'],
        ['Publication','GitHub source/releases/Pages; Vercel ASGI configuration and CPU dependencies; optional container setup']],[.22,.78])
    p('The recorded training environment is Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8 and an NVIDIA GeForce RTX 4050 Laptop GPU with approximately 6 GiB VRAM. Deterministic settings, seeds and environment metadata were recorded. CPU-only hosting dependencies are separated in requirements.txt; the local research environment is not replaced by a cloud installation.')

    h('5. Dataset provenance, audit and partition control')
    p(f'The retained study uses the aligned/cropped UTKFace mirror at {provenance["mirror"]}, version {provenance["mirror_version"]}. The archive contains {audit["scanned_files"]:,} scanned files. The audit retained {audit["included_samples"]:,} samples and excluded {audit["excluded_files"]:,}; exact duplicate removal accounted for {audit["exact_duplicates_removed"]:,} images. It recorded {audit["conflicting_duplicate_groups"]} conflicting duplicate groups and {audit["multi_image_perceptual_groups"]} multi-image perceptual groups. Excluded paths were registered rather than silently deleted.')
    table(['Partition','Under 18','18-59','60+','Total'],[[part.title(),*[f'{audit["split_class_counts"][part][str(i)]:,}' for i in range(3)],f'{audit["split_counts"][part]:,}'] for part in ('train','validation','test')],[.24,.19,.19,.19,.19])
    p('The training partition contains 16,300 images, validation 3,492 and test 3,493. Duplicate/perceptual groups are kept together during assignment. The largest group has 12 images. The data is strongly imbalanced toward ages 18-59, motivating macro F1 and train-only class weighting. No age-group balancing sampler is added on top of the weighting.')
    p('UTKFace lacks reliable participant IDs; perceptual grouping is not proof of identity-independent partitions. Ages are dataset annotations, including automated estimates checked by annotators, not verified birth-record ages. The unavailable original archive had no author-published digest suitable for byte-for-byte mirror confirmation. The mirror archive checksum, extraction scope and local audit are recorded; this is provenance evidence, not a guarantee of original-photo rights. Dataset photos are excluded from public delivery.')

    h('6. The three model families and adapted heads')
    p('ConvNeXt V2 is a modern convolutional architecture associated with fully convolutional masked-autoencoder pretraining and global response normalization. The selected Nano checkpoint is convnextv2_nano.fcmae_ft_in1k. EfficientNetV2 is a convolutional family designed around training and parameter efficiency; this study uses tf_efficientnetv2_s.in1k. Swin Transformer V2 uses hierarchical windowed attention; the selected Tiny checkpoint is swinv2_tiny_window8_256.ms_in1k. These backgrounds motivate architectural diversity; publisher benchmark results are not substituted for VisAge results. [1-3]')
    table(['Model','Features','Head parameters','Total parameters'],[
        ['ConvNeXt V2 Nano','640','1,923','14,984,723'],['EfficientNetV2-S','1,280','3,843','20,181,331'],['Swin Transformer V2 Tiny','768','2,307','27,580,461']],[.43,.17,.2,.2])
    p('Each model reconstructs the declared timm architecture, replaces the classification task with a linear three-logit head and strictly loads the complete selected checkpoint. The head has feature_dimension x 3 weights plus three biases. Softmax converts logits to three nonnegative scores whose total is approximately one. The maximum score selects the age-group class. The model registry pins architecture identifiers, input sizes, normalization and checkpoint provenance.')
    table(['Model','Original published weight revision','Recorded publisher license'],[[NAMES[m],next(r['best']['pretrained']['revision'] for r in runs if r['method']==m),next(r['best']['pretrained']['license'] for r in runs if r['method']==m)] for m in METHODS],[.36,.4,.24])

    h('7. Shared preprocessing, detection and quality handling')
    p('Raw images pass byte-size and decoded-pixel limits, EXIF orientation and RGB normalization. The original JPEG/PNG/WebP decoder is preserved for research consistency. Additional still-image formats use a separate admission wrapper; a bounded JPEG preview is generated for browser display, while inference retains the original oriented RGB pixels. Multipage or animated inputs are rejected. This adds format support without training on preview recompression.')
    p('YuNet 2023mar supplies face boxes and five landmarks. The serving detector uses confidence 0.9, NMS 0.3, maximum detection side 1,280 and a cap of 10 faces. A missing face gives a typed error; multiple faces require explicit selection. The selected face must satisfy size, visibility, roll, alignment, brightness, blur and dynamic-range conditions. One five-point similarity alignment creates the shared 224-pixel canonical crop.')
    p('All classifiers use whole-face Pillow bicubic resizing to 224 and then 256 pixels, without an additional center crop. ConvNeXt and Swin use ImageNet mean/std; the selected TensorFlow EfficientNet checkpoint uses mean/std 0.5. This is a common adapted comparison recipe, not a claim to reproduce each publisher\'s original ImageNet evaluation setting. The selected face checksum is identical for every method in Compare All.')

    h('8. Training protocol and reproducibility')
    table(['Setting','Recorded value'],[
        ['Shared seeds','42 and 43; exactly six completed full-data runs'],['Physical/effective batch','32 / 32'],
        ['Precision and layout','CUDA BF16 training; FP32 validation; channels-last tensors'],['Optimizer','AdamW; backbone LR 0.0001; head LR 0.0005'],
        ['Regularization','Weight decay 0.01; gradient clipping 5.0'],['Stages','Two head-only epochs, followed by backbone unfreezing and fixed cosine schedule'],
        ['Stopping','Minimum 8, maximum 24 epochs; patience 5; strict best validation macro F1'],
        ['Class weighting','Normalized inverse square root of training class counts; no additional balancing sampler'],
        ['Augmentation','Independent horizontal flip 0.5; rotation -8 to +8 degrees; brightness/contrast/saturation 0.85-1.15; normalization after augmentation']],[.28,.72])
    p('Best-checkpoint selection uses the earliest highest validation macro F1. Later fine-tuning does not automatically replace an earlier better checkpoint. Training checkpoints preserve optimizer and RNG state for epoch-boundary continuation. All six completed selected artifacts, configs, histories, prediction CSVs and metadata were verified; complete checkpoint reloads reproduced every validation label and score, with maximum absolute difference 0.0 on the recorded reload check.')
    p('The original eighteen-job protocol remains preserved. A user-approved amendment, made after validation results had been observed, reduces completion scope to seeds 42 and 43. The unfinished Nano seed-44 run is retained but excluded; the other seed-44 jobs are unstarted. Three frozen-backbone and six data-efficiency jobs are deferred, with no invented results. Two seeds provide limited replication; means and sample standard deviations are descriptive, not evidence of statistical superiority.')

    h('9. Evaluation definitions and evidence scopes')
    table(['Metric','Definition and interpretation'],[
        ['Accuracy (%)','100 x correct age-group classifications / evaluated classification inputs'],
        ['Precision / recall','Per-class true positives divided by predicted positives / provided positives'],
        ['F1','2 x precision x recall / (precision + recall), with the implementation\'s zero-denominator convention'],
        ['Macro F1','Unweighted mean of the three class F1 values; exposes imbalance more clearly than accuracy alone'],
        ['Weighted F1','Per-class F1 averaged by support'],['Confusion matrix','Rows: provided age group; columns: model prediction; counts and row-normalized rates'],
        ['ROC / AUC','One age group versus the other two; complete scores generate ROC; macro AUC is the mean of class AUCs'],
        ['Detection AP','Area under the detector precision-recall curve under the recorded IoU/ignore protocol; distinct from classification accuracy'],
        ['Coverage','Accepted images / every attempted pipeline input'],['All-attempted correct rate','Correct accepted predictions / every attempted pipeline input, including rejected photos in the denominator']],[.27,.73])
    p('The dashboard presents bounded display curves for responsiveness while retaining AUC values from complete saved scores. ROC AUC is independently checked using average ranks; confusion counts and other classification metrics are independently derived from checksum-bound prediction CSVs. Display scores for one person are uncalibrated softmax outputs, not verified probabilities of actual age. The UI intentionally labels them Model scores.')

    h('10. Six completed validation runs')
    ordered=[next(r for r in runs if r['method']==m and r['seed']==s) for m in METHODS for s in (42,43)]
    table(['Model / seed','Epochs run','Selected epoch / stage','Accuracy','Macro F1','Weighted F1'],[[NAMES[r['method']]+' / '+str(r['seed']),r['epochs'],f'{r["best"]["epoch"]} / {r["selected_stage"]}',pct(r['verified_metrics']['accuracy']),number(r['verified_metrics']['macro_f1']),number(r['verified_metrics']['weighted_f1'])] for r in ordered],[.3,.11,.23,.12,.12,.12])
    p('Nano seed 42 selects epoch 1 from head-only warm-up, although the run continued for eight epochs. Its subsequent fine-tuning did not exceed that score. Nano seed 43 selects fine-tuned epoch 16. Both EfficientNet and Swin serving checkpoints are fine-tuned. The report preserves this distinction rather than relabeling every selected checkpoint as fine-tuned.')
    figure('reports/pretrained/figures_two_seed/training_curves.png','Figure 2. Actual online augmented-training and FP32 validation macro F1 for all six runs. Stars mark selected epochs.',460)
    figure('reports/pretrained/system_figures/training_loss.png','Figure 3. Recorded augmented online training loss for six runs. Validation loss was not logged in these histories; it is not reconstructed or fabricated.',440)
    p('Nano has a pronounced drop at the first unfrozen epoch in both seeds. Several runs develop a large gap between online training and validation macro F1. These are evidence of training instability and generalization limits worth reporting. The agreed best-validation rule preserves the strongest available checkpoint. No extra tuning was run to improve the appearance of these results.')

    h('11. Default selection from two-seed validation means')
    table(['Model','Mean macro F1','Sample SD','Under-18 to older','Senior recall'],[[NAMES[m],number(selection['validation_means'][m]['macro_f1_mean']),number(selection['validation_means'][m]['macro_f1_std_sample']),pct(selection['validation_means'][m]['under18_older_mean']),pct(selection['validation_means'][m]['senior_recall_mean'])] for m in METHODS],[.34,.17,.14,.18,.17])
    p('The primary ranking uses mean validation macro F1 across the two shared seeds. Families within 0.005 of the highest mean are treated as near-tied. The fixed tie-break order is lower mean under-18-to-older error, higher senior recall, lower measured CPU median, then registry order. Swin and EfficientNet are near-tied; Swin wins the first tie-break because its mean under-18-to-older error is lower. Test outcomes did not choose the default. Each family still serves seed 42, not whichever seed scored better.')

    h('12. Serving-checkpoint aligned validation results')
    selected={m:next(r for r in runs if r['method']==m and r['seed']==42) for m in METHODS}
    table(['Model','Accuracy','Macro F1','Weighted F1','Macro AUC'],[[NAMES[m],pct(selected[m]['verified_metrics']['accuracy']),number(selected[m]['verified_metrics']['macro_f1']),number(selected[m]['verified_metrics']['weighted_f1']),number(selected[m]['roc']['macro_auc_mean'])] for m in METHODS],[.37,.16,.16,.16,.15])
    for m in METHODS:
        p(NAMES[m]+' - per-class aligned validation metrics:')
        table(['Provided group','Support','Precision','Recall','F1','AUC'],[[GROUPS[c['label']],c['support'],number(c['precision']),number(c['recall']),number(c['f1']),number(selected[m]['roc']['per_class'][str(c['label'])]['auc'])] for c in selected[m]['verified_metrics']['per_class']],[.24,.14,.16,.16,.15,.15])
    figure('reports/pretrained/figures_two_seed/serving_validation_confusion.png','Figure 4. Seed-42 aligned validation confusion counts and row percentages.',240)
    figure('reports/pretrained/figures_two_seed/serving_validation_roc.png','Figure 5. Seed-42 aligned validation one-versus-rest ROC curves and per-class AUC.',240)

    h('13. Existing test benchmark: frozen-model evaluation')
    table(['Model','Accuracy','Macro F1','Weighted F1','Macro AUC'],[[NAMES[m],pct(test['methods'][m]['metrics']['accuracy']),number(test['methods'][m]['metrics']['macro_f1']),number(test['methods'][m]['metrics']['weighted_f1']),number(test['methods'][m]['roc']['macro_auc_mean'])] for m in METHODS],[.37,.16,.16,.16,.15])
    p('The test split contains 3,493 images and was observed earlier in the project. Each frozen serving family received one final resumable evaluation after the validation-only default decision was frozen. These are results on an existing benchmark, not independent confirmation. No test-guided training, threshold adjustment or model choice is claimed.')
    for m in METHODS:
        p(NAMES[m]+' - per-class existing test metrics:')
        table(['Provided group','Support','Precision','Recall','F1','AUC'],[[GROUPS[c['label']],c['support'],number(c['precision']),number(c['recall']),number(c['f1']),number(test['methods'][m]['roc']['per_class'][str(c['label'])]['auc'])] for c in test['methods'][m]['metrics']['per_class']],[.24,.14,.16,.16,.15,.15])
    figure('reports/pretrained/system_figures/test_confusion.png','Figure 6. Existing test confusion matrices for the three frozen serving checkpoints.',240)
    figure('reports/pretrained/system_figures/test_roc.png','Figure 7. Existing test one-versus-rest ROC curves; complete saved scores determine AUC.',240)

    h('14. Inference-only robustness results')
    conditions=list(robust['methods']['swinv2_tiny']['conditions'])
    table(['Condition','Recorded perturbation'],[[c.replace('_',' '),robust['methods']['swinv2_tiny']['conditions'][c]['definition']] for c in conditions],[.25,.75])
    table(['Condition','Swin V2 Tiny','ConvNeXt V2 Nano','EfficientNetV2-S'],[[c.replace('_',' '),*[number(robust['methods'][m]['conditions'][c]['metrics']['macro_f1']) for m in METHODS]] for c in conditions],[.28,.24,.24,.24])
    figure('reports/pretrained/figures_two_seed/validation_robustness.png','Figure 8. Macro F1 across all seven fixed perturbations on 3,492 aligned validation images per family.',240)
    p('These 21 conditions were inference-only checks using the selected frozen models. Clean scores reproduce the saved validation baseline. Swin remains strongest in the listed macro-F1 values, while Nano loses most under JPEG degradation. The evaluation covers exactly these deterministic perturbations; it does not prove ordinary-phone robustness, demographic fairness or out-of-distribution reliability.')

    h('15. Face-detector AP and provenance')
    p(f'The unchanged YuNet detector retains verified WIDER FACE validation evidence for {detector["images"]:,} images and {detector["metrics"]["AP50"]["ground_truth_faces"]:,} valid ground-truth faces. AP at IoU 0.50 is {detector["metrics"]["AP50"]["ap_percent"]:.2f}%; AP at IoU 0.75 is {detector["metrics"]["AP75"]["ap_percent"]:.2f}%. Detector bytes, revision, annotations, provenance and saved prediction hashes were checked, together with twelve exact detector reload comparisons.')
    p('The AP protocol evaluates all valid boxes with the recorded ignore policy and all-point integration. It differs from the official WIDER easy/medium/hard leaderboard. Its sweep uses confidence floor 0.05, versus 0.9 in the application before face selection and quality checks. AP is detector evidence, not age-group accuracy or newly trained-detector performance. The dashboard includes precision-recall and FROC curves; compact runtime serialization preserves those curves without changing AP.')
    figure('docs/screenshots/desktop_detection.png','Figure 9. Actual detector dashboard with AP50/AP75, precision-recall and FROC.',410)

    h('16. Shared-pipeline coverage and all-attempted accuracy')
    table(['Attempted','Accepted','Rejected','Coverage'],[[pipeline['attempted'],pipeline['accepted'],pipeline['rejected'],pct(pipeline['coverage'])]],[.25,.25,.25,.25])
    table(['Model','Accepted accuracy','Accepted macro F1','All-attempted correct'],[[NAMES[m],pct(pipeline['methods'][m]['accepted_image_metrics']['accuracy']),number(pipeline['methods'][m]['accepted_image_metrics']['macro_f1']),pct(pipeline['methods'][m]['end_to_end_correct_rate'])] for m in METHODS],[.38,.21,.21,.2])
    p('The 10 rejected validation inputs were NO_FACE outcomes. Every rejection stays in the denominator for all-attempted correct rate. The displayed rejection-reasons table was removed at the user\'s request; the underlying accounting and source evidence remain intact. This diagnostic redetects and realigns existing UTKFace crops, so it measures integration on that material rather than independent field-photo accuracy.')
    figure('docs/screenshots/desktop_pipeline.png','Figure 10. Pipeline dashboard with accepted/all-attempted denominators and no rejection-reasons table.',390)

    h('17. Runtime, model size and recorded training time')
    table(['Model','Runtime ms','p95 ms','Params M','Train min'],[[NAMES[m],f'{profile["methods"][m]["runtime_median_ms"]:.2f}',f'{profile["methods"][m]["runtime_p95_ms"]:.2f}',f'{profile["methods"][m]["parameters"]/1e6:.2f}',f'{profile["methods"][m]["training_seconds"]/60:.2f}'] for m in METHODS],[.38,.155,.155,.155,.155])
    p(profile['runtime_scope']+' '+profile['training_scope'])
    p(f'The isolated local CPU Compare All HTTP benchmark used {benchmark["warmups"]} warm-ups and {benchmark["iterations"]} measured requests. Median round trip was {benchmark["http_round_trip"]["median_ms"]:.2f} ms; p95 was {benchmark["http_round_trip"]["p95_ms"]:.2f} ms; server startup was {benchmark["startup_seconds"]:.2f} seconds. These are repeated requests for one fixed validation input, not concurrent-user or cloud measurements.')
    table(['Model','Warm CUDA median ms','Max CPU/CUDA score difference'],[[NAMES[m],f'{devices["methods"][m]["cuda_batch_one_median_ms"]:.2f}',f'{devices["methods"][m]["maximum_absolute_score_difference"]:.8f}'] for m in METHODS],[.4,.25,.35])
    p('CPU and CUDA labels matched on twelve fixed validation crops and six raw-image pipeline fixtures. Numeric scores satisfied the recorded tolerance (absolute 0.0002, relative 0.001). All three models were also loaded together on the GPU, with recorded peak allocated memory approximately '+f'{devices["raw_service_cpu_cuda"]["gpu_peak_allocated_mib"]:.2f}'+' MiB. This is functional consistency evidence, not a population accuracy estimate.')

    h('18. Browser implementation and score presentation')
    p('The interface uses a persistent workspace navigation, one photo source panel and result cards. A top status area names all three loaded models. The model selector, empty-state comparison and result cards place the recommended model first while API method IDs and output class order remain stable. The selected face overlay displays the estimate and clears whenever the photo, method or selection changes. Compare All uses one selected face for all methods.')
    p('Each model shows exactly three score rows: Under 18, 18-59 and 60+, with percentages, meters and a chosen-class marker. Raw numeric values remain in the API and CLI outputs. Very small positive scores display as less than 0.01% instead of rounding to exact zero; near-one scores can display as greater than 99.99%. There is no next-highest callout. The removed expandable explanation is absent, and percentage headings simply read Model scores. These cosmetic changes do not calibrate the model outputs or change the highest-score decision.')
    figure('docs/screenshots/model_scores.png','Figure 11. Actual recommended-model result card with all three class scores.',280)
    figure('docs/screenshots/desktop_analysis.png','Figure 12. Live workspace, model-name header, supported upload types and recommended-first selector.',390)
    figure('docs/screenshots/desktop_comparison.png','Figure 13. Actual three-model comparison. The validation photo is deliberately hidden in this public report; model outputs are from the same real fixture.',460)
    figure('docs/screenshots/desktop_evaluation.png','Figure 14. Actual dashboard with accuracy, F1, ROC, runtime, parameters, training time and per-class metrics.',460)

    h('19. API contracts, admission and error handling')
    table(['Endpoint','Behavior'],[['GET /health','Readiness, loaded-model count and device'],['GET /v1/models','Model IDs, names, selected default, seed/epoch, parameter count, artifact hashes and processing configuration'],['GET /v1/evaluation','Artifact-bound aggregate classifier, detector, cost and shared-pipeline evidence'],['POST /v1/faces','Raw supported image bytes; candidate boxes/landmarks and converted preview where needed'],['POST /v1/predict','Raw image bytes; optional method and face index; one or three predictions, class scores, shared crop and timing']],[.3,.7])
    p('Queries use method=all or an exact active model ID; omitting method uses the serving-bundle default. Unknown legacy method IDs are rejected rather than silently aliased. Each prediction exposes method, class_id, age_group, class_scores, score_type, timing, manifest hash and device. class_scores contains three entries with class IDs 0, 1 and 2. The frontend validates labels, fixed order, finite range, approximate sum-to-one and agreement between the chosen class and the score maximum before rendering.')
    p('The local upload cap is 8 MiB and the decoded-image cap is 20 megapixels. The Vercel adapter advertises/enforces 4 MiB because the platform has a smaller request-body ceiling; the client displays that cap and rejects oversized input before transport. Errors distinguish empty/corrupt/unsupported/animated files, missing/multiple/small/unusable faces, queue saturation, timeouts and unavailable evidence. Internal exceptions do not leak stack traces to the browser. Request cancellation, stale-operation guards and URL revocation protect the image lifecycle.')

    h('20. Camera lifecycle, privacy and operational boundaries')
    p('The camera controller requests video only after an explicit action, supports preview, capture, retake and device switching, and shuts down tracks after capture or route departure. Browser camera access needs localhost or HTTPS. Simulated camera tests exercise permissions, late-acquired streams, switching, capture and track cleanup. The user independently confirmed that the physical webcam works correctly. Phone hardware acceptance was removed from scope by the user.')
    p('Visitor images are processed in memory, not saved, logged as image bytes or added to model training. The public repository and release do not include raw dataset photos, pilot participants, consent records, credentials or unfinished optimizer checkpoints. HTTP/browser security includes same-origin controls, no-store responses, bounded admission, explicit CORS origins when configured, a restrictive content-security policy and disabled microphone/geolocation permissions. The application does not implement identity verification, liveness/anti-spoofing, user accounts or transaction decisions.')

    h('21. Verification, defects corrected and evidence limits')
    table(['Verification group','Executed result'],[['Full local regression',f'{acceptance["counts"]["tests_full.xml"]} Python tests, no failed/error/skipped cases'],['JavaScript state/format/controller tests',f'{acceptance["counts"]["tests_javascript.xml"]} passed'],['Actual browser integration',f'{acceptance["browser_assertions"]} assertions passed, including HEIC and all score rows'],['Final targeted checks','50 API, image, static-route and compact-evidence checks passed after serialization changes'],['Artifact reloads','Six strict reloads, all 3,492 validation predictions reproduced for each run'],['Fresh extraction','240 manifest entries verified; isolated imports, startup, all predictions, dashboard and HEIC passed'],['Physical webcam','Confirmed working by user; distinct from automated tests']],[.39,.61])
    p('Browser verification found an API/display age-label punctuation mismatch (18-59 versus an en-dash presentation), which was normalized without changing class IDs. A HEIC assertion originally tried to fetch a Blob URL outside the allowed connection policy; the test was corrected to inspect the decoded preview and server response. Runtime packaging revealed that compact detector records needed idempotent FROC preservation; that consumer was fixed and covered by a round-trip regression. Certificate handling uses the Windows trusted store where needed, without disabling TLS verification.')
    p('Regression records document what ran; they do not guarantee absence of all defects. Validation loss was not logged per epoch, so only recorded training loss and training/validation macro-F1 curves are reported. The fresh extraction uses verified existing Python dependencies, not a separately provisioned clean machine. No physical-phone test, independent pilot, calibration study, demographic fairness audit or hosted-inference success is inferred from local functional checks.')

    h('22. Source publication, hosting and fresh delivery')
    p('Public source: https://github.com/GinnoLmao/VisAge . The repository contains reviewed application code, tests, configs, deployment setup, attribution and photo-free documentation screenshots. Large model assets are supplied through https://github.com/GinnoLmao/VisAge/releases/tag/v0.4.0 instead of storing the largest weights in ordinary Git history. The runtime downloader pins archive size and SHA-256 and verifies each extracted file before using the known serving paths.')
    p('GitHub Pages is verified live at https://ginnolmao.github.io/VisAge/ as a documentation and report portal. It is static hosting and does not execute the Python classifiers. Vercel source/dependency configuration is prepared, but the current connection reports HTTP 403 for the account\'s team; authorization is required before a real deployment can be attempted. This report does not label the full model API as publicly hosted. No paid plan, domain purchase or separate hosting subscription has been created. [5-7]')
    table(['Delivery record','Value'],[['Archive','VisAge_Complete_Web_App_v0.4.0.zip'],['Compressed size',f'{delivery["bytes"] / 1024**2:.2f} MiB'],['Files verified',verification['files_verified']],['Archive SHA-256',delivery['sha256']],['Runtime SHA-256',delivery['runtime_archive_sha256']],['Source isolation','Python -I with extracted src inserted explicitly; editable original source is not used'],['Functional checks',', '.join(verification['checks'])]],[.28,.72])
    p(verification['environment_scope']+' '+verification['fixture_scope']+' The full PDF/HTML report is a sibling deliverable rather than being embedded in the archive, avoiding a circular dependency between a report\'s archive digest and an archive containing that report. The historical prior-version ZIP remains unchanged as a rollback.')

    h('23. Setup and operator guide')
    p('For a fresh CPU source checkout: install Python 3.12, create a virtual environment, install requirements.txt, run python scripts/download_runtime.py, then python scripts/serve_api.py --device cpu. Open http://127.0.0.1:8000. The complete delivery ZIP already contains runtime artifacts and therefore does not need the runtime download. The declared requirements still need to be installed. CUDA users should follow the recorded framework/environment instructions in the README.')
    p('To use the app, choose or capture one supported still photo. Select a face when multiple candidates are shown. Run the recommended model or Compare All. Inspect all three score rows and the age-group result, then apply the organization\'s identification policy. Replacing the photo or face clears results; use Retry for a lost connection and Reset to remove the preview. Evaluation tabs contain offline benchmark results, not uploaded-person accuracy.')
    p('The aligned-crop command is python scripts/predict.py FACE.png --method all. The raw-image command is python scripts/predict_image.py PHOTO.jpg --method all. The first assumes one already selected face; the second uses the shared detector/selection pipeline. Do not restart historical training controllers to run the app. Research reproduction additionally requires the frozen permitted dataset, full run artifacts and original protocol inputs.')

    h('24. Limitations, risks and responsible interpretation')
    p('The principal empirical limitation is domain transfer: aligned UTKFace crops and synthetic perturbations are not independent photos from the intended operational setting. Dataset labels are imperfect, and participant identity independence is unverified. Only two seeds were completed, and the approved scope amendment followed observation of validation results. The existing test benchmark was not previously untouched. These limitations constrain claims about generalization and statistical superiority.')
    p('There is no calibrated probability-of-age guarantee. Very high softmax scores can still be wrong, and ambiguous neighboring groups are especially relevant near age 18 and 60. The system has no liveness check or resistance to deliberate manipulation established by this study. The detector benchmark has a custom protocol and a different confidence operating point from serving. Overall correct rate must retain rejected attempts in its denominator. The selected ConvNeXt publisher weights have a noncommercial restriction; public code publication does not remove it or grant rights to dataset photos.')
    p('Operationally, PyTorch has a substantial dependency and cold-start footprint. GitHub Pages is a documentation host; the full inference API needs an appropriate backend. Vercel large-function eligibility, upload ceilings and account authorization must be checked in the actual deployment. Local latency measurements should not be advertised as cloud latency or multi-user capacity. The current package is verified in a fresh folder, with a separate clean-machine dependency installation still unclaimed.')

    h('25. Recommendations and future work')
    table(['Priority','Recommendation','Current scope'],[['Deployment','Complete team authorization, deploy the prepared CPU backend and verify real health/models/predict/evaluation URLs before announcing hosted inference','Pending user authorization'],['Maintenance','Retain artifact hashes, versioned releases, dependency records and original protocol snapshots; monitor errors and resource usage after hosting','Recommended operating practice'],['Installation portability','Validate installation in a separate clean environment and on the chosen Linux host','Not claimed by fresh-folder verification'],['Scientific extension','If later authorized, collect an independent consented/labelled participant sample and assess calibration, subgroup performance and age-boundary errors','Excluded from this delivery by user'],['Model experiments','If reopened, investigate Nano unfreezing instability, more seeds, frozen-backbone controls and data-efficiency studies using training/validation only','Training stopped; nine supporting runs deferred']],[.18,.57,.25])
    p('Recommendations are future options, not unfinished model-improvement actions in this delivery. The user requested that the models remain as they are and excluded pilot testing. Completing hosting authorization and checking the deployed API are deployment tasks; they do not require retraining or changing the selected validation rule.')

    h('26. Conclusions')
    p('VisAge now integrates three verified pretrained age-group classifiers into one consistent detection, selection, alignment and browser workflow. Six completed runs supply limited but explicit replication. The fixed two-seed validation rule selects Swin Transformer V2 Tiny, and the application preserves one seed-42 serving checkpoint per family. Swin has the strongest listed serving-checkpoint validation and existing-test results; EfficientNet is close in the family-mean comparison and faster in the recorded CPU timing. Nano is smaller and faster, with a materially weaker selected seed-42 result and a documented head-only best checkpoint.')
    p('The implementation is functionally verified through regression, real-artifact API/CLI checks, browser flows, CPU/CUDA consistency and a fresh, source-isolated archive extraction. All three class scores are visible, and the evaluation dashboard separates classifier accuracy, detector AP and pipeline coverage. The public repository, release assets and GitHub Pages documentation portal are established. Full Vercel-hosted inference remains dependent on team authorization and subsequent production checks. The resulting system is a review-support application with documented limits, not proof of an individual\'s chronological age or transaction eligibility.')

    h('References and primary documentation')
    refs=[('[1] ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders','https://arxiv.org/abs/2301.00808'),('[2] EfficientNetV2: Smaller Models and Faster Training','https://arxiv.org/abs/2104.00298'),('[3] Swin Transformer V2: Scaling Up Capacity and Resolution','https://arxiv.org/abs/2111.09883'),('[4] UTKFace project and dataset limitations','https://susanqq.github.io/UTKFace/'),('[5] GitHub Pages: creating a site and server-side limits','https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site'),('[6] Vercel Large Functions: 5 GB public beta and eligibility','https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size'),('[7] Vercel Functions limits','https://vercel.com/docs/functions/limitations'),('[8] pillow-heif documentation','https://pillow-heif.readthedocs.io/en/stable/'),('[9] YuNet pinned publisher source','https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet')]
    for title,url in refs:p(title+' - '+url)
    p('Model cards and their exact original revisions are preserved under licenses/pretrained. Project results in this report are computed from the local saved evidence rather than publisher results. Platform documentation was checked on November 23, 2026; hosting limits and permissions can change.')

    h('Appendix A. Serving artifact identity')
    table(['Model','Selected epoch / stage','Checkpoint SHA-256'],[[NAMES[m],str(selected[m]['best']['epoch'])+' / '+selected[m]['selected_stage'],selected[m]['artifact_sha256']] for m in METHODS],[.29,.22,.49])
    p('The serving bundle and dashboard records bind these hashes to the frozen manifest. Runtime restoration, checkpoint loading and delivery checks enforce these identities. No weight changes, retraining or test-guided tuning were performed during the interface, publication or packaging work.')

    h('Appendix B. Evidence inventory and reproduction boundaries')
    inventory=[('configs/pretrained/protocol.json','Original fixed protocol and protected-source/input identities'),('configs/pretrained/two_seed_amendment.json','Six-run scope, seed-44 exclusions and nine deferred jobs'),('reports/pretrained/verified_runs.json','Six actual histories, selected epochs, metrics and saved-prediction identities'),('reports/pretrained/reload_verification.json','Strict reload and exact full-validation prediction comparison'),('reports/pretrained/selection.json','Two-seed means, near-tie rule and default decision'),('reports/pretrained/test_summary.json','Existing-test metrics and full ROC evidence'),('reports/pretrained/robustness.json','Seven validation conditions per serving family'),('reports/pretrained/face_pipeline_validation.json','Coverage, accepted and all-attempted results'),('reports/pretrained/evaluation/detection_lineage.json','Reused detector AP provenance and exact reload checks'),('reports/pretrained/evaluation/application_profile.json','Recorded runtime, parameters and training duration'),('reports/pretrained/ui_refinement/acceptance.json','Latest interface/format regression and browser checks'),('reports/pretrained/delivery/verification.json','Fresh extraction and own-source functional verification'),('reports/pretrained/hardware_acceptance.json','User-confirmed webcam and excluded phone requirement'),('reports/pretrained/pilot_scope.json','Pilot photo testing excluded by user')]
    table(['Evidence path','Purpose'],inventory,[.46,.54])
    p('The full research workspace retains detailed predictions and histories; the runtime delivery needs only selected serving artifacts and compact dashboard evidence. Public Git history excludes private/face-photo material and large runtime weights. Complete raw model scores are retained by the app response, while aggregate report tables use rounded display values. Full-resolution screenshots are embedded in the HTML companion for zoomed inspection.')
    evidence=DOCS/'evidence';evidence.mkdir(exist_ok=True)
    summary={'report_date':'2026-11-23 Asia/Manila','completed_runs':6,'seeds':[42,43],'serving_seed':42,'default_method':selection['default_method'],'validation_means':selection['validation_means'],'serving_validation':{m:selected[m]['verified_metrics'] for m in METHODS},'existing_test':{m:test['methods'][m]['metrics'] for m in METHODS},'pipeline':{'attempted':pipeline['attempted'],'accepted':pipeline['accepted'],'coverage':pipeline['coverage'],'methods':pipeline['methods']},'cost':profile['methods'],'acceptance':acceptance['counts'],'browser_assertions':acceptance['browser_assertions'],'delivery':{'bytes':delivery['bytes'],'sha256':delivery['sha256'],'files':delivery['files']},'hosting':hosting,'pilot':'excluded_by_user'}
    (evidence/'system_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return nodes


def build_pdf(nodes):
    for name,path in [('VisAge','C:/Windows/Fonts/arial.ttf'),('VisAgeBold','C:/Windows/Fonts/arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,path))
    styles=getSampleStyleSheet()
    body=ParagraphStyle('VisAgeBody',fontName='VisAge',fontSize=10,leading=14.4,spaceAfter=9,textColor=colors.HexColor('#24394b'),splitLongWords=True)
    head=ParagraphStyle('VisAgeHeading',parent=body,fontName='VisAgeBold',fontSize=16,leading=20,spaceBefore=18,spaceAfter=11,keepWithNext=True,textColor=colors.HexColor('#173d64'))
    cell=ParagraphStyle('VisAgeCell',parent=body,fontSize=8.3,leading=11,spaceAfter=0)
    header=ParagraphStyle('VisAgeHeaderCell',parent=cell,fontName='VisAgeBold',textColor=colors.HexColor('#173d64'))
    caption=ParagraphStyle('VisAgeCaption',parent=body,fontSize=8.3,leading=11.5,textColor=colors.HexColor('#597082'),spaceAfter=14)
    table_intro=ParagraphStyle('VisAgeTableIntro',parent=body,keepWithNext=True)
    title=ParagraphStyle('VisAgeTitle',parent=head,fontSize=34,leading=40,spaceBefore=38,spaceAfter=20)
    story=[Paragraph('VisAge',title),Paragraph('Complete System Implementation<br/>and Evaluation Report',ParagraphStyle('Subtitle',parent=head,fontSize=22,leading=29,spaceAfter=25)),Paragraph('November 23, 2026 | Asia/Manila<br/>Three pretrained age-group models | Six completed training runs',body),Spacer(1,18),Paragraph('Public source, tested local delivery and evidence-led results',head),Paragraph('This report covers the implemented application, dataset, model training, validation and existing-test results, robustness, detector evidence, score display, verification, public source, delivery, limitations, recommendations and conclusions. Pilot photo testing and further model improvement were excluded by the user. Vercel-hosted inference is awaiting team authorization.',body),Spacer(1,24),Paragraph('Source: github.com/GinnoLmao/VisAge<br/>Documentation: ginnolmao.github.io/VisAge/<br/>Full-resolution figures and screenshots are included in the HTML companion.',body),PageBreak()]
    headings=[v for k,v in nodes if k=='heading']
    story.append(Paragraph('Report contents',head))
    for value in headings:story.append(Paragraph(clean(value),ParagraphStyle('Contents',parent=body,fontSize=9,leading=12,spaceAfter=3)))
    story.append(PageBreak())
    width=A4[0]-94
    def para(value,style):
        text=clean(value)
        if re.fullmatch('[a-f0-9]{64}',text):text=' '.join(text[i:i+16] for i in range(0,64,16))
        return Paragraph(escape(text).replace('\n','<br/>'),style)
    for kind,value in nodes:
        if kind=='heading':story.append(para(value,head))
        elif kind=='paragraph':story.append(para(value,table_intro if value.endswith(':') else body))
        elif kind=='table':
            rows=[[para(v,header) for v in value['headers']]]+[[para(v,cell) for v in row] for row in value['rows']]
            ratios=value['widths'] or [1/len(rows[0])]*len(rows[0]);table=Table(rows,colWidths=[width*r for r in ratios],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3f8')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#bfd0df')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce5ec')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            story.extend([table,Spacer(1,11)])
        elif kind=='figure':
            path=ROOT/value['path'];w,h=ImageReader(str(path)).getSize();scale=min(width/w,value['max_height']/h)
            image=Image(str(path),width=w*scale,height=h*scale);image.hAlign='CENTER'
            story.append(KeepTogether([Spacer(1,7),image,Spacer(1,7),para(value['caption'],caption)]))
        elif kind=='page':story.append(PageBreak())
    pdf=OUT/'VisAge_System_Report.pdf'
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('VisAge',8);canvas.setFillColor(colors.HexColor('#6b7e8e'))
        canvas.drawString(47,25,'VisAge | System implementation and evaluation | November 23, 2026')
        canvas.drawRightString(A4[0]-47,25,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=47,leftMargin=47,topMargin=39,bottomMargin=43,title='VisAge Complete System Implementation and Evaluation Report',author='VisAge Project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (DOCS/pdf.name).write_bytes(pdf.read_bytes())
    return pdf


def build_html(nodes):
    parts=['<header><p class="eyebrow">VISAGE / COMPLETE SYSTEM REPORT</p><h1>Implementation, evaluation and delivery</h1><p>November 23, 2026 · Six completed runs · Three pretrained models</p><p><a href="https://github.com/GinnoLmao/VisAge">Public source</a> · <a href="https://github.com/GinnoLmao/VisAge/releases/tag/v0.4.0">Download release</a></p></header>']
    for kind,value in nodes:
        if kind=='heading':parts.append('<h2>'+html.escape(clean(value))+'</h2>')
        elif kind=='paragraph':parts.append('<p>'+html.escape(clean(value))+'</p>')
        elif kind=='table':parts.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(clean(v))+'</th>' for v in value['headers'])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(clean(c))+'</td>' for c in row)+'</tr>' for row in value['rows'])+'</tbody></table></div>')
        elif kind=='figure':
            encoded=base64.b64encode((ROOT/value['path']).read_bytes()).decode()
            parts.append('<figure><img src="data:image/png;base64,'+encoded+'" alt="'+html.escape(value['caption'])+'"><figcaption>'+html.escape(value['caption'])+'</figcaption></figure>')
    css='body{font:16px/1.75 system-ui,sans-serif;color:#24394b;background:#f5f7fb;margin:0}main{max-width:1120px;margin:32px auto;padding:44px;background:white;border-radius:16px}h1{font-size:38px;line-height:1.2;color:#173d64}h2{font-size:25px;line-height:1.4;margin-top:42px;color:#173d64}header{padding-bottom:22px;border-bottom:1px solid #dce5ec}.eyebrow{font-size:12px;letter-spacing:1.7px}a{color:#2563eb}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:11px;border-bottom:1px solid #dce5ec;vertical-align:top;overflow-wrap:anywhere}th{background:#edf3f8}figure{margin:30px 0}img{max-width:100%;height:auto;display:block;margin:auto;max-height:1500px}figcaption{font-size:13px;color:#597082;margin-top:10px}@media(max-width:700px){main{margin:0;padding:20px;border-radius:0}h1{font-size:29px}table{min-width:650px}}'
    output='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisAge complete system report</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>'
    target=DOCS/'VisAge_System_Report.html';target.write_text(output,encoding='utf-8');(OUT/target.name).write_text(output,encoding='utf-8')
    return target


def main():
    nodes=content();pdf=build_pdf(nodes);htmlfile=build_html(nodes)
    identity={'pdf':{'path':str(pdf),'bytes':pdf.stat().st_size,'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()},'html':{'path':str(htmlfile),'bytes':htmlfile.stat().st_size,'sha256':hashlib.sha256(htmlfile.read_bytes()).hexdigest()},'sections':sum(k=='heading' for k,v in nodes),'figures':sum(k=='figure' for k,v in nodes),'visual_review':'Pending page rendering and inspection.'}
    (R/'system_report.json').write_text(json.dumps(identity,indent=2))
    print(json.dumps(identity),flush=True)


if __name__=='__main__':main()

# VisAge Project Documentation

## Pretrained facial age group classification and web application development

VisAge is an academic computer vision web application that estimates an age group from a user-selected face. It compares ConvNeXt V2 Nano, EfficientNetV2-S, and Swin Transformer V2 Tiny using the same task, dataset partitions, and evaluation definitions. Users can upload a photograph or capture a still image, select a detected face, and inspect one model or all three.

The intended application is a supervised demonstration of age-related decision support, including staff review for age-restricted purchases and senior-discount requests. The system presents an estimated age group and leaves eligibility decisions to the responsible person using the required identification and policy.

This document defines the project background, requirements, model and data choices, architecture, implementation method, environment setup, evaluation protocol, development phases, and submission deliverables. It serves as the specification for building and assessing the complete system. Empirical findings belong in the final experimental report after the declared evaluations are completed.

Project type: computer vision comparison and application development.

Primary task: three-class facial age-group classification.

Delivery target: a reproducible local web application and academic submission package.

\pagebreak

# Contents

\toc

\pagebreak

# 1 Project overview and background

## Project overview

VisAge connects a controlled model comparison with an application that can process new photographs. Its research component studies how three pretrained image-classification architectures adapt to age-group labels. Its application component makes their outputs accessible through a browser, using a shared face detector and preprocessing workflow.

The classification vocabulary is Under 18, 18-59, and 60+. Every model must return exactly one of these categories for the same selected face. The project evaluates both classification quality and the resources needed to produce a prediction. A default model is selected from validation evidence and identified in the interface with a short explanation.

## Background of the study

Facial age estimation is a visual inference task. A face image contains appearance information affected by age, lighting, pose, expression, image quality, and individual variation. A model learns statistical associations between images and provided age annotations. Those associations do not establish a person's birth date or identity.

Transfer learning adapts a model that has already learned image representations from a large dataset. For VisAge, each published ImageNet-1k backbone receives a replacement classifier for the project categories and is fine-tuned on labeled face images. The comparison examines how the selected pretrained systems perform under a common protocol. Architecture, model capacity, and original pretraining recipes are documented because they can all influence the outcome. [2-7]

UTKFace provides face images with age annotations and aligned face crops, making it suitable for investigating this classification task. Its authors identify it as a research dataset with varied image conditions and non-commercial usage restrictions. The project must assess annotation quality and document the limits of its sampled population. [8]

## Local application context

Age-related purchase and discount workflows often involve a staff member reviewing identification and applying an organizational policy. VisAge provides a consistent way to demonstrate a visual estimate, compare model behavior, and explain why a human check remains necessary. The project evaluates whether that supporting workflow is usable; it does not assume that an estimate reduces fraud, replaces identification, or satisfies a legal verification requirement.

The three categories are research labels. A category boundary must not be presented as a complete legal or organizational eligibility rule. Different transactions may require different evidence or thresholds.

## Intended users and stakeholders

The principal users are a demonstrator or staff reviewer who submits a photo, a student researcher who examines model evidence, and an instructor who assesses the methodology and working system. People whose photographs are used are stakeholders in consent, retention, and responsible presentation. The interface should use neutral category names and avoid messages that accuse, approve, or reject a person.

# 2 Problem statement and objectives

## Problem statement

A facial age-group demonstration needs more than a classifier that accepts an aligned crop. It must reliably locate faces in ordinary images, make the selected face unambiguous, apply consistent preprocessing, and present estimates with useful error handling. A comparison is difficult to interpret when models use different partitions, incompatible labels, or incomplete efficiency measurements.

The project therefore addresses this question: how can three pretrained classifiers be compared fairly for facial age-group estimation and integrated into a usable web application that makes the scope of their estimates clear?

## General objective

Develop and evaluate VisAge as a web application that compares three pretrained models for age-group classification using a shared face-processing pipeline and a reproducible experimental protocol.

## Specific objectives

- Prepare an auditable face dataset with consistent age-group labels, duplicate controls, documented permissions, and frozen training, validation, and test membership.
- Adapt verified ConvNeXt V2 Nano, EfficientNetV2-S, and Swin Transformer V2 Tiny checkpoints to the same three-class output contract.
- Train and compare the models using common experimental rules, repeated seeds, and recorded resource measurements.
- Evaluate classification, face detection, and complete-pipeline behavior with metrics that match each component.
- Build upload, camera capture, face selection, comparison, and evaluation-dashboard workflows in a responsive browser interface.
- Verify correctness, artifact reload, device behavior, privacy controls, and package reproducibility before delivery.

## Research questions

1. How do the three pretrained systems differ in macro F1, accuracy, and errors across the age groups?
2. How do their training cost, parameter count, memory use, and batch-one inference time differ on the stated hardware?
3. How much does fine-tuning contribute relative to training only the replacement classifier, and how does available training data affect performance?
4. How do controlled changes in lighting, blur, resolution, compression, and cropping affect predictions?
5. How do face detection, selection, quality rejection, and alignment affect coverage and complete-system performance?

## Significance

The project gives researchers and assessors a traceable example of transfer learning, fair model comparison, and computer vision application development. Its practical contribution is the working workflow and the evidence needed to interpret it. Its research contribution is the measured comparison under the declared conditions; superiority and real-world usefulness must be established through evaluation.

# 3 Scope and system requirements

## Scope of the system

The system processes one still image at a time. A camera preview is used to capture a still frame; continuous age prediction and video tracking are outside the core scope. When several faces are detected, the user selects one. The selected aligned face is then passed to one requested classifier or to all three.

| Class ID | Display label | Annotation mapping |
| --- | --- | --- |
| 0 | Under 18 | Provided age less than 18 |
| 1 | 18-59 | Provided age from 18 through 59 |
| 2 | 60+ | Provided age at least 60 |

The core scope includes dataset preparation, pretrained checkpoint verification, model training, comparison, face detection and alignment, the API, the browser interface, dashboard evidence, testing, and a reproducible release package. A consented local-photo pilot is conditional on participant data and permission. Hosting can be added for a controlled demonstration; local deployment satisfies the application target.

Numerical age regression, identity recognition, face matching, liveness verification, customer identification, automatic purchase approval, and automatic discount authorization are outside the core scope. A database of visitor faces and automatic collection of uploads for training are not required.

## Functional requirements

| ID | Requirement | Observable acceptance condition |
| --- | --- | --- |
| FR01 | Upload or drop a still image | A supported file opens in a preview or produces an actionable error. |
| FR02 | Detect and select a face | Boxes match the oriented image; multiple faces require explicit selection. |
| FR03 | Apply one common face pipeline | Requested models receive the same selected aligned crop. |
| FR04 | Run one or three models | Requested methods return the declared age-group vocabulary and timing fields. |
| FR05 | Draw and display outputs | The selected face is marked, its estimate is labeled, and comparison cards identify each method. |
| FR06 | Explain the default method | The recommendation refers to saved validation evidence and the selection policy. |
| FR07 | Capture a camera frame | Permission, preview, capture, retake, supported switching, and shutdown work coherently. |
| FR08 | Recover from changed input | Image, selection, and method changes clear obsolete predictions and ignore late responses. |
| FR09 | Display evaluation evidence | Tables and charts identify dataset, split, scope, denominator, and loaded model artifacts. |
| FR10 | Retry and reset | Errors allow recovery; reset removes the current photo, selection, and estimates. |

## Nonfunctional requirements

Correctness requires consistent class ordering, transforms, checkpoints, and API contracts. Reproducibility requires versioned configurations, immutable weight revisions, random seeds, data identities, and saved logs. Maintainability requires a shared model registry and separate data, training, inference, and interface components.

Usability requires a responsive layout, visible loading states, readable errors, keyboard access, text alternatives to face-box selection, and clear empty states. Privacy requires in-memory image processing, bounded request sizes, restricted public assets, and no default retention of visitor images. Operational stability requires startup readiness checks, bounded concurrent work, timeouts, and orderly resource cleanup.

## Principal user workflow

1. Open VisAge and wait for model readiness.
2. Choose a photo or explicitly start the camera and capture a frame.
3. Inspect the preview and choose a face when necessary.
4. Select a method or Compare all three and start analysis.
5. Read the overlay, age-group cards, timing information, and staff guidance.
6. Change the input, retry, or reset; open Evaluation to inspect the supporting evidence.

# 4 Model background and selection

## Transfer learning strategy

All three methods use published pretrained backbones. The ImageNet classification layer is replaced by a classifier producing three logits. The implementation must verify that the expected backbone parameters were loaded, train the new classifier, and then fine-tune the backbone according to the frozen protocol. Random initialization is permitted for isolated architecture tests but must never silently replace a missing pretrained checkpoint in an experimental run.

Using ImageNet-1k checkpoints provides a common source dataset designation. It does not make the original training methods or capacities identical. Conclusions will concern the selected pretrained systems under this project protocol rather than architecture alone.

## ConvNeXt V2 Nano

ConvNeXt V2 combines a convolutional architecture with fully convolutional masked-autoencoder pretraining and Global Response Normalization. Its design provides a modern convolutional representation for the comparison. The project will investigate whether those features transfer effectively to facial age categories. [2]

Selected checkpoint: `timm/convnextv2_nano.fcmae_ft_in1k`. Its published model card describes ImageNet-1k pretraining and fine-tuning, approximately 15.6 million parameters with its original classifier, and nominal training and test sizes of 224 and 288 pixels. These are checkpoint specifications, not project performance measurements. [3]

## EfficientNetV2 S

EfficientNetV2 uses convolutional building blocks, including Fused-MBConv, with an architecture developed around training speed and parameter efficiency. It provides a distinct convolutional design whose practical cost and classification quality can be measured in VisAge. The project does not assume that its family name guarantees the fastest local inference. [4]

Selected checkpoint: `timm/tf_efficientnetv2_s.in1k`. These are ImageNet-1k weights trained by the authors in TensorFlow and ported to PyTorch. The published checkpoint has approximately 21.5 million parameters and nominal training and test sizes of 300 and 384 pixels. [5]

## Swin Transformer V2 Tiny

Swin Transformer V2 uses a hierarchical representation and attention organized in windows. The architecture addresses training stability and resolution transfer with techniques including cosine attention and continuous position bias. It provides a Transformer-based alternative to the two CNNs. Whether it benefits this dataset must be established experimentally. [6]

Selected checkpoint: `timm/swinv2_tiny_window8_256.ms_in1k`. The model card identifies an ImageNet-1k pretrained classifier with approximately 28.3 million parameters and a fixed 256-pixel input specification. [7]

## Shared input policy

The primary comparison will use a 256 by 256 whole-face input for each model, subject to the preflight correctness and GPU pilot. This matches the selected Swin configuration and standardizes the image-size budget. It is a deliberate adaptation of the published ConvNeXt and EfficientNet input recipes. No project score will be described as a reproduction of their published ImageNet benchmark.

| Model | RGB mean | RGB standard deviation | Primary project input |
| --- | --- | --- | --- |
| ConvNeXt V2 Nano | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 | 256 by 256 |
| EfficientNetV2-S | 0.5, 0.5, 0.5 | 0.5, 0.5, 0.5 | 256 by 256 |
| Swin V2 Tiny | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 | 256 by 256 |

Each value must be checked against the pinned checkpoint configuration and saved in the trained artifact. Resize the complete canonical face with bicubic interpolation rather than introducing different center crops. The training and serving adapters must record the same canonical resize and model-input sequence. [3,5,7,9]

## Weight provenance and permissions

Record repository, immutable revision, filename, SHA-256, byte size, configuration, and license for every checkpoint. Missing or unexpected backbone keys must fail verification; only the intentionally replaced ImageNet classifier is excluded.

The selected ConvNeXt checkpoint is labeled CC BY-NC 4.0, EfficientNetV2-S Apache 2.0, and Swin V2 Tiny MIT. Preserve their notices and the dataset's research restrictions. The intended release is an academic research demonstration. Any commercial deployment requires an appropriate review of data and model permissions. [3,5,7,8]

# 5 Data preparation and governance

## Primary dataset

Use the aligned and cropped UTKFace images for age-group training. The authors describe a dataset of more than 20,000 images spanning ages from infancy to older adulthood. Age is encoded in the filename alongside other annotations. The project uses age labels for its target and must not interpret other filename attributes as verified identity. [8]

Obtain the dataset from the authors' source or a documented public mirror. For a mirror, record its owner, version, selected directory, retrieval date, archive hash, and extraction rules. A mirror hash establishes the bytes used by the project; it does not establish equivalence with an unavailable author checksum. Preserve the dataset link and permissions in the submission.

## Audit and label generation

- Parse ages and reject malformed or implausible annotations according to a recorded rule.
- Fully decode each image, apply EXIF orientation, convert to RGB, and identify corrupt files or excessive dimensions.
- Record retained and excluded sample identities with reasons rather than silently dropping files.
- Remove exact duplicates and group sufficiently similar images before splitting.
- Map ages to the three declared classes and inspect class counts and imbalance.
- Save source and pixel hashes, image dimensions, group identifiers, and split membership in an audit manifest.

Annotations represent the dataset's provided ages; they are not birth-record verification. UTKFace has no dependable subject-identity key for this study. Perceptual duplicate grouping therefore cannot justify a claim of identity-disjoint evaluation.

## Partitioning and leakage control

Use approximately 70 percent for training, 15 percent for validation, and 15 percent for final test reporting, keeping duplicate groups intact. Freeze actual membership and class counts after the audit. All models and experimental variants must use that same membership. Normalization metadata comes from the checkpoint, while class weights and fitted preprocessing decisions use training data only.

Validation data supports checkpoint selection, stopping, and the default-model decision. Test exposure must be recorded, and test results must not drive further model selection. A previously inspected partition must not be described as newly untouched confirmation data. Fresh independent confirmation requires separately collected data.

## Face detector evaluation data

Use WIDER FACE bounding-box annotations for the detector evaluation. This benchmark concerns faces in varied image conditions and is separate from the age-group dataset. Record its source, version, annotation policy, excluded invalid boxes, and evaluation split. No age labels are inferred from its boxes. [10]

## Conditional local pilot

A local pilot can assess photographs captured in the intended demonstration setting. Collect images only with appropriate consent and reliable age-group labels; where minors participate, follow the institution's guardian-consent process. Use pseudonymous participant IDs to keep a person's images together and prevent leakage between any development and evaluation portions.

Keep consent records separate from model outputs. Define who can access the photos, how long they are retained, how withdrawal is handled, and what may be submitted for assessment. Report pilot scope and sample composition. If no approved data is available, deliver the protocol and label template without claiming pilot findings.

# 6 System architecture

## Component organization

VisAge has an offline research workflow and an online inference workflow. The offline workflow produces verified model artifacts and aggregate evaluation records. The inference service loads those artifacts once and processes each visitor image in memory. The browser receives the selected-face geometry, estimated groups, timing fields, and approved aggregate evidence.

\architecture

Figure 1. The same selected face is used by all three classifiers. Training photos, model artifacts, and aggregate evaluation records have separate roles from a visitor's request.

## Browser layer

The frontend uses HTML, CSS, JavaScript modules, and SVG overlays/charts. It handles upload and drop events, local previews, camera controls, face selection, method selection, loading and cancellation, and result rendering. It also loads the evaluation dashboard. No frontend framework or database is required for the core design.

The browser maintains one current image, selected-face index, requested method set, and request generation. A new input or changed selection invalidates earlier predictions. Preview object URLs are revoked when replaced or cleared.

## Service and inference layer

FastAPI provides the HTTP contract, input admission, structured errors, readiness endpoints, and same-origin static assets. A persistent inference service owns the YuNet detector and all three trained classifiers. Shared mutable inference work is serialized with a bounded queue. New requests must not trigger model downloads or checkpoint loading. [11,12]

A model registry defines active IDs, names, architecture factories, classifier adapters, input transforms, artifact locations, and provenance. Training, CLI inference, API responses, dashboard labels, and packaging use that registry. This prevents a display name from being associated with the wrong checkpoint.

## Artifact and evidence layer

Model artifacts contain trained weights and the metadata needed to reconstruct the architecture and transforms. Evaluation records contain dataset/split identities, metrics, prediction-file hashes, model hashes, preprocessing identity, and protocol/run IDs. The dashboard may expose aggregate records, not research photos or private uploads.

The deployed inference process needs its three serving checkpoints, detector, required metadata, and selected evidence records. Training datasets and optimizer-resume states belong outside public asset directories.

# 7 Face processing and API design

## Shared face processing

Decode the admitted image, apply orientation, and detect faces with the pinned OpenCV Zoo YuNet model. YuNet provides boxes and five facial landmarks for alignment. Assign deterministic face indexes in the oriented image coordinate system and return the boxes for selection. [11]

A single acceptable detection may be selected automatically. Multiple detections require an explicit choice. Validate the selected index against the exact submitted image, estimate a similarity transform from the landmarks, and create one canonical RGB face crop. Quality checks should reject faces that are too small or severely unusable rather than return a misleading estimate.

Each requested classifier transforms that same canonical face to its saved model-input specification. Preserve the crop-policy identifier and pixel hash so direct inference and API behavior can be checked against the same prepared pixels. Document that the classifier dataset uses aligned research crops while the complete system must also handle ordinary-photo detection and alignment.

## Input and quality limits

| Setting | Design limit | Purpose |
| --- | --- | --- |
| File types | Still JPEG, PNG, WebP | Bound decoding and give predictable browser support. |
| Encoded size | Up to 8 MiB | Bound upload and request memory. |
| Decoded size | Up to 20 million pixels | Bound decoded-image memory. |
| Minimum dimensions | At least 32 pixels per side | Reject degenerate input early. |
| Detection resize | Maximum side 1280 pixels | Bound detector work while retaining ordinary-photo detail. |
| Returned faces | At most 10 | Keep selection and work bounded. |
| Minimum selected face | 64 pixels in the original image | Avoid very small faces in the operational workflow. |

These are engineering settings to verify during implementation. Brightness, contrast, blur, visibility, and landmark checks must be versioned and justified using training/development data. They are not guarantees of age reliability. Confidence thresholds used for detector benchmarking and application admission must be recorded separately.

## API endpoints

| Endpoint | Input | Response purpose |
| --- | --- | --- |
| GET / | None | Browser application. |
| GET /health | None | Readiness, loaded-model count, and device. |
| GET /v1/models | None | Available model IDs, default, age groups, and provenance. |
| POST /v1/faces | Raw image bytes | Oriented dimensions, indexed boxes, landmarks, and selection requirement. |
| POST /v1/predict | Same image bytes with method and face_index | Selected face, estimates, shared-crop metadata, and measured timing fields. |
| GET /v1/evaluation | None | Verified aggregate classification, detection, and pipeline evidence. |

Active method IDs are `convnextv2_nano`, `efficientnetv2_s`, and `swinv2_tiny`; `all` requests them in the declared comparison order. Images are sent as the raw request body with the correct image content type. The API contract must distinguish that format from multipart form data.

## Response and error contract

A prediction response identifies the request, model bundle, data freeze, oriented image dimensions, selected-face index, shared crop, and prediction records. Each prediction includes a method ID, class ID, display label, and preprocessing/inference time in milliseconds. A separate timing object accounts for decode, detection, alignment, queue, and request processing stages.

Errors provide a stable code, useful message, safe details, and request ID. Cover empty/unsupported/corrupt/oversize input, no face, ambiguous selection, invalid index, tiny or unusable face, unavailable models, queue/upload timeout, and internal failure. Internal stack traces and image bytes must not be returned to the browser.

Request cleanup must release image references after success, error, or cancellation. An abandoned native inference call must finish safely before releasing a detector/model lock used by another request.

# 8 Training and experimental methodology

## Training objective

Each model produces three raw logits. Use class-weighted cross-entropy for optimization and apply softmax only when deriving class scores for inference and ROC analysis. The prediction is the class with the largest score. Compute class weights from the current run's training subset only.

For class c, define the unnormalized weight as 1 divided by the square root of its training count. Normalize the three weights to a mean of one. Record the formula and resulting values. Use ordinary shuffled training batches rather than adding a second class-balancing sampler to the same weighted objective.

## Initialization and optimization

| Setting | Initial protocol |
| --- | --- |
| Optimizer | AdamW |
| New classifier learning rate | 0.0005 |
| Backbone learning rate | 0.0001 |
| Weight decay | 0.01 |
| Initial frozen stage | 2 head-only epochs |
| Total epoch limit | 24 |
| Stopping | Validation macro F1, patience 5 after at least 8 epochs |
| Learning-rate schedule | Cosine decay, with stage transitions recorded |
| Initial microbatch | 8 |
| Gradient accumulation | 4 steps, effective batch 32 |
| Repeated-run seeds | 42, 43, 44 |

The GPU pilot verifies numerical correctness and resource feasibility before the protocol is frozen. Train sequentially on the RTX 4050 rather than loading three training jobs concurrently. Mixed precision and gradient checkpointing may be used when verified and recorded. Gradient accumulation changes the optimization batch but does not make BatchNorm statistics identical to a physical batch of 32.

When the backbone is frozen, keep it in evaluation mode so normalization buffers and dropout behave consistently while the classifier trains. When fine-tuning begins, restore its intended trainable state. Verify both phases through parameter and buffer checks.

## Augmentation and checkpointing

Use the same bounded geometric and appearance augmentation policy for all three models: horizontal flip, small rotation, and mild brightness/contrast/saturation changes. Validation and test transforms are deterministic. Avoid augmentations that substantially remove the face or alter the class label assumptions. Record exact ranges and interpolation behavior in the training configuration.

Save best-validation checkpoints separately from resume checkpoints. A resumable run stores model, optimizer, scheduler, precision-scaler and random-generator state, epoch/stage, and configuration identity. Record package versions, source hashes, dataset manifest, pretrained revision, device, duration, and peak memory. Save actual training and validation curves without presenting a partially completed job as finished.

## Five planned experiment categories

| Category | Design | Evidence to produce |
| --- | --- | --- |
| Primary comparison | Three models, full training data, three seeds | Per-seed metrics, mean and standard deviation, curves. |
| Data efficiency | Common nested 25 and 50 percent whole-group training subsets | Quality versus available training data. |
| Transfer learning ablation | Frozen-backbone classifier versus full fine-tuning | Contribution of backbone adaptation. |
| Robustness | Fixed brightness, blur, compression, resolution, and crop changes | Sensitivity by condition and age group. |
| Operational evaluation | Batch-one inference and complete API pipeline | Latency, memory, parameter count, coverage, failure accounting. |

The training allocation is nine full-data fine-tuning runs, three frozen-backbone baselines at seed42, and six data-subset runs at seed42: 18 planned training jobs. Frozen-backbone baselines receive their own training budget and stopping rule; the warm-up stage alone is not a complete ablation baseline. Pilot steps and repeated measurements are recorded separately.

## Default model selection

Rank families by mean validation macro F1 across the three seeds. Treat means within 0.005 of the highest mean as near-tied; resolve the tie using lower under-18-to-older error, then higher 60+ recall, then measured batch-one serving latency. Freeze numerical tie and rounding rules in the protocol. Use the seed42 best-validation checkpoint for each family's serving artifact rather than selecting a favorable seed after seeing outcomes.

Test metrics are for final reporting and do not change the selected default. Do not search indefinitely for a target score or remove a model because it underperforms. Record any protocol amendment, its timing, and the reason.

# 9 Evaluation framework

## Classification quality

| Metric | Definition and reporting requirement |
| --- | --- |
| Top-one accuracy | Correct age-group predictions divided by evaluated labeled inputs, reported as a percentage. |
| Per-class precision | True positives divided by predicted positives for that age group. |
| Per-class recall | True positives divided by actual positives for that age group. |
| Per-class F1 | Harmonic mean of precision and recall; define zero-denominator handling. |
| Macro F1 | Unweighted mean of the three class F1 values. |
| Confusion matrix | Actual groups as rows and predicted groups as columns; include counts and row-normalized percentages. |
| ROC and AUC | Each group against the other two using continuous model scores; retain the full curve for numerical AUC. |
| Boundary errors | True under-18 predicted older, and true 60+ predicted below 60, each divided by its actual group size. |

Report per-class AUC and the arithmetic mean of the three class AUCs. A sampled display curve must not change the underlying numerical calculation. If a subset lacks a required positive or negative class, mark the corresponding ROC as undefined rather than manufacture a value.

Softmax scores are uncalibrated model scores unless a separate calibration study is performed. ROC ranking quality is not a person's age certainty. In this project, accuracy concerns age-group classification; it is not identity-recognition accuracy.

## Detector quality

Evaluate YuNet boxes against independent WIDER FACE annotations using precision-recall curves and AP at IoU thresholds 0.50 and 0.75. Define intersection-over-union, confidence sweep, one-to-one matching, duplicate detections, invalid/ignored annotations, and included images before evaluation. State whether the protocol is an all-valid-box study or the official difficulty subsets; do not use the names interchangeably. [10,11]

## Complete pipeline quality

Keep every attempted input in the accounting. Coverage equals accepted inputs divided by attempted inputs. Accepted-only accuracy and F1 describe the subset that reaches classification. The all-attempted correct rate equals correct predictions divided by every attempted input, including rejections. Report rejection counts and reasons with those denominators.

Keep aligned-crop classification, re-detection of research crops, ordinary-photo tests, and a consented local pilot as separately labeled evaluation scopes. Performance in one scope does not establish performance in another.

## Efficiency and analysis

Measure runtime per image in milliseconds with CPU or GPU identified; report batch size, warm-up, preprocessing scope, hardware, and repetition count. Record parameter count in millions and training time in minutes. Include checkpoint size, peak training memory, serving memory, and API median/p95 where practical. Synchronize CUDA measurements appropriately and separate model work from upload, queue, and complete request timing.

Save prediction CSVs and machine-readable metric JSON alongside labeled plots. An independent checker must reconstruct confusion counts, accuracy, F1, and curve inputs from the saved predictions and verify sample/model identities. Analysis should explain errors near category boundaries, imbalance, image quality, and resource tradeoffs without inferring unsupported demographic or causal conclusions.

# 10 Environment setup and project organization

## Development environment

The development target is Windows with Python 3.12 and an NVIDIA RTX 4050 Laptop GPU with about 6 GB VRAM. Use one virtual environment and an explicit dependency lock. Reserve disk capacity for the raw dataset, caches, pretrained files, best/resume checkpoints, and reports; confirm the budget during preflight. A CPU inference path is required for the local demonstration.

| Layer | Technology |
| --- | --- |
| Model training and inference | Python, PyTorch, torchvision, timm, safetensors |
| Image and numeric processing | OpenCV, Pillow, NumPy, SciPy |
| API and local serving | FastAPI and Uvicorn |
| Browser application | HTML, CSS, JavaScript modules, SVG, browser camera API |
| Analysis and figures | pandas where needed, Matplotlib, CSV and JSON artifacts |
| Verification | pytest, HTTP client tests, JavaScript test runner, browser integration checks |
| Reproduction | Jupyter notebook, configuration files, dependency lock, artifact hashes |

Node.js is used for JavaScript tests; it is not required to serve the static frontend. A database is not part of the core architecture.

## Installation procedure

Create the environment from the project root in PowerShell. The following framework pair is the selected reproducibility target, rather than an instruction to install an unspecified latest release. The official PyTorch installation page documents this CUDA wheel combination. [13]

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 `
    --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install timm==1.0.31
.\.venv\Scripts\python.exe -m pip check
```

Verify Python and package versions, CUDA availability, detected GPU memory, and a small GPU forward/backward operation. Save a resolved dependency lock for the intended platform. A compatible NVIDIA driver is required; do not treat successful package installation as proof that GPU execution works.

## Preparation and local startup

Prepare and audit the dataset, freeze its manifest, obtain verified model and detector files, run the GPU pilot, and complete the declared training/evaluation stages. The delivered README must provide exact entry points and configurations for those stages. Training commands must select an explicit model/run configuration and must not silently download unpinned weights.

After the required serving artifacts and evidence records are present, launch the application:

```powershell
.\.venv\Scripts\python.exe scripts/serve_api.py --host 127.0.0.1 --port 8000 --device cpu
```

Open `http://127.0.0.1:8000/` and check readiness before inference. CUDA inference is an explicit device option when supported. Camera access requires localhost or a secure HTTPS context; ordinary HTTP on a network IP address may not expose camera access. [14]

## Repository structure

```text
VisAge/
  src/visage/       data, registry, training, inference, API
    methods/       one model-specific module per backbone
    web/           HTML, CSS, JavaScript and SVG assets
  scripts/         preparation, training, evaluation and serving
  configs/         protocol, model, data and deployment settings
  notebooks/       reproducible experiment notebook
  tests/           unit, integration and browser tests
  data/            provenance, manifests and private local data
  artifacts/       trained weights and artifact metadata
  reports/         logs, predictions, metrics and figures
  docs/            project and user documentation
  output/          final deliverables and verified package
```

Keep raw images, caches, environments, and optimizer states outside public assets and normal source control. Provide model-specific modules for ConvNeXt V2 Nano, EfficientNetV2-S, and Swin V2 Tiny even when common training code is shared. Registry entries and saved transforms must make each method independently reproducible.

# 11 Interface and user experience

## Analysis screen

Use a clear input area and a separate results area, with responsive stacking on phones. Provide upload and camera tabs, a visible file chooser, drag-and-drop feedback, preview, method selection, and a primary analysis action. Explain file limits before submission. Do not prefill the results area with example estimates that look like live outputs.

Draw detected face boxes over the correctly oriented preview. Each box has a matching text button with an accessible name. A selected face uses a distinct visual state. The output overlay identifies the selected face and the requested estimate; comparison cards show the three model names, categories, and measured times without covering important image content.

## Camera workflow

Camera permission starts only after an explicit action. Request video without audio, show a live preview, and enable capture when dimensions are available. A captured frame enters the same processing path as an uploaded image. Retake clears the prior image and predictions. Switching is offered only when devices can be enumerated and more than one is available.

Stop tracks after capture, explicit stop, source/route changes, cancellation, hidden-page events, and page exit. Close a stream that arrives after a cancelled permission request. Explain denied permission, missing/busy devices, unsupported contexts, and capture failure, while retaining an upload alternative. [14]

## Results and evaluation screens

Individual result cards identify the model, age group, and timing scope. Compare all three preserves a consistent method order and indicates disagreement without implying a verified age. The default badge explains the validation-based selection rule. User guidance directs staff to the identification process rather than providing transaction approval.

The dashboard separates age-group classification, face detection, and complete-pipeline evidence. It displays dataset and split labels, sample counts, metric definitions, confidence operating points where relevant, and denominator scopes. It must verify the active artifact identity before presenting a score.

## Accessibility and state management

Support keyboard navigation, visible focus, readable contrast, reduced motion, screen-reader status announcements, and touch-friendly controls. Tables may scroll inside their own region on narrow screens. Error messages should explain the next useful action.

Changing the image, face, or method clears obsolete estimates. Request IDs and generation guards prevent an old response from overwriting newer input. Reset releases preview references and input state. If the server disconnects after readiness, show a retry action and prevent an unavailable model from appearing ready.

# 12 Verification and acceptance

## Verification layers

| Layer | Required checks |
| --- | --- |
| Data | Label mapping, corrupt-file rejection, duplicate grouping, split identity, training-only fitted decisions. |
| Models | Pretrained key/shape matching, normalization, output ordering, finite logits/loss, freeze/unfreeze behavior. |
| Training | Fixed seeds/configuration, correct stopping and schedules, checkpoint/resume integrity, actual curves. |
| Evaluation | Independent metric arithmetic, ROC inputs, detector matching policy, complete rejection accounting. |
| Inference | Reload consistency, CPU/CUDA comparisons within stated tolerances, identical prepared crop across models. |
| API | Raw-body contract, malformed input, limits, queue/cancellation safety, useful errors, readiness and cleanup. |
| Browser | Upload/drop, overlays, selection, single/all methods, stale-response clearing, retry/reset, responsive layout. |
| Camera | Permission errors, preview, capture, retake, supported switching, late stream, track shutdown. |
| Packaging | Complete artifacts, exact hashes, isolated extraction/startup, page/dashboard/prediction verification. |

Use synthetic inputs and small fixed samples for correctness tests. Use declared dataset partitions for experimental measurements. These activities serve different purposes and must not be mixed into an accuracy claim.

## Browser and hardware acceptance

Run browser integration against the actual API and trained artifacts. A simulated video stream can verify capture and cleanup without using a physical webcam, but it does not establish hardware compatibility. Record separate laptop-webcam and physical-phone checks, including browser/context, permissions, orientation, front/rear switching, page visibility, and cleanup behavior.

Do not claim all hardware tests passed when only simulated input was used. Any local pilot has its own consent and labeling requirements. A missing pilot dataset produces a pending pilot protocol rather than invented observations.

## Release acceptance criteria

- All three registered methods load verified serving artifacts and produce the required output type through the application.
- Direct prepared-crop inference, image CLI, and real HTTP requests agree on the selected face and documented numerical tolerances.
- No stale estimate survives an image, face, or method change.
- Invalid, ambiguous, and unusable input receives the declared recovery behavior.
- The dashboard is bound to the correct model and data records and labels each evaluation scope.
- Automated checks complete without unresolved failures; manual tests are recorded as actually performed.
- The package starts from a separate extracted directory using the documented environment and required bundled artifacts.
- Known limitations, conditional work, licenses, and reproducibility instructions accompany the release.

# 13 Deployment privacy and project risks

## Local and hosted operation

The primary deployment is one persistent Uvicorn process serving FastAPI and the static browser assets on localhost. A single worker avoids duplicate model memory and preserves the shared inference lock. Bind to loopback for local demonstrations and verify startup health and orderly shutdown.

A hosted demonstration requires HTTPS, an explicit exposure/configuration review, a compatible memory budget, request limits, process supervision, health monitoring, and restricted logs. Static files must not expose model artifacts, datasets, consent records, or internal source directories. Camera support depends on a secure browser context. [12,14]

## Privacy and responsible operation

Process visitor photos and aligned crops in memory. Do not retain them, log their contents, or add them to training by default. Send no image data to third-party services in the core design. Use no-store responses, same-origin restrictions, safe errors, and an explicit browser camera policy.

Research datasets, consented pilot photos, and transient visitor input have different permissions and retention rules. Keep those workflows separate. Display an estimation notice and retain human responsibility for age-related decisions. Do not interpret a model category as identity, verified chronological age, or automatic eligibility.

## Risks and planned responses

| Risk | Planned response |
| --- | --- |
| Limited GPU memory | Pilot microbatches, accumulation and precision; train sequentially and preserve resume state. |
| Imbalanced or uncertain labels | Audit annotations, train-only weighting, per-group metrics and transparent limitations. |
| Near-duplicate or subject leakage | Remove duplicates, group similar images, freeze membership, qualify identity-related claims. |
| Ordinary-photo domain difference | Shared alignment, quality checks, complete-pipeline accounting and conditional local pilot. |
| Incorrect pretrained transform | Pin configuration and test canonical resize, normalization and output order. |
| Late browser responses or leaked streams | Generation/cancellation guards, track cleanup, and browser integration tests. |
| Incomplete delivery package | Hash manifest and isolated extraction/startup test. |
| Unclear data or weight permissions | Retain source/license evidence and restrict the release to permitted academic use. |

# 14 Implementation plan and supervision

## Development phases

| Phase | Main work | Exit deliverable |
| --- | --- | --- |
| 1 Specification | Define task, requirements, model selection, protocol and acceptance criteria. | Project specification and approved proposal. |
| 2 Environment and weights | Lock dependencies, verify checkpoints, test GPU memory/speed and numerical behavior. | Environment and provenance records with pilot settings. |
| 3 Data preparation | Audit, generate labels, group duplicates, freeze splits, prepare versioned cache. | Audited data manifest and preprocessing configuration. |
| 4 Model implementation | Shared registry, three method modules, transfer learning, artifact reload tests. | Tested training/inference modules. |
| 5 Experiments | Complete primary, subset, frozen-backbone, robustness and efficiency studies. | Run records, predictions, curves and comparison evidence. |
| 6 Selection and evidence | Verify metrics, choose default from validation, produce final reporting records. | Selected bundle and verified evaluation artifacts. |
| 7 Application | Integrate face pipeline, API, upload/camera interface and dashboard. | Complete local web application. |
| 8 Acceptance and delivery | Test, fix, document, package and verify isolated startup. | Verified release and academic submission. |

Phases have dependency-based completion criteria rather than an assumed calendar deadline. Provide instructor check-ins at the agreed milestones. The GPU pilot supplies runtime estimates; checkpointing supports interruption and resumption. Freeze the experimental protocol before full runs and record justified amendments.

## Team responsibilities

Allocate responsibility for data/provenance, model/training work, backend/face processing, browser integration, evaluation/reproducibility, and documentation/presentation. Members may share roles, but the final contribution statement must describe actual work. Every presenter should be able to explain the model-loading, preprocessing, training, and evaluation code they claim responsibility for.

Record the use of AI coding assistance in the contribution statement, including the tasks supported and the checks performed by the group. Review generated code and citations rather than treating assistance as evidence of correctness. [1]

## Change and evidence management

Use versioned configuration files and immutable artifact identities. A run record must identify its data, source revision, initialization, settings, seed, device, and completion state. Changes that affect the comparison require an explicit protocol amendment before further measurement. Reports must distinguish completed experiments from planned or interrupted work.

# 15 Required deliverables and submission

## Academic deliverables

The submission combines the working system with the evidence needed to assess the comparison. The course brief specifies code/notebook, dataset access, PDF reporting, slides, and a contribution statement. The method-specific materials in this specification concern the three declared pretrained architectures. [1]

| Deliverable | Required contents | Completion evidence |
| --- | --- | --- |
| One-page proposal | Task, three models, dataset, local-data plan, risks, group details. | Instructor-approved task and method specification. |
| Executed Jupyter notebook | Environment, data/provenance checks, training/evaluation workflow and visible outputs. | Actual top-to-bottom execution in the documented environment. |
| Three method modules | Separate modules for ConvNeXt V2 Nano, EfficientNetV2-S and Swin V2 Tiny, with shared utilities as needed. | Import/reload tests and reproducible method configuration. |
| Source and configuration | Data preparation, training, inference, API, frontend, tests and locked settings. | Runnable commands and passing release checks. |
| Model and detector artifacts | Serving weights, initialization provenance, transform metadata, licenses and hashes. | Verified loading and independent package startup. |
| Data access information | Author/mirror links, permissions, manifests, labels and exclusions. | Traceable dataset identities; local data only when approved and used. |
| Experimental evidence | Curves, prediction files, metrics, efficiency measurements and analysis. | Consistent same-partition comparison with labeled units. |
| Working application | Upload/capture, selected-face outputs, three-model comparison, default explanation and dashboard. | Demonstration on new permitted input. |
| Final PDF report | Introduction, Methods, Experiments, Results, Discussion, limitations and references. | Claims supported by completed evidence. |
| Presentation slides | A 10-minute explanation of the problem, methods, experiments, system and conclusions. | Timed rehearsal and preparation for 5 minutes of questions. |
| Contribution statement | One paragraph describing each member's actual work and disclosed AI assistance. | Member-reviewed and consistent with project records. |

A short demonstration recording of approximately three minutes is recommended as a supporting artifact. It should show the three methods on the same permitted input and include a useful error/recovery example. A local pilot dataset and its findings are conditional deliverables when approved data is used. [1]

## Final research report structure

The Introduction establishes the problem, motivation, objectives, scope, and research questions. Methods describes the dataset, splits, pretrained checkpoints, transforms, classifier adaptation, training and system architecture. Experiments defines the run matrix, baselines, metrics, efficiency measurements and controls.

Results is written after evaluation and contains actual measurements, labeled curves, confusion matrices, ROC/AUC, detector evidence and complete-pipeline accounting. Discussion explains error patterns and tradeoffs and answers the research questions. Limitations addresses annotation quality, data leakage uncertainty, domain shift, device/pilot scope and decision-support boundaries. References identify the original papers, dataset sources, checkpoints and software documentation.

## Submission package and handoff

Prepare one archive named `GroupNo_CVProject.zip`, replacing GroupNo with the actual group identifier. Include the notebook, method modules, source/configuration, README, permitted artifacts, data links/manifests, PDF report, slides, contribution statement, test/evaluation records, and a file-hash manifest. Include raw local photos only through an approved channel and only when consent permits submission. Exclude visitor images, environments, download archives and unnecessary caches. [1]

The README must explain environment installation, data acquisition, trained-artifact placement, training/evaluation entry points, application startup, test commands, and known limitations. Verify the package by extracting it into a separate directory, starting the service, opening the page/dashboard, and submitting a permitted image to all three methods. Record the package hash and shutdown of owned test processes.

Completion means the three trained pretrained classifiers are functional in the application and the required evidence and submission materials can be reproduced or inspected. Physical-device testing, a conditional local pilot, and public hosting must each be described according to their actual completion state.

# 16 References

[1] C. Dofitas. Computer Vision Comparison Project. West Visayas State University, course project brief, October 26, 2026. Supplied course document.

[2] S. Woo et al. ConvNeXt V2 Co-designing and Scaling ConvNets with Masked Autoencoders. 2023. https://arxiv.org/abs/2301.00808

[3] timm. ConvNeXt V2 Nano checkpoint and model card. https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in1k

[4] M. Tan and Q. V. Le. EfficientNetV2 Smaller Models and Faster Training. ICML, 2021. https://arxiv.org/abs/2104.00298

[5] timm. TensorFlow EfficientNetV2-S ImageNet-1k checkpoint and model card. https://huggingface.co/timm/tf_efficientnetv2_s.in1k

[6] Z. Liu et al. Swin Transformer V2 Scaling Up Capacity and Resolution. CVPR, 2022. https://arxiv.org/abs/2111.09883

[7] timm. Swin V2 Tiny window8 256 ImageNet-1k checkpoint and model card. https://huggingface.co/timm/swinv2_tiny_window8_256.ms_in1k

[8] UTKFace authors. UTKFace Large Scale Face Dataset. https://susanqq.github.io/UTKFace/

[9] Hugging Face timm documentation. Data configuration and transforms. https://huggingface.co/docs/timm/main/en/reference/data

[10] S. Yang, P. Luo, C. C. Loy, and X. Tang. WIDER FACE A Face Detection Benchmark. 2016. https://arxiv.org/abs/1511.06523

[11] OpenCV Zoo. YuNet face detection model and demonstration. https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

[12] FastAPI documentation. Deployment. https://fastapi.tiangolo.com/deployment/

[13] PyTorch documentation. Previous version installation commands including PyTorch 2.11.0 and torchvision 0.26.0. https://pytorch.org/get-started/previous-versions/

[14] MDN Web Docs. MediaDevices getUserMedia. https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia

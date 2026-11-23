import { AnalysisSession, CameraController, RequestGate, AppError, api, explainError, METHODS, GROUPS, MODEL_INFO, formatTime, scoreBreakdown, formatModelScore, orderedMethods } from "./core.js";
import { element as el, icon } from "./dom.js";
import { EvaluationDashboard } from "./evaluation.js";

const $ = id => document.getElementById(id);
let source = "upload", route = "analyze", defaultMethod = METHODS[0], lastFaces = null, lastSelected = null, lastResult = null;
let methodChosen = false;
let evaluationLoaded = false, evaluationTab = "classification";
const connectionGate = new RequestGate(), evaluationGate = new RequestGate(), dashboard = new EvaluationDashboard();
const session = new AnalysisSession({ notify: renderAnalysis });
const camera = new CameraController({ video: $("camera-preview"), notify: () => renderAnalysis(session.state) });
const announce = text => { $("announcer").textContent = text; };
function setText(id, text) { $(id).textContent = text; }
function showError(value) {
  $("input-error").hidden = !value;
  if (value) { setText("input-error-title", value.title); setText("input-error-message", value.message); }
}
function renderAnalysis(state) {
  if (state.error?.code === "OFFLINE") { $("service-status").dataset.state = "offline"; setText("service-status-text", "Server unavailable"); $("reconnect").hidden = false; }
  const live = camera.state.phase === "live", starting = camera.state.phase === "starting";
  $("upload-empty").hidden = source !== "upload" || Boolean(state.preview);
  $("camera-empty").hidden = source !== "camera" || Boolean(state.preview) || live;
  $("photo-stage").hidden = !state.preview;
  $("video-stage").hidden = !live;
  $("camera-controls").hidden = !live;
  $("switch-camera").disabled = camera.state.devices.length < 2;
  $("camera-preview").classList.toggle("mirrored", camera.state.facing === "user");
  $("input-busy").hidden = !(state.busy === "detecting" || starting);
  setText("input-busy-text", starting ? "Waiting for the camera…" : "Finding faces…");
  $("cancel-request").hidden = false;
  $("method-select").disabled = state.busy === "detecting";
  $("method-select").value = state.method;
  $("analyze-photo").disabled = !state.ready || !state.preview || state.selected === null || Boolean(state.busy) || live || starting;
  setText("analyze-button-text", state.busy === "predicting" ? "Analyzing…" : state.method === "all" ? "Compare all three methods" : "Analyze selected face");
  const preview = $("photo-preview");
  if (state.preview && preview.getAttribute("src") !== state.preview) preview.src = state.preview;
  if (!state.preview) preview.removeAttribute("src");
  $("photo-details").hidden = !state.preview;
  if (state.preview) {
    setText("file-name", state.name);
    const size = state.blob.size >= 1024 ** 2 ? `${(state.blob.size / 1024 ** 2).toFixed(1)} MiB` : `${(state.blob.size / 1024).toFixed(1)} KiB`;
    setText("file-meta", [size, state.image ? `${state.image.width} × ${state.image.height} px` : "Still image"].join(" · "));
    setText("preview-label", !state.faces.length ? "Photo preview" : state.selected === null ? "Choose a face" : `Face ${state.selected + 1} selected`);
    setText("replace-photo", source === "camera" ? "Retake photo" : "Change photo");
  }
  showError(state.error || (source === "camera" && camera.state.phase === "error" ? camera.state.error : null));
  $("face-selection").hidden = !state.faces.length;
  if (lastFaces !== state.faces || lastSelected !== state.selected) {
    const restore = document.activeElement?.dataset.faceIndex;
    const buttons = state.faces.map(face => {
      const node = el("button", "face-chip"); node.type = "button"; node.dataset.faceIndex = String(face.index);
      node.setAttribute("aria-pressed", String(face.index === state.selected)); node.setAttribute("aria-label", `Select face ${face.index + 1}`);
      node.append(icon(face.index === state.selected ? "check" : "scan"), document.createTextNode(`Face ${face.index + 1}`));
      node.addEventListener("click", () => { session.selectFace(face.index); announce(`Face ${face.index + 1} selected. Previous results cleared.`); }); return node;
    });
    $("face-buttons").replaceChildren(...buttons);
    setText("face-count", `${state.faces.length} ${state.faces.length === 1 ? "face" : "faces"} detected`);
    setText("selection-help", state.faces.length > 1 && state.selected === null ? "Choose one to continue." : "All methods use the same selected face.");
    if (restore !== undefined) buttons.find(node => node.dataset.faceIndex === restore)?.focus({ preventScroll: true });
    lastFaces = state.faces; lastSelected = state.selected; renderOverlay();
  }
  const predicting = state.busy === "predicting";
  $("results-empty").hidden = predicting || Boolean(state.result);
  $("results-loading").hidden = !predicting;
  $("prediction-results").hidden = !state.result;
  setText("prediction-progress", state.method === "all" ? "Comparing all three methods…" : `Running ${MODEL_INFO[state.method].name}…`);
  if (lastResult !== state.result) { lastResult = state.result; renderResults(state); renderOverlay(); }
}
function renderOverlay() {
  const state = session.state, svg = $("face-overlay"); svg.replaceChildren();
  if (!state.image || !state.preview) return;
  const { width, height } = state.image;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`); svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const stage = $("photo-stage"), displayedWidth = Math.min(stage.clientWidth || width, (stage.clientHeight || height) * width / height);
  const textSize = 12 * width / Math.max(1, displayedWidth), labelHeight = textSize * 1.9;
  const create = (tag, attrs, text) => { const node = document.createElementNS(svg.namespaceURI, tag); Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value))); if (text) node.textContent = text; return node; };
  state.faces.forEach(face => {
    const [x, y, w, h] = face.bbox, selected = face.index === state.selected;
    const group = create("g", { class: `face-box${selected ? " selected" : ""}`, role: "button", tabindex: 0, "aria-label": `Select face ${face.index + 1}`, "aria-pressed": selected, "data-face-index": face.index });
    group.append(create("rect", { x, y, width: w, height: h, rx: textSize * .35, class: "box-rect" }));
    const result = selected && state.result ? (state.result.predictions.find(value => value.method === defaultMethod) || state.result.predictions[0]) : null;
    const differs = selected && state.result && new Set(state.result.predictions.map(value => value.class_id)).size > 1;
    const label = result ? `Face ${face.index + 1} · ${GROUPS[result.class_id]}${differs ? " · differs" : ""}` : `Face ${face.index + 1}`;
    const labelWidth = Math.min(width, textSize * (label.length * .62 + 1.2));
    const labelY = Math.max(0, y - labelHeight), labelX = Math.max(0, Math.min(x, width - labelWidth));
    group.append(create("rect", { x: labelX, y: labelY, width: labelWidth, height: labelHeight, rx: textSize * .3, class: "box-tag" }));
    group.append(create("text", { x: labelX + textSize * .6, y: labelY + textSize * 1.27, "font-size": textSize, class: "box-label" }, label));
    const select = () => { session.selectFace(face.index); announce(`Face ${face.index + 1} selected. Previous results cleared.`); };
    group.addEventListener("click", select);
    group.addEventListener("keydown", event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); select(); } }); svg.append(group);
  });
}
function renderScores(value) {
  const breakdown = scoreBreakdown(value), section = el("section", "model-scores");
  section.setAttribute("aria-label", `${MODEL_INFO[value.method].name} age-group model scores`);
  section.append(el("h4", "score-heading", "Model scores"));
  breakdown.classes.forEach(entry => {
    const selected = entry.class_id === value.class_id, row = el("div", `score-row${selected ? " score-selected" : ""}`);
    row.dataset.classId = entry.class_id;
    const label = el("div", "score-label"), name = el("span", "", GROUPS[entry.class_id]);
    label.append(name, el("span", "score-percent", formatModelScore(entry.score)));
    if (selected) name.append(el("span", "score-choice", "Chosen"));
    const meter = el("meter", "score-meter"); meter.min = 0; meter.max = 1; meter.value = entry.score;
    meter.setAttribute("aria-label", `${GROUPS[entry.class_id]}: ${formatModelScore(entry.score)} model score${selected ? ", chosen group" : ""}`);
    row.append(label, meter); section.append(row);
  });
  return section;
}
function renderResults(state) {
  $("model-results").replaceChildren(); $("result-context").replaceChildren(); $("staff-guidance").replaceChildren(); $("request-timing").replaceChildren();
  if (!state.result) { setText("results-subtitle", "A shared face. A fair comparison."); return; }
  const result = state.result, differs = new Set(result.predictions.map(value => value.class_id)).size > 1;
  setText("results-subtitle", `Face ${state.selected + 1} · ${result.predictions.length === 3 ? "Three-method comparison" : "Selected method"}`);
  $("result-context").append(el("span", "", "Estimated age group"), el("span", `context-badge${differs ? " disagreement" : ""}`, differs ? "Methods differ" : "Shared face crop"));
  const displayOrder = orderedMethods(defaultMethod);
  [...result.predictions].sort((a, b) => displayOrder.indexOf(a.method) - displayOrder.indexOf(b.method)).forEach(value => {
    const meta = MODEL_INFO[value.method], card = el("article", `model-result${value.method === defaultMethod ? " recommended" : ""}`), head = el("div", "model-result-head"), mark = el("span", `model-mini ${meta.tone}`), words = el("div");
    card.dataset.method = value.method;
    mark.append(icon(meta.icon)); words.append(el("h3", "", meta.name), el("p", "model-description", meta.kind)); head.append(mark, words);
    if (value.method === defaultMethod) head.append(el("span", "recommended-pill", "Recommended"));
    const body = el("div", "model-result-body"), age = el("div", "age-value", GROUPS[value.class_id]), timing = el("span", "model-time"); age.append(el("span", "", "years")); timing.append(icon("clock"), document.createTextNode(formatTime(value.inference_and_preprocessing_ms))); body.append(age, timing); card.append(head, body, renderScores(value)); $("model-results").append(card);
  });
  $("request-timing").append(el("span", "", "Method timings include preprocessing"), el("span", "", `Server: ${formatTime(result.timings_ms?.request_total ?? result.timings_ms?.processing_total)}`));
  $("compare-all").hidden = result.predictions.length === 3;
  const guidance = ["The selected method estimates Under 18. Staff should verify age using the required identification before an age-restricted purchase.", "The selected method estimates 18–59. Follow the applicable ID-check policy; the estimate does not establish purchase or discount eligibility.", "The selected method estimates 60+. Staff should confirm age and senior-discount eligibility with the required identification."];
  const main = result.predictions.find(value => value.method === defaultMethod) || result.predictions[0];
  $("staff-guidance").append(el("strong", "", "Staff guidance"), el("p", "", differs ? "The methods estimate different groups. Review the estimates and use valid identification to confirm age or eligibility." : guidance[main.class_id]));
  announce(`${result.predictions.map(value => `${MODEL_INFO[value.method].name}: ${GROUPS[value.class_id]}`).join(". ")}. Estimates support staff review.`);
  if (route === "analyze" && matchMedia("(max-width:1020px)").matches) $("results-title").closest("section").scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
}
function setSource(value) {
  if (!["upload", "camera"].includes(value) || source === value) return;
  camera.stop(); session.clear(); $("photo-input").value = ""; source = value;
  document.querySelectorAll("[data-source]").forEach(node => { const active = node.dataset.source === source; node.setAttribute("aria-selected", String(active)); node.tabIndex = active ? 0 : -1; });
  $("source-panel").setAttribute("aria-labelledby", `tab-${source}`); renderAnalysis(session.state);
}
async function chooseFile(file) {
  if (!file) return;
  if (source !== "upload") setSource("upload"); camera.stop();
  await session.setInput(file, file.name || "Selected photo");
  if (session.state.error) announce(session.state.error.title);
  else if (session.state.faces.length) announce(`${session.state.faces.length} ${session.state.faces.length === 1 ? "face detected and selected" : "faces detected. Choose one to continue"}.`);
}
function beginCamera() { session.clear(); $("photo-input").value = ""; announce("Starting camera. Previous photo and results cleared."); void camera.start(); }
async function capturePhoto() {
  $("capture-photo").disabled = true;
  try { const blob = await camera.capture(); if (blob) { await session.setInput(blob, "Camera capture.jpg"); announce(session.state.error?.title || "Photo captured. Select a face and analyze."); } }
  catch (error) { session.showError(error instanceof AppError ? error : new AppError("CAMERA_CAPTURE")); }
  finally { $("capture-photo").disabled = false; }
}
function keyboardTabs(selector, action) {
  const tabs = [...document.querySelectorAll(selector)];
  tabs.forEach((tab, i) => tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowRight") next = (i + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (i + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0; if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); action(tabs[next]); tabs[next].focus(); }
  }));
}
function setEvaluationTab(value) {
  evaluationTab = value;
  document.querySelectorAll("[data-eval]").forEach(node => { const active = node.dataset.eval === value; node.setAttribute("aria-selected", String(active)); node.tabIndex = active ? 0 : -1; });
  ["classification", "detection", "pipeline"].forEach(name => { $(`eval-${name}`).hidden = name !== value; });
}
async function loadEvaluation(force = false) {
  if (evaluationLoaded && !force) return;
  const job = evaluationGate.start(); $("evaluation-status").hidden = false; $("evaluation-error").hidden = true; $("evaluation-content").hidden = true; $("refresh-evaluation").disabled = true;
  try {
    const data = await api.evaluation(job.signal);
    if (!evaluationGate.current(job)) return;
    if (!data.classification || !data.detection || !data.face_pipeline) throw new AppError("BAD_RESPONSE");
    $("evaluation-content").hidden = false; dashboard.render(data, defaultMethod); setEvaluationTab(evaluationTab); evaluationLoaded = true;
  } catch (error) {
    if (!evaluationGate.current(job)) return;
    const problem = explainError(error); $("evaluation-error").replaceChildren(icon("info")); const words = el("div"); words.append(el("strong", "", problem.title), el("p", "", problem.message)); $("evaluation-error").append(words); $("evaluation-error").hidden = false;
  } finally {
    if (evaluationGate.current(job)) { $("evaluation-status").hidden = true; $("refresh-evaluation").disabled = false; }
  }
}
function showRoute() {
  const requested = location.hash.replace(/^#\/?/, "") || "analyze";
  route = ["analyze", "evaluation", "about"].includes(requested) ? requested : "analyze";
  if (route !== "analyze") camera.stop();
  document.querySelectorAll(".view").forEach(node => { node.hidden = node.id !== `view-${route}`; });
  document.querySelectorAll("[data-route]").forEach(node => { if (node.dataset.route === route) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current"); });
  setText("route-label", { analyze: "Face analysis", evaluation: "Evaluation", about: "How it works" }[route]);
  document.title = `${{ analyze: "Analyze a face", evaluation: "Evaluation", about: "How it works" }[route]} — VisAge`;
  if (route === "evaluation") void loadEvaluation();
  if (route === "analyze") renderAnalysis(session.state);
  window.scrollTo({ top: 0, behavior: "instant" });
}
async function connect() {
  const job = connectionGate.start(); $("service-status").dataset.state = "connecting"; setText("service-status-text", "Connecting to models"); $("reconnect").hidden = true;
  try {
    const [health, models] = await Promise.all([api.health(), api.models()]);
    if (!connectionGate.current(job)) return;
    if (health.status !== "ready" || !METHODS.every(method => models.methods.some(value => value.id === method))) throw new AppError("NOT_READY");
    defaultMethod = METHODS.includes(models.default_method) ? models.default_method : METHODS[0];
    if (Number.isInteger(models.input_limits?.max_bytes) && models.input_limits.max_bytes > 0) {
      session.maxUploadBytes = models.input_limits.max_bytes;
      document.querySelector('#upload-empty .file-hint').textContent = `JPEG, PNG, WebP, HEIC/HEIF, AVIF, BMP, TIFF · Up to ${models.input_limits.max_bytes / 1024**2} MiB`;
    }
    if (!methodChosen) session.setMethod(defaultMethod);
    METHODS.forEach(method => { const option = [...$("method-select").options].find(value => value.value === method); if (option) option.textContent = MODEL_INFO[method].name + (method === defaultMethod ? " · Recommended" : ""); });
    const statusNames = orderedMethods(defaultMethod).map(method => el("span", "model-status-name", MODEL_INFO[method].name));
    $("service-status-text").replaceChildren(...statusNames);
    orderedMethods(defaultMethod).forEach(method => { const option = [...$("method-select").options].find(value => value.value === method); if (option) $("method-select").insertBefore(option, $("method-select").querySelector('option[value="all"]')); });
    const emptyModels = document.querySelector(".empty-models");
    orderedMethods(defaultMethod).forEach(method => { const node = [...emptyModels.children].find(value => value.textContent.includes(MODEL_INFO[method].name)); if (node) emptyModels.append(node); });
    $("service-status").dataset.state = "ready"; session.ready(true);
  } catch {
    if (!connectionGate.current(job)) return;
    $("service-status").dataset.state = "offline"; setText("service-status-text", "Server unavailable"); $("reconnect").hidden = false; session.ready(false);
  }
}

$("browse-photo").addEventListener("click", () => $("photo-input").click());
$("photo-input").addEventListener("change", event => { const file = event.target.files[0]; event.target.value = ""; void chooseFile(file); });
document.querySelectorAll("[data-source]").forEach(node => node.addEventListener("click", () => setSource(node.dataset.source)));
keyboardTabs("[data-source]", node => setSource(node.dataset.source));
$("method-select").addEventListener("change", event => { methodChosen = true; session.setMethod(event.target.value); announce("Method changed. Previous results cleared."); });
$("analyze-photo").addEventListener("click", () => void session.analyze());
$("compare-all").addEventListener("click", () => { methodChosen = true; session.setMethod("all"); void session.analyze(); });
$("remove-photo").addEventListener("click", () => { session.clear(); $("photo-input").value = ""; announce("Photo and results cleared."); });
$("cancel-request").addEventListener("click", () => { camera.stop(); session.clear(); announce("Request cancelled. Photo and results cleared."); });
$("replace-photo").addEventListener("click", () => { if (source === "camera") beginCamera(); else $("photo-input").click(); });
$("start-camera").addEventListener("click", beginCamera);
$("capture-photo").addEventListener("click", () => void capturePhoto());
$("stop-camera").addEventListener("click", () => { camera.stop(); announce("Camera stopped."); });
$("switch-camera").addEventListener("click", () => void camera.switchCamera());
$("reconnect").addEventListener("click", () => void connect());
$("refresh-evaluation").addEventListener("click", () => void loadEvaluation(true));
document.querySelectorAll("[data-eval]").forEach(node => node.addEventListener("click", () => setEvaluationTab(node.dataset.eval)));
keyboardTabs("[data-eval]", node => setEvaluationTab(node.dataset.eval));
document.querySelector(".skip-link").addEventListener("click", event => { event.preventDefault(); $("main-content").focus(); });
$("photo-preview").addEventListener("load", renderOverlay);
$("photo-preview").addEventListener("error", () => { if (session.state.preview) session.showError(new AppError("INVALID_IMAGE")); });
const overlayObserver = new ResizeObserver(renderOverlay); overlayObserver.observe($("photo-stage"));
let dragDepth = 0; const drop = $("drop-zone");
drop.addEventListener("dragenter", event => { event.preventDefault(); dragDepth += 1; drop.classList.add("drag-active"); });
drop.addEventListener("dragover", event => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"; });
drop.addEventListener("dragleave", event => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) drop.classList.remove("drag-active"); });
drop.addEventListener("drop", event => { event.preventDefault(); dragDepth = 0; drop.classList.remove("drag-active"); const files = [...(event.dataTransfer?.files || [])]; if (files.length !== 1) { session.clear(); session.showError(new AppError("MULTIPLE_UPLOADS")); return; } void chooseFile(files[0]); });
window.addEventListener("dragover", event => { if ([...(event.dataTransfer?.types || [])].includes("Files")) event.preventDefault(); });
window.addEventListener("drop", event => { if ([...(event.dataTransfer?.types || [])].includes("Files")) event.preventDefault(); });
window.addEventListener("hashchange", showRoute);
window.addEventListener("online", () => { void connect(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) camera.stop(); });
window.addEventListener("pagehide", () => { camera.stop(); session.dispose(); connectionGate.cancel(); evaluationGate.cancel(); dashboard.dispose(); overlayObserver.disconnect(); });
window.addEventListener("pageshow", event => { if (event.persisted) { overlayObserver.observe($("photo-stage")); evaluationLoaded = false; showRoute(); void connect(); } });
history.scrollRestoration = "manual";
showRoute(); renderAnalysis(session.state); void connect();

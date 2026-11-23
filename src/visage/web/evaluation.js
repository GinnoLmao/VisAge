import { METHODS, GROUPS, MODEL_INFO, formatNumber, formatPercent, orderedMethods } from "./core.js";
import { element as el, icon, button, panel, note } from "./dom.js";
import { lineChart, legend } from "./charts.js";
const byId = id => document.getElementById(id);
function stats(items) {
  const grid = el("div", `stats-grid${items.length === 4 ? " four" : ""}`);
  items.forEach(([label, value, detail]) => { const card = el("div", "stat-card"); card.append(el("span", "stat-label", label), el("strong", "stat-value", value), el("span", "stat-note", detail)); grid.append(card); }); return grid;
}
function empty(title, text) { const node = el("div", "empty-evidence"); node.append(el("h2", "", title), el("p", "", text)); return node; }
function modelCell(method) {
  const meta = MODEL_INFO[method], cell = el("div", "model-table-cell"), mark = el("span", `model-mini ${meta.tone}`), text = el("div"); mark.append(icon(meta.icon)); text.append(el("strong", "", meta.name), el("small", "", meta.kind)); cell.append(mark, text); return cell;
}
function table(headers, rows, caption) {
  const scroll = el("div", "table-scroll"), node = el("table", "metric-table"), head = el("thead"), row = el("tr"), body = el("tbody");
  scroll.tabIndex = 0; scroll.setAttribute("role", "region"); scroll.setAttribute("aria-label", caption);
  const title = el("caption", "sr-only", caption); node.append(title);
  headers.forEach(label => { const th = el("th", "", label); th.scope = "col"; row.append(th); }); head.append(row);
  rows.forEach(values => { const tr = el("tr"); values.forEach((value, i) => { const cell = el(i === 0 ? "th" : "td"); if (i === 0) { cell.scope = "row"; cell.className = "model-row-heading"; } if (value instanceof Node) cell.append(value); else cell.textContent = String(value); tr.append(cell); }); body.append(tr); });
  node.append(head, body); scroll.append(node); return scroll;
}
function confusion(matrix, normalized, names) {
  const wrap = el("div", "matrix-wrapper");
  if (!matrix || !normalized) { wrap.append(el("p", "chart-caption", "Detailed prediction evidence is unavailable for this artifact.")); return wrap; }
  const node = el("table", "matrix-table"), head = el("thead"), row = el("tr"); node.append(el("caption", "sr-only", "Confusion matrix. Rows are provided age groups; columns are model predictions."));
  row.append(el("th", "", "Actual / predicted")); names.forEach(name => { const th = el("th", "", name); th.scope = "col"; row.append(th); }); head.append(row); node.append(head);
  const body = el("tbody"); matrix.forEach((values, i) => {
    const tr = el("tr"), label = el("th", "", names[i]); label.scope = "row"; tr.append(label);
    values.forEach((count, j) => { const td = el("td"); td.dataset.level = String(Math.round(normalized[i][j] * 10)); td.append(el("span", "matrix-count", formatNumber(count)), el("span", "matrix-rate", formatPercent(normalized[i][j] * 100))); tr.append(td); }); body.append(tr);
  }); node.append(body); wrap.append(node, el("p", "matrix-caption", "Rows: provided age groups. Columns: predictions.\nPercentages are normalized within each row.")); return wrap;
}
export class EvaluationDashboard {
  constructor() { this.data = null; this.model = METHODS[0]; this.disposers = []; this.classificationDisposers = []; }
  dispose() { this.disposers.forEach(fn => fn()); this.classificationDisposers.forEach(fn => fn()); this.disposers = []; this.classificationDisposers = []; }
  render(data, defaultMethod = METHODS[0]) {
    this.dispose(); this.data = data; this.model = defaultMethod;
    byId("eval-classification").replaceChildren(); byId("eval-detection").replaceChildren(); byId("eval-pipeline").replaceChildren();
    this.classification(); this.detection(); this.pipeline();
  }
  classification() {
    const root = byId("eval-classification"), data = this.data.classification;
    if (data.status !== "complete") { root.append(empty("Classification evidence is pending", "Complete and verify the saved model evaluation to populate this view.")); return; }
    const methods = orderedMethods(this.model).filter(method => data.methods[method]);
    if (!methods.includes(this.model)) this.model = methods[0];
    root.append(stats([["Validation images", formatNumber(data.samples), "Original aligned UTKFace crops"], ["Compared methods", formatNumber(methods.length), "Same frozen validation partition"], ["Age groups", formatNumber(data.class_names?.length || GROUPS.length), "Under 18 · 18–59 · 60+"], ["Recommended model", MODEL_INFO[this.model]?.name || "Unavailable", "Chosen across two training seeds"]]));
    root.append(note("Classifier evaluation · Frozen UTKFace validation split. These scores use the original aligned face crops. They do not measure ordinary-camera performance or certainty for a person."));
    const comparison = panel("Three approaches, one comparison", "Accuracy, macro F1 and mean one-vs-rest ROC AUC for the current checkpoints.", "VALIDATION"); comparison.classList.add("table-panel");
    comparison.append(table(["Model", "Accuracy (%)", "Macro F1", "Weighted F1", "Mean AUC", "Runtime (ms)", "Params (M)", "Train time (min)"], methods.map(method => { const value = data.methods[method]; return [modelCell(method), formatPercent(value.accuracy_percent), formatNumber(value.macro_f1, 4), formatNumber(value.weighted_f1, 4), formatNumber(value.macro_auc, 4), formatNumber(value.runtime_median_ms, 2), formatNumber(value.parameters_millions, 2), formatNumber(value.training_minutes, 2)]; }), "Metric 1: age-group accuracy (%); Metric 2: macro F1; runtime, parameters and training time for the current checkpoints"));
    comparison.append(el("p", "table-note", "Each row shows the app’s seed 42 checkpoint. The recommended family is chosen across seeds 42 and 43. Macro F1 weights groups equally; AUC measures ranking, not certainty for a person.")); root.append(comparison);
    if (data.runtime_scope) comparison.append(el("p", "table-note", data.runtime_scope));
    if (data.training_scope) comparison.append(el("p", "table-note", data.training_scope));
    const selector = el("div", "chart-select"); selector.setAttribute("aria-label", "Model to inspect");
    methods.forEach(method => { const control = button(MODEL_INFO[method].name, "", () => { this.model = method; selector.querySelectorAll("button").forEach(node => node.setAttribute("aria-pressed", String(node.dataset.method === method))); this.modelDetails(); }); control.dataset.method = method; control.setAttribute("aria-pressed", String(this.model === method)); selector.append(control); });
    root.append(selector, el("div", "", undefined)); const details = el("div"); details.id = "classification-model-details"; root.append(details); this.modelDetails();
  }
  modelDetails() {
    this.classificationDisposers.forEach(fn => fn()); this.classificationDisposers = [];
    const root = byId("classification-model-details"), value = this.data.classification.methods[this.model], names = this.data.classification.class_names || GROUPS;
    const charts = el("div", "chart-grid"), rocCard = el("section", "panel chart-panel"), matrixCard = el("section", "panel chart-panel");
    rocCard.append(el("h2", "", "ROC curves"), el("p", "chart-subtitle", `${MODEL_INFO[this.model].name} · One age group versus the other two`));
    const plot = el("div", "plot-area"); rocCard.append(plot);
    const series = Object.entries(value.one_vs_rest_roc).map(([label, curve]) => ({ label: `${names[Number(label)]} · ${formatNumber(curve.auc, 4)}`, points: curve.fpr.map((x, i) => [x, curve.tpr[i]]) }));
    rocCard.append(legend(series), el("p", "chart-caption", "Displayed curves are sampled for readability. AUC values use the complete saved score curves."));
    matrixCard.append(el("h2", "", "Where predictions land"), el("p", "chart-subtitle", `${MODEL_INFO[this.model].name} · Confusion matrix`), confusion(value.confusion_matrix, value.row_normalized_confusion_matrix, names));
    if (value.per_class) {
      const boundary = el("div", "boundary-metrics");
      [[value.under18_predicted_older_percent, "True under-18 predicted older"], [value.senior_predicted_below60_percent, "True 60+ predicted below 60"]].forEach(([metric, label]) => { const item = el("div"); item.append(el("strong", "", formatPercent(metric)), el("span", "", label)); boundary.append(item); }); matrixCard.append(boundary);
    }
    charts.append(rocCard, matrixCard); root.replaceChildren(charts);
    if (value.per_class) { const perClass = panel("Per age group metrics", "Precision, recall and F1 for the selected serving checkpoint."); perClass.classList.add("table-panel"); perClass.append(table(["Age group", "Support", "Precision", "Recall", "F1"], value.per_class.map(c => [names[c.label], formatNumber(c.support), formatNumber(c.precision, 4), formatNumber(c.recall, 4), formatNumber(c.f1, 4)]), "Per age group classification metrics")); root.append(perClass); }
    this.classificationDisposers.push(lineChart(plot, series, { title: `${MODEL_INFO[this.model].name} age-group ROC curves`, baseline: true }));
  }
  detection() {
    const root = byId("eval-detection"), data = this.data.detection;
    if (data.status !== "complete") { root.append(empty("Detection evidence is pending", "Complete and verify the detector benchmark to populate this view.")); return; }
    const ap50 = data.metrics.AP50, ap75 = data.metrics.AP75;
    root.append(stats([["WIDER validation images", formatNumber(data.images), "Complete validation split"], ["AP at IoU 0.50", formatPercent(ap50?.ap_percent), "All-valid-box protocol"], ["AP at IoU 0.75", formatPercent(ap75?.ap_percent), "A stricter box-overlap threshold"], ["Valid ground-truth faces", formatNumber(ap50?.ground_truth_faces), "Independently supplied annotations"]]));
    root.append(note("Detector evaluation · WIDER FACE validation. This all-valid-box AP protocol differs from the official easy/medium/hard leaderboard. Detector AP measures face boxes, separately from age-group accuracy."));
    const charts = el("div", "chart-grid"), pr = el("section", "panel chart-panel"), froc = el("section", "panel chart-panel"), prPlot = el("div", "plot-area"), frocPlot = el("div", "plot-area");
    pr.append(el("h2", "", "Precision and recall"), el("p", "chart-subtitle", "YuNet · Two box-overlap thresholds"), prPlot);
    const series = Object.entries(data.metrics).map(([name, curve]) => ({ label: `${name} · ${formatPercent(curve.ap_percent)}`, points: curve.recall.map((x, i) => [x, curve.precision[i]]) })); pr.append(legend(series));
    froc.append(el("h2", "", "Recall and false detections"), el("p", "chart-subtitle", "FROC · IoU 0.50"), frocPlot, el("p", "chart-caption", "False positives are counted per evaluated image. The benchmark confidence floor bounds the available recall."));
    charts.append(pr, froc); root.append(charts);
    this.disposers.push(lineChart(prPlot, series, { title: "Face-detection precision-recall curves", xLabel: "Recall", yLabel: "Precision" }));
    const points = ap50?.froc?.false_positives_per_image?.map((x, i) => [x, ap50.froc.recall[i]]) || [], max = points.length ? Math.max(1, ...points.map(point => point[0])) : 1;
    this.disposers.push(lineChart(frocPlot, [{ label: "AP50", points }], { title: "Face-detection FROC", xLabel: "False positives per image", yLabel: "Face recall", xMax: Math.ceil(max / 10) * 10 }));
    const details = el("section", "panel detail-panel"); details.append(el("h2", "", "Read the operating point alongside the score"));
    details.append(el("p", "", `The AP sweep uses a confidence floor of ${formatNumber(data.benchmark_configuration?.score_threshold, 2)}. The application uses ${formatNumber(data.serving_confidence_threshold, 2)} before face selection and quality checks. These are different evaluation and application operating points.`));
    details.append(el("p", "", "Invalid annotation boxes follow the recorded ignore policy; duplicate valid hits count as false positives. This development benchmark does not replace independent local-photo testing.")); root.append(details);
  }
  pipeline() {
    const root = byId("eval-pipeline"), data = this.data.face_pipeline;
    if (data.status !== "complete") { root.append(empty("Pipeline evidence is pending", "Complete the shared detection, alignment and quality diagnostic to populate this view.")); return; }
    root.append(stats([["Attempted images", formatNumber(data.attempted), "Every attempted input is counted"], ["Accepted images", formatNumber(data.accepted), "Passed detection and quality checks"], ["Coverage", formatPercent(data.coverage * 100), `${formatNumber(data.rejected)} rejected inputs`]]));
    root.append(note("Pipeline diagnostic · Existing UTKFace validation crops are detected and aligned again. These results measure integration on research crops, not independent camera-photo accuracy. The consented local pilot remains pending participant data."));
    const coverage = el("section", "panel detail-panel"), bar = el("progress", "coverage-track"); bar.max = data.attempted || 1; bar.value = data.accepted; bar.setAttribute("aria-label", "Accepted images out of all attempted images");
    const caption = el("div", "coverage-legend"); caption.append(el("span", "", `${formatNumber(data.accepted)} accepted`), el("span", "", `${formatNumber(data.rejected)} rejected`));
    coverage.append(el("h2", "", "Every attempt stays in the picture"), el("p", "", "Coverage includes rejected images. End-to-end correct rate divides correct predictions by all attempted images, so rejecting a difficult input does not silently inflate the overall result."), bar, caption); root.append(coverage);
    const comparison = panel("Accepted predictions and end-to-end results", "Compare the two denominators explicitly.", "VALIDATION"); comparison.classList.add("table-panel");
    comparison.append(table(["Model", "Accepted accuracy", "Accepted F1", "End-to-end correct"], METHODS.filter(method => data.methods[method]).map(method => { const value = data.methods[method]; return [modelCell(method), formatPercent(value.accepted_accuracy_percent), formatNumber(value.accepted_macro_f1, 4), formatPercent(value.end_to_end_correct_percent)]; }), "Accepted-image and all-attempted-image pipeline metrics"));
    comparison.append(el("p", "table-note", `Accepted-image metrics use ${formatNumber(data.accepted)} inputs. End-to-end rates use all ${formatNumber(data.attempted)} attempts.`)); root.append(comparison);
  }
}

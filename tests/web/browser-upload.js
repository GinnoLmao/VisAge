import { METHODS, MODEL_INFO, GROUPS, orderedMethods } from "/assets/core.js";
// Browser integration tests against the actual DOM, fetch client and loaded API.
// Only the isolated QA server imports this module, with ?qa_upload=1.
const byId = id => document.getElementById(id);
const passed = [];
async function until(predicate, message, timeout = 15000) {
  const end = performance.now() + timeout;
  while (performance.now() < end) { if (predicate()) return; await new Promise(resolve => setTimeout(resolve, 30)); }
  throw new Error(message);
}
function check(condition, message) { if (!condition) throw new Error(message); passed.push(message); }
function drop(files) {
  const transfer = new DataTransfer(); files.forEach(file => transfer.items.add(file));
  byId("drop-zone").dispatchEvent(new DragEvent("drop", { bubbles: true, cancelable: true, dataTransfer: transfer }));
}
function pick(file) {
  const transfer = new DataTransfer(); transfer.items.add(file); byId("photo-input").files = transfer.files;
  byId("photo-input").dispatchEvent(new Event("change", { bubbles: true }));
}
function errorIs(codeText) { return !byId("input-error").hidden && byId("input-error-title").textContent.includes(codeText); }
export async function run() {
  try {
    await until(() => byId("service-status").dataset.state === "ready", "Models never became ready");
    const metadata = await (await fetch("/v1/models")).json();
    check(byId("method-select").value === metadata.default_method, "The menu starts on the validation-selected default");
    check([...byId("method-select").options].map(v => v.value).filter(v => v !== "all").join() === orderedMethods(metadata.default_method).join(), "The active menu has all three models with the recommended model first");
    check(METHODS.every(m => byId("service-status-text").textContent.includes(MODEL_INFO[m].name)), "The connection header displays the names of all three models");
    check(!/Academic prototype|RESEARCH WORKSPACE|Built for research|Computer vision research/.test(document.body.textContent), "Requested prototype and research branding is removed");
    byId("tab-upload").click();
    const response = await fetch("/__qa__/fixture.jpg"), blob = await response.blob();
    const photo = new File([blob], "validation-fixture.jpg", { type: "image/jpeg" });
    drop([photo]);
    await until(() => !byId("analyze-photo").disabled, "Drop did not complete face detection");
    check(byId("face-buttons").children.length === 1, "Dropped image detects and selects one face");
    byId("method-select").value = "all"; byId("method-select").dispatchEvent(new Event("change")); byId("analyze-photo").click();
    await until(() => !byId("prediction-results").hidden, "Three-method prediction did not complete");
    check(byId("model-results").children.length === 3, "Compare all renders three real model results");
    check(METHODS.every(m => byId("model-results").textContent.includes(MODEL_INFO[m].name)), "Result cards identify all three new architectures");
    check(GROUPS.some(g => byId("face-overlay").textContent.includes(g)), "The selected face overlay shows its estimated age group");
    check(byId("model-results").textContent.includes("ms"), "Result cards show real processing times");
    const raw = await (await fetch("/v1/predict?method=all&face_index=0", {method:"POST", body:photo, headers:{"Content-Type":"image/jpeg"}})).json();
    const cards = [...byId("model-results").children];
    check(cards[0].dataset.method === metadata.default_method, "The recommended model appears first in Compare All results");
    check(cards.every(card => card.querySelectorAll(".score-row").length === 3), "Every model shows all three age-group scores");
    check(cards.every(card => [...card.querySelectorAll("meter")].every((meter, j) => Math.abs(meter.value - raw.predictions.find(value => value.method === card.dataset.method).class_scores[j].score) < .000001)), "Score bars match the unrounded real API outputs");
    check(cards.every((card, i) => [...card.querySelectorAll(".score-row")].every((row, j) => row.dataset.classId === String(j) && row.textContent.includes(GROUPS[j]))), "All three scores retain their class labels and fixed order");
    check(cards.every(card => card.querySelector(".score-selected").dataset.classId === String(raw.predictions.find(value => value.method === card.dataset.method).class_id)), "The chosen class matches the highest model score");
    check(cards.every(card => card.querySelector(".score-heading").textContent === "Model scores") && !byId("score-explanation"), "Percentage headings say Model scores and the removed explanation is absent");
    check(!/next highest|next-ranked/i.test(byId("prediction-results").textContent), "All classes are shown without a next-highest callout");
    const heicBlob = await (await fetch("/__qa__/fixture.heic")).blob();
    drop([new File([heicBlob], "validation-fixture.heic", {type:"image/heic"})]);
    await until(() => !byId("analyze-photo").disabled && byId("photo-preview").naturalWidth > 0, "HEIC preview did not load");
    const heicDetection = await (await fetch("/v1/faces", {method:"POST", body:heicBlob, headers:{"Content-Type":"image/heic"}})).json();
    check(heicDetection.preview?.content_type === "image/jpeg" && byId("photo-preview").src.startsWith("blob:") && byId("photo-preview").naturalWidth > 0, "HEIC receives a browser-compatible preview from the server");
    byId("analyze-photo").click(); await until(() => !byId("prediction-results").hidden, "HEIC prediction did not finish");
    check(byId("model-results").querySelectorAll("meter").length === 9, "HEIC reaches all three models and displays all nine scores");
    drop([new File(["broken"], "broken.jpg", { type: "image/jpeg" })]);
    await until(() => errorIs("read this image"), "Corrupt image error did not appear");
    check(byId("prediction-results").hidden && byId("analyze-photo").disabled, "Corrupt image clears previous results and disables prediction");
    drop([new File([new Uint8Array(8 * 1024 ** 2 + 1)], "large.jpg", { type: "image/jpeg" })]);
    await until(() => errorIs("too large"), "Oversized image error did not appear");
    check(byId("photo-stage").hidden, "Oversized image is rejected before preview");
    drop([photo, photo]); await until(() => errorIs("one photo"), "Multiple-file error did not appear");
    check(byId("prediction-results").hidden, "Multiple-file drop keeps results clear");
    const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 400;
    const ctx = canvas.getContext("2d"); ctx.fillStyle = "#888888"; ctx.fillRect(0, 0, 640, 400);
    const blank = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
    drop([new File([blank], "blank.png", { type: "image/png" })]);
    await until(() => errorIs("find a face"), "No-face error did not appear");
    check(byId("analyze-photo").disabled && byId("face-buttons").children.length === 0, "No-face image cannot be analyzed");
    pick(photo); await until(() => !byId("analyze-photo").disabled, "File picker recovery did not detect a face");
    check(byId("input-error").hidden, "A valid file-picker selection recovers from upload errors");
    byId("analyze-photo").click(); await until(() => !byId("prediction-results").hidden, "Recovered prediction did not complete");
    byId("method-select").value = "convnextv2_nano"; byId("method-select").dispatchEvent(new Event("change"));
    check(byId("prediction-results").hidden && !byId("analyze-photo").disabled, "Changing the method clears results and keeps the selected face");
    check(!GROUPS.some(g => byId("face-overlay").textContent.includes(g)), "Changing the method also clears the stale overlay age group");
    check(byId("model-results").children.length === 0, "Changing the method also removes stale class scores");
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = (path, ...options) => String(path).startsWith("/v1/predict") ? Promise.reject(new TypeError("Simulated connection loss")) : originalFetch(path, ...options);
      byId("analyze-photo").click();
      await until(() => errorIs("reach VisAge"), "Connection-loss feedback did not appear");
      check(!byId("reconnect").hidden && byId("analyze-photo").disabled, "Connection loss exposes Retry and pauses prediction");
    } finally { globalThis.fetch = originalFetch; }
    byId("reconnect").click();
    await until(() => byId("service-status").dataset.state === "ready" && !byId("analyze-photo").disabled, "Reconnect did not restore prediction");
    check(byId("input-error").hidden && byId("face-buttons").children.length === 1, "Reconnect clears the connection error and preserves the selected face");
    byId("tab-camera").click(); byId("start-camera").click();
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
    check(byId("model-results").querySelectorAll("meter").length === 9, "Camera results expose all three scores from each model");
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
    check(["Accuracy (%)", "Macro F1", "Runtime (ms)", "Params (M)", "Train time (min)"].every(label => byId("eval-classification").textContent.includes(label)), "Dashboard contains both metrics, runtime, parameters and training time");
    for (const method of METHODS) {
      document.querySelector(`#eval-classification button[data-method="${method}"]`).click();
      check(byId("classification-model-details").textContent.includes(MODEL_INFO[method].name) && byId("classification-model-details").querySelector("svg"), `ROC and confusion details render for ${method}`);
    }
    check(document.documentElement.scrollWidth <= innerWidth + 1, "The dashboard fits the browser viewport");
    byId("eval-detection-tab").click();
    check(byId("eval-detection").textContent.includes("AP at IoU 0.50") && byId("eval-detection").querySelector("svg"), "Detection AP and its curves remain available");
    byId("eval-pipeline-tab").click();
    check(!byId("eval-pipeline").textContent.includes("Reasons returned by the shared pipeline") && !byId("eval-pipeline").querySelector('div[aria-label="Pipeline rejection reasons"]'), "The removed pipeline rejection-reasons table is absent");
    const events = await (await fetch("/__qa__/events")).json();
    check(events.filter(e => e.event === "camera-tracks").at(-1)?.active === 0, "Every simulated camera track was stopped");
    location.hash = "analyze";
    await until(() => !byId("view-analyze").hidden, "Could not return to analysis");
    byId("remove-photo").click();
    check(byId("photo-stage").hidden && byId("prediction-results").hidden && byId("analyze-photo").disabled && !byId("photo-preview").hasAttribute("src"), "Reset removes preview, selection and results");
    await fetch("/__qa__/events", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ event: "upload-integration", ok: true, passed }) });
    document.getElementById("qa-camera-badge").textContent = `QA upload integration · ${passed.length} checks passed`;
  } catch (error) {
    await fetch("/__qa__/events", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ event: "upload-integration", ok: false, passed, error: String(error) }) });
    document.getElementById("qa-camera-badge").textContent = `QA upload integration FAILED · ${String(error)}`;
  }
}

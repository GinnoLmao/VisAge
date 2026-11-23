// QA-only simulated video devices. This module is absent from production HTML.
const mode = new URLSearchParams(location.search).get("qa_camera") || "normal";
const active = new Set();
const badge = document.createElement("aside"); badge.id = "qa-camera-badge";
document.body.append(badge);
function report(record) { void fetch("/__qa__/events", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ time: new Date().toISOString(), mode, ...record }) }); }
function status() { badge.textContent = `QA simulated camera · ${mode} · active tracks ${active.size}`; report({ event: "camera-tracks", active: active.size }); }
const image = new Image(); image.src = "/__qa__/fixture.jpg";
const devices = ["qa-front", "qa-back"].map((deviceId, i) => ({ kind: "videoinput", deviceId, groupId: "qa", label: `Simulated camera ${i + 1}` }));
Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: {
  enumerateDevices: async () => devices,
  getUserMedia: async constraints => {
    report({ event: "camera-request", audio: constraints.audio, exactDevice: constraints.video?.deviceId?.exact || null });
    if (mode !== "normal") throw new DOMException("Simulated device failure", { denied: "NotAllowedError", missing: "NotFoundError", busy: "NotReadableError" }[mode] || "NotReadableError");
    await image.decode();
    const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext("2d");
    function draw() { ctx.fillStyle = "#dfe7f3"; ctx.fillRect(0, 0, 640, 480); ctx.drawImage(image, 170, 90, 300, 300); }
    draw(); const timer = setInterval(draw, 80); const stream = canvas.captureStream(12);
    const track = stream.getVideoTracks()[0]; const stop = track.stop.bind(track);
    const deviceId = constraints.video?.deviceId?.exact || "qa-front";
    track.getSettings = () => ({ deviceId, facingMode: deviceId === "qa-front" ? "user" : "environment", width: 640, height: 480 });
    track.stop = () => { clearInterval(timer); active.delete(track); stop(); status(); };
    active.add(track); status(); return stream;
  }
} });
status();
window.addEventListener("error", event => report({ event: "browser-error", message: event.message }));
window.addEventListener("unhandledrejection", event => report({ event: "browser-rejection", message: String(event.reason) }));
let timer;
function snapshot() {
  clearTimeout(timer); timer = setTimeout(() => {
    report({ event: "render", route: location.hash, width: innerWidth, documentWidth: document.documentElement.scrollWidth, active: active.size,
      faceButtons: document.querySelectorAll("#face-buttons button").length,
      results: document.querySelector("#prediction-results:not([hidden])")?.textContent || "",
      error: document.querySelector("#input-error:not([hidden])")?.textContent || "",
      headings: [...document.querySelectorAll(".view:not([hidden]) h1, .view:not([hidden]) h2")].map(node => node.textContent) });
  }, 200);
}
new MutationObserver(snapshot).observe(document.getElementById("main-content"), { subtree: true, childList: true, attributes: true });
window.addEventListener("resize", snapshot); window.addEventListener("hashchange", snapshot); snapshot();
if (new URLSearchParams(location.search).get("qa_upload") === "1") {
  window.addEventListener("load", () => { void import("/__qa__/upload-tests.js").then(module => module.run()); }, { once: true });
}

import { METHODS, MODEL_INFO } from "./models.js";
export { METHODS, MODEL_INFO };
export const GROUPS = ["Under 18", "18–59", "60+"];
export const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence", "image/avif", "image/bmp", "image/x-ms-bmp", "image/tiff"];
export function orderedMethods(recommended) { return METHODS.includes(recommended) ? [recommended, ...METHODS.filter(method => method !== recommended)] : [...METHODS]; }
function nativePreview(blob, name) { return !/heic|heif|tiff|avif|bmp/i.test(blob.type || "") && !/\.(heic|heif|tiff?|avif|bmp)$/i.test(name); }
function previewBlob(preview) {
  if (preview?.content_type !== "image/jpeg" || !Number.isInteger(preview.width) || !Number.isInteger(preview.height)
    || Math.min(preview.width, preview.height) < 1 || Math.max(preview.width, preview.height) > 1280
    || typeof preview.data_base64 !== "string" || preview.data_base64.length > 4000000 || !/^[A-Za-z0-9+/]+={0,2}$/.test(preview.data_base64)) throw new AppError("BAD_RESPONSE");
  try { return new Blob([Uint8Array.from(atob(preview.data_base64), value => value.charCodeAt(0))], { type: "image/jpeg" }); }
  catch { throw new AppError("BAD_RESPONSE"); }
}
export class AppError extends Error {
  constructor(code, message, details = {}) { super(message); this.name = "AppError"; this.code = code; this.details = details; }
}
export function explainError(error) {
  const messages = {
    NO_FACE: ["We couldn’t find a face", "Try a clearer, front-facing photo with the whole face visible."],
    MULTIPLE_FACES: ["Choose one face", "Select a face box or a face button, then analyze again."],
    TOO_MANY_FACES: ["A little too crowded", "Choose a photo with fewer people or take a closer photo of one person."],
    FACE_TOO_SMALL: ["Move a little closer", "The selected face is too small to assess. Try a higher-resolution photo or a closer capture."],
    UNUSABLE_FACE: ["Try a clearer photo", "Keep the full face visible and upright, with even lighting and less blur."],
    IMAGE_TOO_LARGE: ["This file is too large", "Choose a JPEG, PNG or WebP image up to 8 MiB."],
    IMAGE_DIMENSIONS_TOO_LARGE: ["The image resolution is too large", "Resize the image to 20 megapixels or fewer, then try again."],
    IMAGE_TOO_SMALL: ["This image is too small", "Choose a larger image with a clearly visible face."],
    EMPTY_IMAGE: ["The file is empty", "Choose another photo and try again."],
    INVALID_IMAGE: ["We couldn’t read this image", "Choose an undamaged JPEG, PNG or WebP photo."],
    UNSUPPORTED_IMAGE: ["Choose a supported photo", "Use JPEG, PNG, WebP, HEIC/HEIF, AVIF, BMP or a single-page TIFF."],
    MULTIPLE_UPLOADS: ["Choose one photo at a time", "Drop a single image. You can select a face if the photo contains several people."],
    ANIMATED_IMAGE: ["Use a still photo", "Animated images are not supported. Choose one still frame instead."],
    UNSUPPORTED_MEDIA_TYPE: ["Choose a supported photo", "Use JPEG, PNG, WebP, HEIC/HEIF, AVIF, BMP or a single-page TIFF."],
    SERVER_BUSY: ["The models are busy", "Wait a moment, then analyze the selected face again."],
    QUEUE_TIMEOUT: ["The request waited too long", "The models are busy. Please try again shortly."],
    NOT_READY: ["The models are not ready yet", "Wait for the connection indicator to show models ready, then retry."],
    OFFLINE: ["We couldn’t reach VisAge", "Check that the local VisAge server is running, then use Retry in the connection indicator."],
    REQUEST_TIMEOUT: ["The request took too long", "Check the connection and try again. A clearer, smaller photo may help."],
    UPLOAD_TIMEOUT: ["The upload took too long", "Try a smaller file or a more stable connection."],
    BAD_RESPONSE: ["We couldn’t read the result", "The response was incomplete. Please try again."],
    CAMERA_UNAVAILABLE: ["Camera access is unavailable", "Open VisAge on localhost or a secure connection in a camera-capable browser. You can also upload a photo."],
    CAMERA_DENIED: ["Camera permission was not granted", "Allow camera access using your browser’s site controls, or choose Upload photo."],
    CAMERA_MISSING: ["No camera was found", "Connect a camera or upload a photo from your device."],
    CAMERA_BUSY: ["The camera couldn’t start", "Another application may be using it. Close that application and retry, or upload a photo."],
    CAMERA_ENDED: ["The camera stopped", "Start the camera again, or choose Upload photo."],
    CAMERA_CAPTURE: ["We couldn’t capture the photo", "Wait for the camera preview, then try again."],
    EVALUATION_UNAVAILABLE: ["Evaluation records are unavailable", "The saved evidence may not match the loaded models. Retry or check the server’s evaluation records."],
  };
  if (error?.code === "IMAGE_TOO_LARGE" && Number.isFinite(error.details?.maxBytes)) return {title:"This file is too large",message:`Choose a photo up to ${error.details.maxBytes / 1024**2} MiB.`,code:error.code};
  const [title, message] = messages[error?.code] || ["Something interrupted the analysis", "Please try again. If the problem continues, check the VisAge server."];
  return { title, message, code: error?.code || "UNKNOWN" };
}
export function validatePhoto(blob, name = "", maxBytes = 8 * 1024**2) {
  if (!blob || !Number.isFinite(blob.size) || blob.size <= 0) throw new AppError("EMPTY_IMAGE");
  if (blob.size > maxBytes) throw new AppError("IMAGE_TOO_LARGE", "Image exceeds the upload limit.", {maxBytes});
  const type = (blob.type || "").toLowerCase();
  if (!IMAGE_TYPES.includes(type) && !(["", "application/octet-stream"].includes(type) && /\.(jpe?g|png|webp|heic|heif|avif|bmp|tiff?)$/i.test(name))) throw new AppError("UNSUPPORTED_IMAGE");
}
export async function requestJSON(path, { signal, timeout = 45000, fetcher = globalThis.fetch, ...options } = {}) {
  const controller = new AbortController();
  let expired = false;
  const relay = () => controller.abort();
  if (signal?.aborted) controller.abort();
  signal?.addEventListener("abort", relay, { once: true });
  const timer = setTimeout(() => { expired = true; controller.abort(); }, timeout);
  try {
    const response = await fetcher(path, { cache: "no-store", credentials: "same-origin", ...options, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { throw new AppError("BAD_RESPONSE"); }
    if (!response.ok) throw new AppError(data?.error?.code || "INFERENCE_FAILED", data?.error?.message || "Request failed", data?.error?.details);
    return data;
  } catch (error) {
    if (expired) throw new AppError("REQUEST_TIMEOUT");
    if (controller.signal.aborted) throw error;
    if (error instanceof AppError) throw error;
    throw new AppError("OFFLINE");
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", relay);
  }
}
export const api = {
  health: () => requestJSON("/health", { timeout: 6000 }),
  models: () => requestJSON("/v1/models", { timeout: 6000 }),
  evaluation: (signal) => requestJSON("/v1/evaluation", { signal, timeout: 25000 }),
  faces: (blob, signal) => requestJSON("/v1/faces", { method: "POST", body: blob, headers: { "Content-Type": blob.type || "application/octet-stream" }, signal }),
  predict: (blob, method, face, signal) => requestJSON(`/v1/predict?method=${encodeURIComponent(method)}&face_index=${face}`, { method: "POST", body: blob, headers: { "Content-Type": blob.type || "application/octet-stream" }, signal }),
};
export class RequestGate {
  constructor() { this.generation = 0; this.controller = null; }
  cancel() { this.controller?.abort(); this.controller = null; this.generation += 1; }
  start() { this.cancel(); this.controller = new AbortController(); return { generation: this.generation, signal: this.controller.signal }; }
  current(job) { return this.generation === job.generation && !job.signal.aborted; }
}
function blankState(method = METHODS[0], ready = false) {
  return { method, ready, blob: null, name: "", preview: "", image: null, faces: [], selected: null, busy: null, result: null, error: null };
}
function validFaces(response) {
  const image = response?.image;
  return image && Number.isFinite(image.width) && Number.isFinite(image.height) && image.width >= 32 && image.height >= 32 && image.width * image.height <= 20000000
    && Array.isArray(response.faces) && response.faces.length > 0 && response.faces.length <= 10
    && new Set(response.faces.map(face => face.index)).size === response.faces.length
    && response.faces.every(face => Number.isInteger(face.index) && face.index >= 0 && face.index < 10
      && Array.isArray(face.bbox) && face.bbox.length === 4 && face.bbox.every(Number.isFinite) && face.bbox[2] > 0 && face.bbox[3] > 0);
}
export function scoreBreakdown(value) {
  const scores = value?.class_scores;
  if (value?.score_type !== "uncalibrated_softmax" || !Array.isArray(scores) || scores.length !== GROUPS.length
    || !scores.every((entry, index) => entry.class_id === index && typeof entry.age_group === "string"
      && entry.age_group.replaceAll("–", "-") === GROUPS[index].replaceAll("–", "-")
      && typeof entry.score === "number" && Number.isFinite(entry.score) && entry.score >= 0 && entry.score <= 1)
    || Math.abs(scores.reduce((sum, entry) => sum + entry.score, 0) - 1) > 0.000001) return null;
  const selected = scores.reduce((best, entry) => entry.score > best.score ? entry : best);
  if (value.class_id !== selected.class_id) return null;
  return { classes: scores, selected };
}
function validPrediction(response, state) {
  const expected = state.method === "all" ? METHODS : [state.method];
  return response?.selected_face?.index === state.selected && response?.image?.width === state.image.width && response?.image?.height === state.image.height
    && Array.isArray(response.predictions) && response.predictions.length === expected.length
    && response.predictions.every((value, i) => value.method === expected[i] && Number.isInteger(value.class_id) && value.class_id >= 0 && value.class_id <= 2 && scoreBreakdown(value));
}
export class AnalysisSession {
  constructor({ client = api, notify = () => {}, createURL = blob => URL.createObjectURL(blob), revokeURL = url => URL.revokeObjectURL(url) } = {}) {
    this.api = client; this.notify = notify; this.createURL = createURL; this.revokeURL = revokeURL; this.gate = new RequestGate(); this.state = blankState();
  }
  emit() { this.notify(this.state); }
  ready(value) { this.state.ready = Boolean(value); if (value && this.state.error?.code === "OFFLINE") this.state.error = null; this.emit(); }
  clear() {
    this.gate.cancel();
    if (this.state.preview) this.revokeURL(this.state.preview);
    this.state = blankState(this.state.method, this.state.ready); this.emit();
  }
  setMethod(method) {
    if (![...METHODS, "all"].includes(method)) return false;
    if (this.state.method === method) return true;
    if (this.state.busy === "predicting") { this.gate.cancel(); this.state.busy = null; }
    this.state.method = method; this.state.result = null; this.emit(); return true;
  }
  selectFace(index) {
    if (!this.state.faces.some(face => face.index === index)) return false;
    if (index === this.state.selected) return true;
    this.gate.cancel(); this.state.selected = index; this.state.result = null; this.state.busy = null; this.state.error = null; this.emit(); return true;
  }
  showError(error) { this.state.error = explainError(error); this.emit(); }
  async setInput(blob, name = "Selected photo") {
    this.clear();
    try { validatePhoto(blob, name, this.maxUploadBytes); } catch (error) { this.showError(error); return; }
    const job = this.gate.start();
    this.state.blob = blob; this.state.name = name; this.state.busy = "detecting"; this.emit();
    try {
      const response = await this.api.faces(blob, job.signal);
      if (!this.gate.current(job)) return;
      if (!validFaces(response)) throw new AppError("BAD_RESPONSE");
      this.state.image = response.image; this.state.faces = response.faces;
      this.state.selected = response.faces.length === 1 ? response.faces[0].index : null;
      this.state.preview = this.createURL(response.preview ? previewBlob(response.preview) : blob);
    } catch (error) {
      if (!this.gate.current(job)) return;
      if (["NO_FACE", "TOO_MANY_FACES"].includes(error.code) && nativePreview(blob, name)) this.state.preview = this.createURL(blob);
      this.state.error = explainError(error);
      if (error.code === "OFFLINE") this.state.ready = false;
    } finally {
      if (this.gate.current(job)) { this.state.busy = null; this.emit(); }
    }
  }
  async analyze() {
    if (this.state.busy || !this.state.blob || this.state.selected === null || !this.state.ready) return;
    const job = this.gate.start(); this.state.busy = "predicting"; this.state.result = null; this.state.error = null; this.emit();
    try {
      const result = await this.api.predict(this.state.blob, this.state.method, this.state.selected, job.signal);
      if (!this.gate.current(job)) return;
      if (!validPrediction(result, this.state)) throw new AppError("BAD_RESPONSE");
      this.state.result = result;
    } catch (error) {
      if (this.gate.current(job)) { this.state.error = explainError(error); if (error.code === "OFFLINE") this.state.ready = false; }
    } finally {
      if (this.gate.current(job)) { this.state.busy = null; this.emit(); }
    }
  }
  dispose() { this.clear(); }
}
function stopStream(stream) { stream?.getTracks().forEach(track => track.stop()); }
function cameraError(error) {
  if (error instanceof AppError) return error;
  const codes = { NotAllowedError: "CAMERA_DENIED", SecurityError: "CAMERA_DENIED", NotFoundError: "CAMERA_MISSING", DevicesNotFoundError: "CAMERA_MISSING", NotReadableError: "CAMERA_BUSY", TrackStartError: "CAMERA_BUSY", OverconstrainedError: "CAMERA_MISSING" };
  return new AppError(codes[error?.name] || "CAMERA_BUSY");
}
export class CameraController {
  constructor({ video, mediaDevices = globalThis.navigator?.mediaDevices, canvasFactory = () => document.createElement("canvas"), notify = () => {} } = {}) {
    this.video = video; this.mediaDevices = mediaDevices; this.canvasFactory = canvasFactory; this.notify = notify;
    this.stream = null; this.generation = 0; this.state = { phase: "idle", devices: [], facing: "user", error: null, device: null };
  }
  emit() { this.notify(this.state); }
  stop(phase = "idle") {
    this.generation += 1; const stream = this.stream; this.stream = null;
    if (this.video) this.video.srcObject = null;
    this.state.phase = phase; stopStream(stream); this.emit();
  }
  async start({ device, facing = this.state.facing } = {}) {
    this.stop(); const generation = this.generation;
    this.state.phase = "starting"; this.state.facing = facing; this.state.error = null; this.emit();
    if (!this.mediaDevices?.getUserMedia) { this.state.phase = "error"; this.state.error = explainError(new AppError("CAMERA_UNAVAILABLE")); this.emit(); return; }
    let acquired;
    try {
      acquired = await this.mediaDevices.getUserMedia({ audio: false, video: { ...(device ? { deviceId: { exact: device } } : { facingMode: { ideal: facing } }), width: { ideal: 1280 }, height: { ideal: 960 } } });
      if (generation !== this.generation) { stopStream(acquired); return; }
      this.stream = acquired; this.video.srcObject = acquired;
      await this.video.play();
      if (generation !== this.generation) { stopStream(acquired); return; }
      const settings = acquired.getVideoTracks()[0]?.getSettings?.() || {};
      this.state.device = settings.deviceId || device || null;
      this.state.facing = settings.facingMode || facing;
      let devices = [];
      try { devices = (await this.mediaDevices.enumerateDevices?.() || []).filter(value => value.kind === "videoinput"); } catch { /* Preview still works without device enumeration. */ }
      if (generation !== this.generation) { stopStream(acquired); return; }
      this.state.devices = devices; this.state.phase = "live";
      acquired.getVideoTracks().forEach(track => { track.onended = () => { if (generation === this.generation && this.state.phase === "live") { this.stop("error"); this.state.error = explainError(new AppError("CAMERA_ENDED")); this.emit(); } }; });
      this.emit();
    } catch (error) {
      stopStream(acquired);
      if (generation !== this.generation) return;
      this.stream = null; this.video.srcObject = null; this.state.phase = "error"; this.state.error = explainError(cameraError(error)); this.emit();
    }
  }
  async switchCamera() {
    if (this.state.phase !== "live" || this.state.devices.length < 2) return;
    const i = this.state.devices.findIndex(device => device.deviceId === this.state.device);
    const device = this.state.devices[(i + 1) % this.state.devices.length];
    await this.start({ device: device.deviceId, facing: this.state.facing === "user" ? "environment" : "user" });
  }
  async capture() {
    if (this.state.phase !== "live" || !this.video.videoWidth || !this.video.videoHeight) throw new AppError("CAMERA_CAPTURE");
    const generation = this.generation; const canvas = this.canvasFactory();
    const ratio = Math.min(1, 1280 / Math.max(this.video.videoWidth, this.video.videoHeight));
    canvas.width = Math.max(1, Math.round(this.video.videoWidth * ratio)); canvas.height = Math.max(1, Math.round(this.video.videoHeight * ratio));
    const context = canvas.getContext("2d");
    if (!context) throw new AppError("CAMERA_CAPTURE");
    context.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", .92));
    if (generation !== this.generation) return null;
    if (!blob) throw new AppError("CAMERA_CAPTURE");
    this.stop("captured"); return blob;
  }
}
export function formatNumber(value, digits = 0) { return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }); }
export function formatPercent(value) { return value === null || value === undefined || !Number.isFinite(value) ? "—" : `${value.toFixed(2)}%`; }
export function formatModelScore(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) return "—";
  if (value > 0 && value < 0.0001) return "<0.01%";
  if (value < 1 && value > 0.9999) return ">99.99%";
  return formatPercent(value * 100);
}
export function formatTime(value) { return value === null || value === undefined || !Number.isFinite(value) || value < 0 ? "—" : `${value.toFixed(1)} ms`; }

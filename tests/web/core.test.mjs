import test from "node:test";
import assert from "node:assert/strict";
import { AnalysisSession, CameraController, AppError, RequestGate, requestJSON, validatePhoto, explainError, formatPercent, formatTime, scoreBreakdown, formatModelScore, GROUPS, orderedMethods } from "../../src/visage/web/core.js";

const photo = () => new Blob(["sample image bytes"], { type: "image/jpeg" });
const faces = (count = 1) => ({ image: { width: 200, height: 200 }, faces: Array.from({ length: count }, (_, index) => ({ index, bbox: [10 + index * 50, 20, 45, 90] })) });
const prediction = (method = "convnextv2_nano", index = 0) => ({ image: { width: 200, height: 200 }, selected_face: { index }, predictions: (method === "all" ? ["convnextv2_nano", "efficientnetv2_s", "swinv2_tiny"] : [method]).map(method => ({ method, class_id: 1, score_type: "uncalibrated_softmax", class_scores: GROUPS.map((age_group, class_id) => ({ class_id, age_group, score: [.1, .8, .1][class_id] })) })) });
function deferred() { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; }
function sessionFor(client) { let serial = 0; const revoked = []; const session = new AnalysisSession({ client, createURL: () => `blob:test-${++serial}`, revokeURL: url => revoked.push(url) }); session.ready(true); return { session, revoked }; }

test("new input wins over a late detector response and old requests are aborted", async () => {
  const first = deferred(), second = deferred(), signals = [];
  const { session } = sessionFor({ faces: (_, signal) => { signals.push(signal); return signals.length === 1 ? first.promise : second.promise; } });
  const a = session.setInput(photo(), "first.jpg"), b = session.setInput(photo(), "second.jpg");
  assert.equal(signals[0].aborted, true);
  second.resolve(faces(2)); await b; first.resolve(faces()); await a;
  assert.equal(session.state.name, "second.jpg"); assert.equal(session.state.faces.length, 2); assert.equal(session.state.selected, null); assert.equal(session.state.preview, "blob:test-1");
});

test("changing selected face discards a late prediction and clears displayed results", async () => {
  const pending = deferred(); const { session } = sessionFor({ faces: async () => faces(2), predict: () => pending.promise });
  await session.setInput(photo()); session.selectFace(0); const run = session.analyze(); session.selectFace(1);
  pending.resolve(prediction()); await run;
  assert.equal(session.state.selected, 1); assert.equal(session.state.result, null); assert.equal(session.state.busy, null);
});

test("changing method while detecting preserves detection and uses the new method", async () => {
  const pending = deferred(), called = []; const { session } = sessionFor({ faces: () => pending.promise, predict: async (_, method, face) => { called.push([method, face]); return prediction(method, face); } });
  const run = session.setInput(photo()); session.setMethod("all"); pending.resolve(faces()); await run; await session.analyze();
  assert.deepEqual(called, [["all", 0]]); assert.equal(session.state.result.predictions.length, 3);
});

test("changing method cancels old inference without accepting stale results", async () => {
  const pending = deferred(); const { session } = sessionFor({ faces: async () => faces(), predict: () => pending.promise });
  await session.setInput(photo()); const run = session.analyze(); session.setMethod("efficientnetv2_s"); pending.resolve(prediction()); await run;
  assert.equal(session.state.result, null); assert.equal(session.state.method, "efficientnetv2_s"); assert.equal(session.state.busy, null);
});

test("reset revokes preview URLs, forgets image bytes and ignores pending work", async () => {
  const pending = deferred(); const { session, revoked } = sessionFor({ faces: async () => faces(), predict: () => pending.promise });
  await session.setInput(photo()); const run = session.analyze(); session.clear(); pending.resolve(prediction()); await run;
  assert.deepEqual(revoked, ["blob:test-1"]); assert.equal(session.state.blob, null); assert.equal(session.state.result, null); assert.equal(session.state.preview, "");
});

test("invalid image selection clears old results and never calls the API", async () => {
  let calls = 0; const { session, revoked } = sessionFor({ faces: async () => { calls++; return faces(); } });
  await session.setInput(photo()); await session.setInput(new Blob(["svg"], { type: "image/svg+xml" }), "fake.svg");
  assert.equal(calls, 1); assert.equal(session.state.blob, null); assert.equal(session.state.error.code, "UNSUPPORTED_IMAGE"); assert.equal(revoked.length, 1);
  assert.throws(() => validatePhoto({ size: 8388609, type: "image/jpeg" }), /./);
});

test("no-face failure retains a safe preview but never enables a prediction", async () => {
  const { session } = sessionFor({ faces: async () => { throw new AppError("NO_FACE"); } });
  await session.setInput(photo()); assert.equal(session.state.selected, null); assert.equal(session.state.faces.length, 0); assert.equal(session.state.error.code, "NO_FACE"); assert.ok(session.state.preview);
});

test("malformed or mismatched server results are rejected instead of displayed", async () => {
  const { session } = sessionFor({ faces: async () => faces(), predict: async () => prediction("convnextv2_nano", 9) });
  await session.setInput(photo()); await session.analyze(); assert.equal(session.state.error.code, "BAD_RESPONSE"); assert.equal(session.state.result, null);
});

test("formatters preserve absent values and uncalibrated scores are not invented", () => {
  assert.equal(formatPercent(null), "—"); assert.equal(formatPercent(0), "0.00%"); assert.equal(formatTime(undefined), "—"); assert.equal(formatTime(-1), "—");
  assert.equal(explainError(new AppError("CAMERA_DENIED")).code, "CAMERA_DENIED");
});

test("request client sends raw image bytes and exposes typed errors", async () => {
  const blob = photo(); let received;
  await requestJSON("/v1/faces", { method: "POST", body: blob, fetcher: async (_, options) => { received = options; return { ok: true, json: async () => faces() }; } });
  assert.equal(received.body, blob); assert.equal(received.cache, "no-store");
  await assert.rejects(requestJSON("/v1/faces", { fetcher: async () => ({ ok: false, json: async () => ({ error: { code: "NO_FACE", message: "No face" } }) }) }), error => error.code === "NO_FACE");
});

test("request timeout aborts transport and becomes a recoverable timeout", async () => {
  await assert.rejects(requestJSON("/test", { timeout: 5, fetcher: (_, { signal }) => new Promise((_, reject) => signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")))) }), error => error.code === "REQUEST_TIMEOUT");
});

function cameraFixture(mediaDevices) {
  const video = { srcObject: null, videoWidth: 1920, videoHeight: 1080, play: async () => {} };
  let output; const draws = [];
  const canvas = { getContext: () => ({ drawImage: (...args) => draws.push(args) }), toBlob: (callback, type) => { output = new Blob(["captured"], { type }); callback(output); } };
  const camera = new CameraController({ video, mediaDevices, canvasFactory: () => canvas }); return { camera, video, canvas, draws };
}
function stream(id = "front") { const track = { stopped: 0, stop() { this.stopped++; }, getSettings: () => ({ deviceId: id, facingMode: id === "front" ? "user" : "environment" }) }; return { track, getTracks: () => [track], getVideoTracks: () => [track] }; }

test("camera permission denial is recoverable and never retains a stream", async () => {
  const { camera, video } = cameraFixture({ getUserMedia: async () => { throw new DOMException("Denied", "NotAllowedError"); } });
  await camera.start(); assert.equal(camera.state.error.code, "CAMERA_DENIED"); assert.equal(camera.state.phase, "error"); assert.equal(video.srcObject, null);
});

test("stopping while permission is pending closes any late-acquired camera tracks", async () => {
  const pending = deferred(), acquired = stream(); const { camera, video } = cameraFixture({ getUserMedia: () => pending.promise });
  const start = camera.start(); camera.stop(); pending.resolve(acquired); await start;
  assert.equal(acquired.track.stopped, 1); assert.equal(video.srcObject, null); assert.equal(camera.state.phase, "idle");
});

test("capture creates bounded JPEG pixels and stops every video track", async () => {
  const acquired = stream(); const { camera, canvas, video, draws } = cameraFixture({ getUserMedia: async () => acquired, enumerateDevices: async () => [] });
  await camera.start(); const blob = await camera.capture();
  assert.equal(blob.type, "image/jpeg"); assert.equal(canvas.width, 1280); assert.equal(canvas.height, 720); assert.equal(draws.length, 1); assert.equal(acquired.track.stopped, 1); assert.equal(video.srcObject, null); assert.equal(camera.state.phase, "captured");
});

test("camera switching stops the previous device and requests the selected device", async () => {
  const a = stream("front"), b = stream("rear"), constraints = [];
  const { camera } = cameraFixture({ getUserMedia: async value => { constraints.push(value); return constraints.length === 1 ? a : b; }, enumerateDevices: async () => [{ kind: "videoinput", deviceId: "front" }, { kind: "videoinput", deviceId: "rear" }] });
  await camera.start(); await camera.switchCamera(); assert.equal(a.track.stopped, 1); assert.equal(camera.stream, b); assert.equal(constraints[1].audio, false); assert.deepEqual(constraints[1].video.deviceId, { exact: "rear" }); camera.stop(); assert.equal(b.track.stopped, 1);
});

test("a stale canvas capture cannot stop a newly started camera", async () => {
  const a = stream(), b = stream("rear"); let n = 0, finish;
  const { camera, canvas } = cameraFixture({ getUserMedia: async () => ++n === 1 ? a : b, enumerateDevices: async () => [] });
  canvas.toBlob = callback => { finish = callback; };
  await camera.start(); const capture = camera.capture(); await camera.start(); finish(photo()); assert.equal(await capture, null); assert.equal(camera.stream, b); assert.equal(b.track.stopped, 0); camera.stop();
});

test("a request gate invalidates every earlier operation", () => {
  const gate = new RequestGate(), a = gate.start(), b = gate.start(); assert.equal(gate.current(a), false); assert.equal(a.signal.aborted, true); assert.equal(gate.current(b), true); gate.cancel(); assert.equal(gate.current(b), false);
});

test("a lost connection pauses inference and reconnect clears the error without losing the selected face", async () => {
  let connected = false;
  const { session } = sessionFor({ faces: async () => faces(), predict: async () => { if (!connected) throw new AppError("OFFLINE"); return prediction(); } });
  await session.setInput(photo()); await session.analyze();
  assert.equal(session.state.ready, false); assert.equal(session.state.error.code, "OFFLINE"); assert.equal(session.state.selected, 0);
  connected = true; session.ready(true); assert.equal(session.state.error, null);
  await session.analyze(); assert.equal(session.state.result.predictions[0].class_id, 1);
  await assert.rejects(requestJSON("/health", { fetcher: async () => { throw new TypeError("Network failed"); } }), error => error.code === "OFFLINE");
});

test("missing, occupied and unsupported cameras expose recoverable errors", async () => {
  for (const [name, code] of [["NotFoundError", "CAMERA_MISSING"], ["NotReadableError", "CAMERA_BUSY"]]) {
    const { camera, video } = cameraFixture({ getUserMedia: async () => { throw new DOMException("Unavailable", name); } });
    await camera.start(); assert.equal(camera.state.error.code, code); assert.equal(camera.stream, null); assert.equal(video.srcObject, null);
  }
  const { camera } = cameraFixture({}); await camera.start(); assert.equal(camera.state.error.code, "CAMERA_UNAVAILABLE");
});

test("device enumeration failure permits capture, and an ended track shuts down preview", async () => {
  const acquired = stream();
  const { camera, video } = cameraFixture({ getUserMedia: async () => acquired, enumerateDevices: async () => { throw new Error("Unavailable"); } });
  await camera.start(); assert.equal(camera.state.phase, "live"); assert.deepEqual(camera.state.devices, []);
  acquired.track.onended(); assert.equal(camera.state.phase, "error"); assert.equal(camera.state.error.code, "CAMERA_ENDED"); assert.equal(video.srcObject, null); assert.equal(acquired.track.stopped, 1);
});


test("all three model scores retain fixed group order and the highest-score decision", () => {
  const value = prediction().predictions[0];
  const result = scoreBreakdown(value);
  assert.deepEqual(result.classes.map(v => v.class_id), [0, 1, 2]);
  assert.deepEqual(result.classes.map(v => v.score), [.1, .8, .1]);
  assert.equal(result.selected.class_id, 1);
  assert.equal(Object.hasOwn(result, "next"), false);
  value.class_scores[1].age_group = "18-59";
  assert.equal(scoreBreakdown(value).classes.length, 3, "Canonical API hyphen and presentation en dash describe the same class");
  value.class_scores[1].age_group = "60+";
  assert.equal(scoreBreakdown(value), null, "An actual class-name mismatch is rejected");
});

test("malformed or mismatched score distributions cannot reach result cards", () => {
  for (const edit of [v => delete v.class_scores, v => v.class_scores.pop(), v => v.class_scores[0].score = -1,
    v => v.class_scores[1].score = NaN, v => v.class_scores[0].score = .5, v => v.class_id = 0,
    v => v.class_scores.reverse(), v => v.score_type = "calibrated_probability"] ) {
    const value = prediction().predictions[0]; edit(value); assert.equal(scoreBreakdown(value), null);
  }
});

test("score display does not round positive tiny scores to zero or near-one scores to certainty", () => {
  assert.equal(formatModelScore(.6), "60.00%"); assert.equal(formatModelScore(.4), "40.00%");
  assert.equal(formatModelScore(0), "0.00%"); assert.equal(formatModelScore(.000001), "<0.01%");
  assert.equal(formatModelScore(.999999), ">99.99%"); assert.equal(formatModelScore(1), "100.00%");
  for (const value of [null, undefined, NaN, Infinity, -.1, 1.1, "0.6"]) assert.equal(formatModelScore(value), "—");
});

test("analysis rejects a result whose displayed label contradicts its model scores", async () => {
  const value = prediction(); value.predictions[0].class_scores = GROUPS.map((age_group, class_id) => ({age_group, class_id, score:[.6,.4,0][class_id]}));
  const {session} = sessionFor({faces: async () => faces(), predict: async () => value});
  await session.setInput(photo()); await session.analyze();
  assert.equal(session.state.result, null); assert.equal(session.state.error.code, "BAD_RESPONSE");
});


test("recommended-first ordering preserves all three model IDs", () => {
  assert.deepEqual(orderedMethods("swinv2_tiny"), ["swinv2_tiny", "convnextv2_nano", "efficientnetv2_s"]);
  assert.equal(new Set(orderedMethods("convnextv2_nano")).size, 3);
});

test("extra still-image types and untyped extensions pass admission", () => {
  for (const type of ["image/heic", "image/heif", "image/avif", "image/bmp", "image/tiff", "image/jpeg"]) validatePhoto(new Blob(["test"], {type}));
  for (const name of ["a.heic", "a.heif", "a.tiff", "a.avif", "a.jpeg"]) validatePhoto(new Blob(["test"], {type:"application/octet-stream"}), name);
  assert.throws(() => validatePhoto(new Blob(["test"], {type:"image/svg+xml"}), "a.heic"));
});

test("converted previews use browser-compatible pixels while inference keeps original upload bytes", async () => {
  const upload = new Blob(["heic encoded bytes"], {type:"image/heic"}); let preview, analyzed;
  const session = new AnalysisSession({client:{faces:async()=>({...faces(),preview:{content_type:"image/jpeg",width:200,height:200,data_base64:btoa("jpeg bytes")}}),predict:async(blob)=>{analyzed=blob;return prediction();}},createURL:blob=>{preview=blob;return "blob:converted";},revokeURL:()=>{}});
  session.ready(true);await session.setInput(upload,"photo.heic");await session.analyze();
  assert.equal(preview.type,"image/jpeg");assert.equal(await preview.text(),"jpeg bytes");assert.equal(analyzed,upload);assert.ok(session.state.result);
});


test("cloud upload caps are enforced before sending image bytes", () => {
  assert.throws(() => validatePhoto(new Blob([new Uint8Array(33)], {type:"image/jpeg"}), "photo.jpg", 32), error => error.code === "IMAGE_TOO_LARGE" && error.details.maxBytes === 32);
  validatePhoto(new Blob([new Uint8Array(31)], {type:"image/jpeg"}), "photo.jpg", 32);
});

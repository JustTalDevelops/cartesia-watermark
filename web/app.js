"use strict";
const $ = (id) => document.getElementById(id);
const input = $("file-input"), choose = $("choose-file"), dropzone = $("dropzone");
const processButton = $("process"), range = $("strength"), player = $("player");
const originalButton = $("play-original"), processedButton = $("play-processed");
const trialPlayer = $("trial-player");
const methodSelect = $("method");
const storageKey = "recording-tester-last-job";
let file = null, busy = false, jobId = null, generation = 0, timer = null, started = 0;
let playbackVersion = 0, playbackTime = 0, playbackPlaying = false;

function remember(value) {
  try { value ? localStorage.setItem(storageKey, value) : localStorage.removeItem(storageKey); } catch (_) {}
  try {
    const url = new URL(window.location.href);
    value ? url.searchParams.set("job", value) : url.searchParams.delete("job");
    history.replaceState(null, "", url);
  } catch (_) {}
}
function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}
function status(message) {
  $("activity").hidden = false;
  if ($("status").textContent !== message) $("status").textContent = message;
}
function setBusy(value) {
  busy = value;
  choose.disabled = value;
  range.disabled = value;
  methodSelect.disabled = value;
  input.disabled = value;
  processButton.disabled = value || !file;
  $("another").disabled = value;
  $("delete").disabled = value;
  processButton.textContent = value ? "Processing…" : "Process recording";
  if (!value) { clearInterval(timer); timer = null; }
}
function startTimer() {
  started = Date.now();
  clearInterval(timer);
  $("elapsed").textContent = "";
  timer = setInterval(() => { $("elapsed").textContent = `${Math.floor((Date.now() - started) / 1000)}s`; }, 1000);
}
function clearPlayer() {
  playbackVersion++;
  player.pause();
  player.removeAttribute("src");
  player.load();
  trialPlayer.pause();
  trialPlayer.removeAttribute("src");
  trialPlayer.load();
  $("trial").hidden = true;
  playbackTime = 0;
  playbackPlaying = false;
  $("result").hidden = true;
}
function selectFile(selected) {
  if (busy || !selected) return;
  $("error").hidden = true;
  if (!selected.size) { showError("This file is empty. Choose another recording."); return; }
  if (selected.size > 100 * 1024 * 1024) { showError("This file is over 100 MB. Choose a smaller recording."); return; }
  generation++;
  jobId = null;
  remember(null);
  clearPlayer();
  $("activity").hidden = true;
  file = selected;
  $("file-title").textContent = selected.name;
  $("file-description").textContent = `${(selected.size / (1024 * 1024)).toFixed(1)} MB · Ready to process`;
  choose.textContent = "Choose a different file";
  processButton.disabled = false;
}
choose.addEventListener("click", () => { input.value = ""; input.click(); });
input.addEventListener("change", () => selectFile(input.files[0]));
for (const type of ["dragenter", "dragover"]) dropzone.addEventListener(type, (event) => {
  event.preventDefault();
  if (!busy) dropzone.classList.add("dragging");
});
for (const type of ["dragleave", "drop"]) dropzone.addEventListener(type, (event) => {
  event.preventDefault(); dropzone.classList.remove("dragging");
});
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
range.addEventListener("input", () => { $("strength-value").value = `${range.value}%`; });
methodSelect.addEventListener("change", () => {
  const reconstruction = methodSelect.value !== "editor";
  $("editor-settings").hidden = reconstruction;
  $("method-help").textContent = methodSelect.value === "reconstruct_strong"
    ? "For persistent watermarks. Listen for changes in the words and voice. Allow about 2–3 minutes per minute of audio."
    : reconstruction
    ? "Rebuilds speech using the approved reconstruction method. Allow about 2–3 minutes per minute of audio."
    : "Makes small edits with the earlier V3 model. This is faster, but remained detectable on both test recordings.";
});

function responseMessage(text, fallback) {
  try { const value = JSON.parse(text).detail; if (typeof value === "string") return value; } catch (_) {}
  return fallback;
}
function upload() {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const query = new URLSearchParams({filename: file.name, method: methodSelect.value,
      strength: methodSelect.value === "editor" ? String(Number(range.value) / 100) : "1"});
    request.open("POST", `/api/jobs?${query}`);
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.setRequestHeader("X-Requested-With", "recording-tester");
    request.timeout = 10 * 60 * 1000;
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        $("upload-progress").hidden = false;
        $("upload-progress").value = event.loaded / event.total * 100;
        status(`Uploading your recording… ${Math.round(event.loaded / event.total * 100)}%`);
      }
    };
    request.onload = () => {
      if (request.status !== 202) return reject(new Error(responseMessage(request.responseText, "Upload failed. Try your recording again.")));
      try { resolve(JSON.parse(request.responseText).id); } catch (_) { reject(new Error("Upload response was interrupted. Try again.")); }
    };
    request.onerror = () => reject(new Error("The connection was interrupted. Check Tailscale and try again."));
    request.ontimeout = () => reject(new Error("The upload took too long. Try a smaller recording."));
    request.send(file);
  });
}
function showResult(row, focus) {
  setBusy(false);
  $("activity").hidden = true;
  $("result").hidden = false;
  $("result-name").textContent = row.filename;
  const minutes = Math.floor(row.seconds / 60), seconds = Math.floor(row.seconds % 60).toString().padStart(2, "0");
  const methodName = row.method === "reconstruct_strong" ? "Stronger reconstruction" : row.method === "reconstruct" ? "Reconstruction" : "Small edits";
  $("metadata").textContent = `${minutes}:${seconds} · ${row.sample_rate / 1000} kHz · ${row.channels === 1 ? "Mono" : "Stereo"} · ${methodName}`;
  const verification = $("verification");
  verification.className = "help";
  if (row.verification && row.verification.detected === true) {
    verification.textContent = "Cartesia check: watermark still detected.";
    verification.className = "error";
  } else if (row.verification && row.verification.detected === false) {
    verification.textContent = "Cartesia check: watermark not detected.";
  } else {
    verification.textContent = "Watermark detection has not been checked.";
  }
  originalButton.setAttribute("aria-pressed", "true");
  processedButton.setAttribute("aria-pressed", "false");
  player.src = `/api/jobs/${row.id}/audio/original`;
  player.setAttribute("aria-label", "Original recording");
  $("download").href = `/api/jobs/${row.id}/audio/download`;
  $("trial").hidden = !row.trial;
  if (row.trial) {
    $("trial-quality").textContent = row.trial.quality;
    trialPlayer.src = `/api/jobs/${row.id}/audio/trial-preview`;
    $("trial-download").href = `/api/jobs/${row.id}/audio/trial`;
  }
  if (focus) $("result-heading").focus({preventScroll: false});
}
async function poll(id, version, focus = true) {
  let failures = 0;
  while (version === generation) {
    let response, row;
    try {
      response = await fetch(`/api/jobs/${id}`, {cache: "no-store"});
      row = await response.json();
    } catch (_) {
      if (++failures >= 15) throw new Error("The connection was interrupted. Refresh this page to reconnect to your recording.");
      status("Reconnecting to your recording…");
      await new Promise((resolve) => setTimeout(resolve, 2000));
      continue;
    }
    if (version !== generation) return;
    if (!response.ok) {
      if (response.status === 404) remember(null);
      throw new Error(typeof row.detail === "string" ? row.detail : "Could not open this recording. Try uploading it again.");
    }
    failures = 0;
    if (row.state === "done") { showResult(row, focus); return; }
    if (row.state === "error") { remember(null); throw new Error(row.message); }
    status(row.message);
    await new Promise((resolve) => setTimeout(resolve, 900));
  }
}
processButton.addEventListener("click", async () => {
  if (!file || busy) return;
  const version = ++generation;
  clearPlayer();
  $("error").hidden = true;
  $("upload-progress").value = 0;
  setBusy(true); startTimer(); status("Uploading your recording…");
  try {
    jobId = await upload(); remember(jobId);
    $("upload-progress").hidden = true;
    status("Your recording is queued.");
    await poll(jobId, version);
  } catch (error) {
    if (version !== generation) return;
    setBusy(false); $("activity").hidden = true; showError(error.message);
  }
});
function switchAudio(kind) {
  const button = kind === "original" ? originalButton : processedButton;
  if (button.getAttribute("aria-pressed") === "true" || !jobId) return;
  const version = ++playbackVersion;
  if (player.readyState >= 1) { playbackTime = player.currentTime; playbackPlaying = !player.paused; }
  const time = playbackTime, playing = playbackPlaying;
  player.pause();
  player.src = `/api/jobs/${jobId}/audio/${kind}`;
  player.setAttribute("aria-label", kind === "original" ? "Original recording" : "Processed recording");
  originalButton.setAttribute("aria-pressed", String(kind === "original"));
  processedButton.setAttribute("aria-pressed", String(kind === "processed"));
  player.addEventListener("loadedmetadata", () => {
    if (version !== playbackVersion) return;
    player.currentTime = Math.min(time, Math.max(0, player.duration - .01));
    if (playing) player.play().catch(() => {});
  }, {once: true});
}
originalButton.addEventListener("click", () => switchAudio("original"));
processedButton.addEventListener("click", () => switchAudio("processed"));
player.addEventListener("play", () => trialPlayer.pause());
trialPlayer.addEventListener("play", () => player.pause());
trialPlayer.addEventListener("error", () => {
  if (trialPlayer.getAttribute("src")) showError("The listening trial could not load. Try refreshing the page.");
});
player.addEventListener("error", () => {
  if (player.getAttribute("src")) showError("Audio playback could not load. Try refreshing, or download the WAV to listen.");
});
function reset() {
  if (busy) return;
  generation++; file = null; jobId = null; remember(null); clearPlayer();
  input.value = ""; $("error").hidden = true; $("activity").hidden = true;
  $("file-title").textContent = "Drop a recording here";
  $("file-description").textContent = "or choose one from your device";
  choose.textContent = "Choose audio file"; processButton.disabled = true; choose.focus();
}
$("another").addEventListener("click", reset);
$("delete").addEventListener("click", async () => {
  if (!jobId || busy) return;
  $("delete").disabled = true;
  try {
    const response = await fetch(`/api/jobs/${jobId}`, {method: "DELETE", headers: {"X-Requested-With": "recording-tester"}});
    if (!response.ok && response.status !== 404) throw new Error("Files could not be deleted. Try again.");
    reset();
  } catch (error) { showError(error.message); }
  finally { $("delete").disabled = false; }
});
(async function restore() {
  const linkedJob = new URLSearchParams(window.location.search).get("job");
  let saved = linkedJob;
  if (!saved) { try { saved = localStorage.getItem(storageKey); } catch (_) { return; } }
  if (!saved || !/^[a-f0-9]{32}$/.test(saved)) return;
  remember(saved);
  jobId = saved;
  const version = ++generation;
  setBusy(true); startTimer(); status("Opening your last recording…");
  try { await poll(saved, version, Boolean(linkedJob)); }
  catch (error) { setBusy(false); $("activity").hidden = true; showError(error.message); }
})();

/* MOM Server frontend - vanilla JS, zero dependencies, works fully offline.
   Style rule: JS strings use double quotes; HTML built in JS uses single-quoted
   attributes so the two never collide. */
"use strict";

const STAGES = ["queued", "decoding", "diarizing", "transcribing", "aligning", "summarizing", "completed"];
const AVATAR_COLORS = ["#4f8cff", "#e5484d", "#2fbf71", "#f5a623", "#9a6bff", "#00b8d4", "#ff7096"];

const $ = (id) => document.getElementById(id);
let apiKey = localStorage.getItem("mom_api_key") || "";
let pollTimer = null;
let currentMeetingId = null;
let mediaRecorder = null;
let recordChunks = [];
let recordTimer = null;
let recordSeconds = 0;

const ICON_OK = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
const ICON_ERROR = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
const ICON_INFO = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';

/* ---------------- API helper ---------------- */
async function api(path, options = {}) {
  const headers = Object.assign({ "X-API-Key": apiKey }, options.headers || {});
  const response = await fetch(path, Object.assign({}, options, { headers }));
  if (response.status === 401) {
    setStatusPill("bad", "invalid API key");
    throw new Error("Invalid API key");
  }
  return response;
}

/* ---------------- toasts ---------------- */
function toast(message, kind = "info", timeoutMs = 3800) {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  const icon = kind === "success" ? ICON_OK : kind === "error" ? ICON_ERROR : ICON_INFO;
  const color = kind === "success" ? "var(--ok)" : kind === "error" ? "var(--danger)" : "var(--accent)";
  el.innerHTML = "<span class='t-icon' style='color:" + color + "'>" + icon + "</span><span></span>";
  el.lastElementChild.textContent = message;
  $("toast-stack").appendChild(el);
  setTimeout(() => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 260);
  }, timeoutMs);
}

/* ---------------- modal ---------------- */
function confirmModal(title, body) {
  return new Promise((resolve) => {
    $("modal-title").textContent = title;
    $("modal-body").textContent = body;
    $("modal-backdrop").classList.remove("hidden");
    const done = (answer) => {
      $("modal-backdrop").classList.add("hidden");
      resolve(answer);
    };
    $("modal-confirm-btn").onclick = () => done(true);
    $("modal-cancel-btn").onclick = () => done(false);
  });
}

/* ---------------- status pill ---------------- */
function setStatusPill(kind, text, detail) {
  const el = $("engine-status");
  el.className = "status-pill status-" + kind;
  $("engine-status-text").textContent = text;
  el.title = detail || text;
}

async function refreshEngineStatus() {
  if (!apiKey) { setStatusPill("unknown", "not connected"); return; }
  try {
    const r = await api("/api/system/status");
    const s = await r.json();
    const problems = [];
    if (!s.ffmpeg_available) problems.push("ffmpeg missing");
    if (!s.ollama_reachable) problems.push("ollama down");
    else if (!s.ollama_model_available) problems.push("LLM model missing");
    if (!s.diarization_configured && !s.allow_no_diarization) problems.push("HF_TOKEN missing");
    const detail =
      "ffmpeg: " + (s.ffmpeg_available ? "ok" : "missing") +
      " | ollama model: " + s.ollama_model +
      " | whisper: " + s.whisper_model +
      " | diarization: " + (s.diarization_configured ? s.diarization_model : "off");
    if (problems.length === 0) {
      setStatusPill(
        "ok",
        s.diarization_configured ? "engines ready" : "ready (speaker ID off)",
        detail
      );
    } else {
      setStatusPill("warn", "degraded: " + problems.join(", "), detail);
    }
  } catch (_e) {
    setStatusPill("bad", "server unreachable");
  }
}

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tabs").forEach((group) => {
  group.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      group.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const cardBody = group.parentElement;
      cardBody.querySelectorAll(".tab-body").forEach((body) => body.classList.remove("active"));
      cardBody.querySelector("#tab-" + btn.dataset.tab).classList.add("active");
    });
  });
});

/* ---------------- recording ---------------- */
$("rec-start-btn").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => recordChunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(recordChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      const player = $("rec-playback");
      player.src = URL.createObjectURL(blob);
      player.style.display = "block";
      uploadBlob(blob, "browser-recording-" + Date.now() + ".webm");
    };
    mediaRecorder.start();
    $("rec-start-btn").disabled = true;
    $("rec-start-btn").classList.add("recording");
    $("rec-stop-btn").disabled = false;
    recordSeconds = 0;
    recordTimer = setInterval(() => {
      recordSeconds += 1;
      const mm = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
      const ss = String(recordSeconds % 60).padStart(2, "0");
      $("rec-timer").textContent = mm + ":" + ss;
    }, 1000);
  } catch (err) {
    toast("Microphone access denied: " + err.message, "error");
  }
});

$("rec-stop-btn").addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  clearInterval(recordTimer);
  $("rec-start-btn").disabled = false;
  $("rec-start-btn").classList.remove("recording");
  $("rec-stop-btn").disabled = true;
});

/* ---------------- upload ---------------- */
const dropZone = $("drop-zone");
dropZone.addEventListener("click", () => $("file-input").click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); }
});
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
$("file-input").addEventListener("change", (e) => {
  if (e.target.files.length) uploadFile(e.target.files[0]);
});

function uploadFile(file) {
  const form = new FormData();
  form.append("file", file, file.name);
  sendUpload(form, file.name);
}

function uploadBlob(blob, filename) {
  const form = new FormData();
  form.append("file", blob, filename);
  sendUpload(form, filename);
}

async function sendUpload(form, filename) {
  try {
    const r = await api("/api/meetings/upload", { method: "POST", body: form });
    if (r.status === 401) return;
    if (!r.ok) {
      const detail = await r.json().catch(() => ({ detail: r.statusText }));
      toast("Upload failed: " + (detail.detail || r.statusText), "error", 6000);
      return;
    }
    const body = await r.json();
    $("progress-card").classList.remove("hidden");
    $("proc-filename").textContent = "(" + filename + ")";
    startPolling(body.job_id);
    refreshHistory();
    toast("Upload accepted - processing started", "success");
  } catch (err) {
    toast("Upload error: " + err.message, "error");
  }
}

/* ---------------- polling + progress ---------------- */
function renderStages(currentStage, failed) {
  const idx = STAGES.indexOf(currentStage);
  $("stage-chips").innerHTML = STAGES.map((stage, i) => {
    let cls = "chip";
    let mark = "";
    if (failed) {
      if (stage === currentStage) cls += " failed";
      else if (i < idx) { cls += " done"; mark = ICON_OK; }
    } else {
      if (i < idx) { cls += " done"; mark = ICON_OK; }
      if (stage === currentStage && stage !== "completed") cls += " active";
    }
    return "<span class='" + cls + "'>" + mark + stage + "</span>";
  }).join("");
}

function updateProgress(job) {
  const fill = $("progress-fill");
  $("progress-percent").textContent = job.progress + "%";
  fill.style.width = job.progress + "%";
  fill.classList.toggle("indeterminate", job.progress <= 4 && job.status !== "completed" && job.status !== "failed");
  $("progress-message").textContent = job.stage_message || "";
  renderStages(job.stage, job.status === "failed");
}

function startPolling(jobId) {
  stopPolling();
  pollTimer = setInterval(async () => {
    try {
      const r = await api("/api/jobs/" + jobId);
      if (!r.ok) { stopPolling(); return; }
      const job = await r.json();
      updateProgress(job);
      if (job.status === "completed") {
        stopPolling();
        loadResults(job.job_id, true);
        refreshHistory();
        refreshEngineStatus();
        toast("Meeting processed successfully", "success");
      } else if (job.status === "failed") {
        stopPolling();
        refreshHistory();
        toast("Processing failed - see progress card for details", "error", 7000);
      }
    } catch (_e) { /* transient network blip - keep polling */ }
  }, 1500);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

/* ---------------- results ---------------- */
async function loadResults(meetingId, scroll) {
  try {
    const [tRes, mRes, sRes] = await Promise.all([
      api("/api/meetings/" + meetingId + "/transcript"),
      api("/api/meetings/" + meetingId + "/minutes"),
      api("/api/meetings/" + meetingId + "/speakers"),
    ]);
    if (tRes.ok) $("transcript-view").textContent = await tRes.text();
    if (mRes.ok) {
      const md = await mRes.text();
      $("minutes-view").innerHTML = renderMarkdown(md);
      window.__currentMinutes = md;
    }
    currentMeetingId = meetingId;
    $("results-card").dataset.meetingId = meetingId;
    if (sRes.ok) renderRenamePanel(await sRes.json());
    $("results-card").classList.remove("hidden");
    if (scroll) $("results-card").scrollIntoView({ behavior: "smooth" });
  } catch (_e) { /* ignore */ }
}

/* ---------------- speaker renaming ---------------- */
function renderRenamePanel(data) {
  const speakers = data.speakers || [];
  const rows = $("rename-rows");
  const badge = $("speaker-count-badge");
  rows.innerHTML = "";
  badge.textContent = speakers.length || "";
  badge.classList.toggle("hidden", speakers.length === 0);

  const hasDiarization = speakers.length > 0;
  $("no-speakers-note").classList.toggle("hidden", hasDiarization);
  $("rename-panel").classList.toggle("hidden", !hasDiarization);
  $("rename-status").textContent = "";
  if (!hasDiarization) return;

  speakers.forEach((s) => {
    const color = AVATAR_COLORS[(s.speaker_number - 1) % AVATAR_COLORS.length];
    const row = document.createElement("div");
    row.className = "rename-row";
    row.innerHTML =
      "<span class='avatar' style='background:" + color + "'>" + s.speaker_number + "</span>" +
      "<input type='text' data-speaker='" + s.speaker_number + "' placeholder='Speaker " +
      s.speaker_number + "' maxlength='80' />" +
      "<span class='speaker-label'>rename</span>" +
      (s.sample_quote ? "<p class='quote'></p>" : "");
    row.querySelector("input").value = s.name || "";
    if (s.sample_quote) row.querySelector(".quote").textContent = '"' + s.sample_quote + '"';
    rows.appendChild(row);
  });
}

async function saveNames() {
  if (!currentMeetingId) return;
  const names = {};
  document.querySelectorAll("#rename-rows input").forEach((inp) => {
    names[inp.dataset.speaker] = inp.value.trim();
  });
  const btn = $("save-names-btn");
  btn.classList.add("loading");
  try {
    const r = await api("/api/meetings/" + currentMeetingId + "/speakers", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });
    if (!r.ok) throw new Error("server rejected the update");
    const [tRes, mRes] = await Promise.all([
      api("/api/meetings/" + currentMeetingId + "/transcript"),
      api("/api/meetings/" + currentMeetingId + "/minutes"),
    ]);
    if (tRes.ok) $("transcript-view").textContent = await tRes.text();
    if (mRes.ok) {
      const md = await mRes.text();
      $("minutes-view").innerHTML = renderMarkdown(md);
      window.__currentMinutes = md;
    }
    $("rename-status").textContent = "saved";
    toast("Speaker names saved - transcript, minutes and downloads updated", "success");
  } catch (err) {
    toast("Could not save names: " + err.message, "error");
  } finally {
    btn.classList.remove("loading");
  }
}

/* ---------------- downloads + copy ---------------- */
$("dl-transcript-btn").addEventListener("click", () => downloadText(
  $("transcript-view").textContent,
  "transcript_" + ($("results-card").dataset.meetingId || "") + ".txt"
));
$("dl-minutes-btn").addEventListener("click", () => downloadText(
  window.__currentMinutes || "",
  "minutes_" + ($("results-card").dataset.meetingId || "") + ".md"
));
$("copy-minutes-btn").addEventListener("click", async () => {
  const text = window.__currentMinutes || "";
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    toast("Minutes copied to clipboard", "success");
  } catch (_e) {
    downloadText(text, "minutes_clipboard.md");
  }
});

function downloadText(text, filename) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* Minimal markdown renderer (headings, bold, lists, code, hr).
   Dependency-free on purpose: the UI must work on an air-gapped LAN. */
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function inlineMd(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}
function renderMarkdown(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "";
  let listType = null;
  const closeList = () => {
    if (listType) { html += "</" + listType + ">"; listType = null; }
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^```/.test(line)) { closeList(); continue; }
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) { closeList(); html += "<h" + h[1].length + ">" + inlineMd(h[2]) + "</h" + h[1].length + ">"; continue; }
    const ul = line.match(/^\s*[-*]\s+(.*)/);
    if (ul) {
      if (listType !== "ul") { closeList(); html += "<ul>"; listType = "ul"; }
      html += "<li>" + inlineMd(ul[1]) + "</li>"; continue;
    }
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (ol) {
      if (listType !== "ol") { closeList(); html += "<ol>"; listType = "ol"; }
      html += "<li>" + inlineMd(ol[1]) + "</li>"; continue;
    }
    if (/^---+$/.test(line)) { closeList(); html += "<hr>"; continue; }
    if (line.trim() === "") { closeList(); continue; }
    closeList();
    html += "<p>" + inlineMd(line) + "</p>";
  }
  closeList();
  return html;
}

/* ---------------- history ---------------- */
function relativeTime(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return mins + " min ago";
  const hours = Math.floor(mins / 60);
  if (hours < 24) return hours + " h ago";
  return new Date(iso).toLocaleString();
}

async function refreshHistory() {
  const tbody = $("history-body");
  try {
    const r = await api("/api/meetings");
    if (!r.ok) return;
    const meetings = await r.json();
    if (!meetings.length) {
      tbody.innerHTML = "<tr><td colspan='7' class='muted'>No meetings yet - record or upload audio above</td></tr>";
      return;
    }
    tbody.innerHTML = "";
    for (const m of meetings) {
      const tr = document.createElement("tr");
      tr.className = "clickable";
      const badge =
        m.status === "completed" ? "completed" :
        m.status === "failed" ? "failed" : "running";
      tr.title = new Date(m.created_at).toLocaleString();
      tr.innerHTML =
        "<td>" + relativeTime(m.created_at) + "</td>" +
        "<td>" + escapeHtml(m.original_filename) + "</td>" +
        "<td><span class='badge " + badge + "'>" + m.status + "</span></td>" +
        "<td>" + (m.duration_sec ? Math.round(m.duration_sec) + "s" : "-") + "</td>" +
        "<td>" + (m.num_speakers || "-") + "</td>" +
        "<td>" + (m.processing_ms ? Math.round(m.processing_ms / 1000) + "s" : "-") + "</td>" +
        "<td><button class='link-btn' title='Delete meeting'>delete</button></td>";
      tr.addEventListener("click", () => loadResults(m.id, true));
      tr.querySelector("button").addEventListener("click", async (e) => {
        e.stopPropagation();
        const yes = await confirmModal(
          "Delete this meeting?",
          '"' + m.original_filename + '" and its transcript, minutes and audio files will be removed permanently.'
        );
        if (!yes) return;
        await api("/api/meetings/" + m.id, { method: "DELETE" });
        toast("Meeting deleted", "info");
        refreshHistory();
      });
      tbody.appendChild(tr);
    }
  } catch (_e) {
    tbody.innerHTML = "<tr><td colspan='7' class='muted'>Server unreachable</td></tr>";
  }
}

/* ---------------- boot ---------------- */
$("save-key-btn").addEventListener("click", () => {
  apiKey = $("api-key-input").value.trim();
  localStorage.setItem("mom_api_key", apiKey);
  refreshEngineStatus();
  refreshHistory();
  if (apiKey) toast("Connected. Key stored in this browser only.", "success");
});
$("refresh-history-btn").addEventListener("click", refreshHistory);
$("save-names-btn").addEventListener("click", saveNames);

if (apiKey) {
  $("api-key-input").value = apiKey;
  refreshEngineStatus();
  refreshHistory();
} else {
  $("history-body").innerHTML = "<tr><td colspan='7' class='muted'>Enter your API key above to begin</td></tr>";
}

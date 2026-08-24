# MOM Rebuild Log

The complete story of rebuilding MOM-device from a Raspberry Pi prototype into an organization-ready meeting-minutes server. Written to be retold: every phase explains *what* was done, *why*, and *what changed compared to before*. Decisions are recorded inline in bold DECISION blocks; problems that occurred live in [PROBLEMS.md](PROBLEMS.md).

---

## Phase 0 — Where We Started (the honest assessment)

**What existed:** a Streamlit script (`src/final.py, 782 lines) that recorded microphone audio through sounddevice, denoised it, sent it to Picovoice Falcon for diarization, shelled out to `whisper-cli.exe` once per speaker segment, and asked a local Ollama model (`deepseek-r1:1.5b) to write minutes. Built for a Raspberry Pi; paths hardcoded to two different machines; a live API key committed to git; `app.py broken mid-function; empty `whisper.cpp/llama.cpp` folders.

**Why it couldn't scale to an organization:

| Problem | Consequence |
|---|---|
| Streamlit execution model (full rerun per interaction) | One user's refresh kills another's state |
| Subprocess-per-segment transcription | Minutes of process-spawn overhead; temp-file juggling; fragile on hour-long meetings |
| Session-only storage | Server restart loses everything; no meeting history |
| Hardcoded absolute paths | Broke on any machine except two specific ones |
| Secrets in source | Key permanently compromised in git history |
| No APIs | Nothing can integrate with it programmatically |
| Tiny Pi-era models | ~14–22% WER class STT; 1.5B reasoning model writing prose |

**Knowledge-graph evidence (graphify):** god nodes were exactly the monolith entry points (`transcribe_with_diarization, generate_meeting_minutes), and community cohesion scored 0.14–0.21 — independent confirmation that the code was one tangled blob pretending to be modules.

---

## Phase 1 — Knowledge Graph (graphify)

**Done: installed `graphifyy CLI, registered project skill, excluded data artifacts via `.graphifyignore`, built `graphify-out/ (84 nodes · 91 edges · 21 communities) with zero cloud cost — AST extraction is local; the README semantic pass ran on our own local Ollama.

**DECISION — graphify over manual documentation:** the graph regenerates (`graphify update . costs nothing after code changes), so architecture knowledge never goes stale the way wikis do.

Three real problems occurred and were fixed here (duplicate installs shadowing PATH, missing optional extra, wrong default local model) — see PROBLEMS.md P1–P3.

---

## Phase 2 — PROJECT_GUIDE.md

**Done: wrote `docs/PROJECT_GUIDE.md — the beginner-friendly master document covering the pipeline story, all technology decisions with benchmark tables (STT WER, diarization DER), hardware fit with VRAM budget math, target architecture, API surface, security posture, glossary, roadmap.

**Key benchmarks that drove model selection (Aug 2026 leaderboards):

- STT chosen: Whisper large-v3-turbo (MIT, 99 languages incl. Hindi, ~7.75% avg WER, ~2–3 GB VRAM int8, native word timestamps). NVIDIA Canary-Qwen tops English leaderboards (~5.63%) but is English-only — disqualifying for Indian-accent English + Hinglish meetings. Parakeet TDT is fastest but covers only 25 European languages.
- Diarization chosen: pyannote community-1 (MIT, AMI-meetings DER 17.0% vs 18.8% for 3.1) with 3.1 as config fallback. Replaced Picovoice Falcon (closed, quota-metered, committed key incident).
- MoM LLM: qwen2.5:7b via Ollama (upgraded from deepseek-r1:1.5b), switchable by config.

---

## Phase 3 — Repo Hygiene

**Done: deleted broken `src/app.py; moved working logic to `legacy/ for porting reference; removed 33 test recordings (12.7 MB) and 46 generated outputs; deleted empty `whisper.cpp//llama.cpp dirs; kept exactly one real recording as `tests/fixtures/sample_meeting.wav for end-to-end testing; replaced `.gitignore with a proper one (secrets, runtime data, model binaries); removed Claude-specific tooling folder after confirming it was unused in our workflow.

**DECISION — rotate, don't rewrite history:** the leaked Picovoice key stays in git history; rewriting history is destructive and pointless once the key is revoked at the provider.

---

## Phase 4 — Backend Build

*(details appended below as implementation completes)*

### Architecture implemented

```
backend/app/
├── main.py         FastAPI app: startup wiring, CORS, static UI mount
├── config.py       pydantic-settings; every knob via .env
├── security.py     X-API-Key / Bearer check (constant-time compare)
├── db.py           SQLAlchemy engine/session (SQLite)
├── models.py       Meeting table = meeting + embedded job state
├── schemas.py      Pydantic response contracts
├── worker.py       queue.Queue + thread pool (GPU-serialized), staged pipeline
└── services/
    ├── audio.py          ffmpeg decode → 16 kHz mono WAV + ffprobe duration
    ├── transcription.py  faster-whisper lazy singleton, word timestamps, Silero VAD
    ├── diarization.py    pyannote lazy singleton, community-1 → 3.1 fallback,
    │                     turn building with legacy merge-gap logic ported
    ├── alignment.py      pure functions: words × turns → labeled transcript
    └── minutes.py        Ollama batch → per-batch summaries → synthesis (ported)

### Key decisions

- **One table (Meeting) doubles as job record.** A separate jobs table would duplicate ids and joins for zero benefit at this scale; status/stage/progress live on the meeting row.
- **Worker concurrency defaults to 1.** The GPU is the bottleneck resource; serializing jobs keeps latency predictable and VRAM safe. Concurrency >1 remains configurable.
- **Heavy ML libraries imported lazily inside service functions.** The API boots even if torch is absent; `/api/system/status reports engine readiness honestly instead of crashing at import time. Also makes unit tests instant (no model loads).
- **Graceful degradation without diarization.** If HF_TOKEN is missing or gated-model access fails, the pipeline still produces a timestamped transcript and minutes — with a clear warning surfaced in job messages. Meetings are too valuable to lose entirely to one optional component.

---

## Phase 5 — API Verification (batch-by-batch, human-in-the-loop)

Instead of coding blind, every layer was verified through its public API before moving on. All commands below are plain `curl.exe` against the running server.

### Batch 1 — boot & auth

| Check | Result |
|---|---|
| `GET /api/health` (no auth) | `{"status":"ok"}` |
| `GET /api/system/status` without key | **401** after fix (was 200 initially - see PROBLEMS P9) |
| same with key | full engine JSON: ffmpeg ok, ollama ok + models listed, whisper/pyannote config shown |

### Batch 2 — full pipeline on a real recording (`tests/fixtures/sample_meeting.wav`, 20 s)

Two uploads ran back-to-back, accidentally proving the queue and the cache:

| Run | Processing time | Speedup | Notes |
|---|---|---|---|
| Cold (model downloaded inside the job) | 387.1 s | 0.05x | the misleading "diarizing" label era - see P6 |
| Warm (model pre-cached via prewarm script) | 6.58 s | **3.04x** | fixed costs dominate on 20 s audio; long meetings amortize them |

Transcript quality: word-perfect on the sample including Indian-accent English; correct `[MM:SS]` timestamps; language auto-detected `en`. Minutes structure from qwen2.5:3b: all six required sections present, no hallucinated decisions ("No decisions were made" honestly stated).

Known small-model quirk recorded: "using llama" got attributed as *Llama used for transcription* - acceptable at 3B scale; upgrading OLLAMA_MODEL to qwen2.5:7b is config-only.

### Batch 3 — error paths & lifecycle

| Check | Result |
|---|---|
| `DELETE /api/meetings/{id}` | 204; meeting vanished from list |
| Upload `.env.example` (bad extension) | 415 + allowed-list message |
| Poll nonexistent job id | 404 |

---

## Phase 7 — End-to-End Verification Results (live numbers)

Hardware during tests: Intel Ultra 9 185H, RTX 2000 Ada 8 GB, faster-whisper large-v3-turbo int8_float16 on CUDA, qwen2.5:3b via local Ollama, WORKER_CONCURRENCY=1.

- Pipeline stages exercised: decode(ffmpeg) -> diarization(skip w/o token) -> transcribe(word timestamps, VAD) -> align -> LLM minutes -> persist(SQLite + files)
- Storage verified: uploads/, outputs/, mom.db populated under data/; DELETE removes rows AND files
- Auth: constant-time key compare; health endpoint intentionally public

### Diarization verification (Day 2 session)

The HF-token-enabled run surfaced three stacked pyannote 4.x incompatibilities (full details in [PROBLEMS.md](PROBLEMS.md) P10–P12), each fixed between upload-poll cycles:

| # | Problem | Fix |
|---|---|---|
| P10 | `use_auth_token` rejected by pyannote 4 | renamed kwarg to `token`; latent missing `import torch` found in same function |
| P11 | torchcodec broken on Windows + torch 2.5.1+cu121 | decode normalized WAV via scipy.io.wavfile, pass waveform dict to pipeline |
| P12 | community-1 returns `DiarizeOutput`, not `Annotation` | unwrap via `getattr(output, "speaker_diarization", output)` |

Graceful degradation proved its worth throughout: every failed attempt still delivered full transcript + minutes, with the skip reason visible in job messages.

**Final verified run** (20 s fixture, models cached): **38 s processing (0.53x realtime), diarization ACTIVE**. Transcript:

```
[00:00 - 00:05] Speaker 1: We are trying to transcribe our audios using whisper and for dialization...
[00:09 - 00:18] Speaker 2: We are testing audio input and we are using falcon for dialization...
```

Two speakers cleanly separated, correct attribution, no hallucinated turns.

### Long-meeting benchmark (Day 2 session)

Real AMI corpus meeting (ES2002a, four-speaker design kick-off, 21.2 min / 1273 s, single-channel headset mix):

| Metric | Value |
|---|---|
| Audio duration | 1273 s |
| End-to-end processing | 234 s → **5.44x realtime** |
| LLM batches | 6 (map-reduce summarization engaged as designed) |
| Speaker labels produced | exactly 4, correct role attribution (PM = dominant Speaker 1, three intros = distinct voices) |
| Errors | none |

Context: the legacy Pi-era setup promised 2–5 minutes of processing *per minute* of audio (~0.2–0.4x realtime). The rebuilt pipeline is roughly an order of magnitude faster while producing strictly more output (speaker-labeled transcript + structured minutes).

**Backend sign-off: complete** — every pipeline stage verified through its public APIs, including on real multi-speaker meeting audio.

---

## Phase 8 — Web UI & Speaker Naming (Day 2 session)

### UI rebuild

The drafted `frontend/` was redesigned to production quality: sticky header with live engine-status pill, step-numbered cards, a full button system (primary/danger/ghost/subtle/loading states), toast notifications replacing browser alerts, a custom delete-confirmation modal, copy-to-clipboard, keyboard-accessible drop zone and focus-visible rings, indeterminate progress shimmer while jobs queue. All dependency-free (Decision 5 holds: air-gapped LAN rendering).

### Speaker naming shipped (Option A of Decision 11)

- **Storage:** `speaker_names_json` on the meetings table; idempotent SQLite migration added to `init_db` (verified against a copy of the live database - all prior meetings preserved).
- **API:** `GET/PATCH /api/meetings/{id}/speakers`. The GET returns each detected speaker with their longest utterance as an identification quote; the PATCH validates, strips, caps at 80 chars.
- **Serve-time substitution:** stored transcripts/minutes always keep generic `Speaker N`; names are substituted when serving transcript/minutes/detail. Renaming is instant, reversible, never touches the GPU pipeline, and downloads inherit named output.
- **UI:** Speakers tab with colored avatar per speaker, sample quotes as hints, Save button with loading state; views refresh after save.
- **Tests:** new suite in `backend/tests/test_speakers.py`; full suite green at 35 tests.

Also added `scripts/watch_job.ps1` - terminal progress bar for any job id.

### Regression suite from the field

Five fresh AMI corpus meetings (13-40 min, varied scenarios) were pushed through the completed UI by the operator as acceptance testing. All processed successfully - no errors reported.

**Status after Phase 8:** build phases 4-7 complete, UI complete, speaker naming complete. Remaining roadmap: deployment packaging (Phase 8 of PROJECT_GUIDE roadmap) and the login+voiceprint future feature (Decision 11 later half).

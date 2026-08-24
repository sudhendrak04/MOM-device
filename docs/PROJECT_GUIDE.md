# MOM Project Guide

A complete explanation of what we are building, why every technology was chosen, and how the whole system works — written so that anyone (technical or non-technical) can follow along from start to finish.

---

## Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [The Journey of a Meeting](#2-the-journey-of-a-meeting)
3. [Every Decision We Took, and Why](#3-every-decision-we-took-and-why)
4. [Model Comparison and Benchmarks](#4-model-comparison-and-benchmarks)
5. [Your Hardware vs Requirements](#5-your-hardware-vs-requirements)
6. [Target Architecture](#6-target-architecture)
7. [API Surface](#7-api-surface)
8. [Security and Privacy](#8-security-and-privacy)
9. [Glossary](#9-glossary)
10. [Roadmap](#10-roadmap)

---

## 1. What We Are Building

**One sentence:** a private, self-hosted meeting assistant that listens to your meeting recording, figures out who said what, writes it all down word-for-word, and then produces clean, structured meeting minutes (MoM) automatically.

**The problem it solves.** In most organizations, someone spends 30–60 minutes after every meeting typing up minutes. Notes get lost in WhatsApp chats, action items are forgotten, and there is no reliable record of what was actually decided. Cloud tools (Otter, Fireflies) solve this but send your audio to someone else's servers — a hard no for internal, confidential discussions.

**What this project does instead:** everything runs on one PC inside your office network. Audio never leaves the machine. Any employee opens a browser page, records or uploads the meeting, and gets back a speaker-labeled transcript plus formatted minutes they can download.

**Where it came from.** This started as an edge-device experiment ("MOM-device") on a Raspberry Pi using tiny compressed models (`whisper.cpp`, `llama.cpp`) because the Pi had almost no computing power. We are now rebuilding it as a proper organizational product for normal PCs — which means we throw away the tiny-engine compromises and use full-size, high-accuracy models.

---

## 2. The Journey of a Meeting

Here is what happens to a meeting, step by step, told as a story.

### Step 0 — Capture
An employee opens `http://office-server:8000` in their browser. They either:
- click **Record** and speak during the meeting (the browser's microphone captures sound), or
- drag-and-drop an existing audio file (WAV, MP3, M4A, WEBM — anything).

The browser sends the audio to the server. The server replies instantly with a **meeting ID** — like a token number at a shop. Processing starts in the background; nobody has to watch a spinner.

### Step 1 — Decode and Clean (ffmpeg + noise handling)
The uploaded file might be any format. `ffmpeg` converts it into a single standard format: mono, 16 kHz WAV — the format every model downstream expects. If the audio is noisy, a light cleanup pass improves it before analysis.

*Why 16 kHz mono? Speech models are trained on this exact shape. Stereo music-quality files just waste memory.*

### Step 2 — Find the Speech (VAD — Voice Activity Detection)
Silence is not just useless, it is dangerous: language models tend to "hallucinate" — inventing words — when fed silence. A tiny, very fast model called **Silero VAD** scans the audio and marks where humans are actually talking vs where there is silence/background noise. Only real speech moves forward.

### Step 3 — Who Spoke When (Diarization)
Now we answer: *how many people talked, and during which time ranges?*

**Speaker diarization** produces something like:

```
00:00.0 – 00:14.2   Speaker 1
00:14.5 – 00:31.0   Speaker 2
00:31.8 – 00:45.1   Speaker 1
...
```

It does not know names — only "this voice is the same person as that earlier voice." Our engine here is **pyannote.audio** (the open-source standard). Note: it labels voices, not identities. Mapping "Speaker 1" to "Priya" can be done later by the person reviewing the minutes.

### Step 4 — What Was Said (Transcription)
**faster-whisper** (an optimized build of OpenAI's Whisper) converts speech to text. Crucially, we request **word-level timestamps**: not just "here is the sentence," but "this word was spoken between second 12.4 and 12.8."

This replaces the old approach completely. The Raspberry-Pi version launched the whisper command-line tool once *per speaker segment*, saving each result to a temp file — slow and fragile on long meetings. Now one process handles the whole meeting in memory.

### Step 5 — Match Words to People (Alignment)
Two lists now exist: *words with timestamps* and *speaker turns with timestamps*. Alignment simply overlaps them:

```
Word "budget" spoken at 15.2s  →  falls inside Speaker 2's turn  →  Speaker 2 said "budget"
```

Result: a readable, speaker-labeled transcript.

```
[00:00.0] Speaker 1: Good morning everyone, let's start with the Q3 numbers.
[00:14.5] Speaker 2: Sure. The budget variance is within two percent...
```

### Step 6 — Write the Minutes (LLM via Ollama)
A local Large Language Model (**qwen2.5:7b** served by Ollama) reads the transcript and writes structured minutes: overview, discussion points, decisions made, action items with owners, next steps.

Long meetings produce transcripts longer than what fits in the model's memory window at once, so we reuse a proven trick already in the old code: split the transcript into overlapping chunks, summarize each, then synthesize all summaries into one final document. Overlap prevents sentences from being cut in half mid-thought.

### Step 7 — Store and Serve
Everything (audio, raw transcript, labeled transcript, minutes, timings) is saved into a database and files on disk. The employee sees live progress ("Diariazing… Transcribing segment 40/120… Writing minutes…"), then downloads the transcript and minutes. The meeting history stays searchable later — restarting the server loses nothing.

---

## 3. Every Decision We Took, and Why

### Decision 1 — Drop `whisper.cpp` and `llama.cpp`

| | Old (Pi era) | New (PC era) |
|---|---|---|
| Reason they existed | Pi had ~1 GB usable RAM; needed C++ engines with 8-bit compressed models | PC has 64 GB RAM + 8 GB GPU |
| STT engine | `whisper-cli.exe` subprocess per segment | faster-whisper, one in-process call |
| Model | ggml-small-q8_0 (~14–22% WER class) | large-v3-turbo (~7.75% avg WER) |
| MoM engine | deepseek-r1:1.5b (reasoning-tiny, weak prose) | qwen2.5:7b (far stronger instruction following) |

The `.cpp` engines were brilliant engineering *for a Raspberry Pi*. On a PC they only add fragility (subprocess spawning, path hunting across six possible binary locations, temp-file juggling) with zero benefit. Full models via Python libraries are faster here *and* more accurate.

### Decision 2 — faster-whisper instead of other Whisper runtimes

Same OpenAI weights, executed by CTranslate2 (a fast inference runtime): roughly 4x the throughput of naive PyTorch Whisper at ~40% less memory, native word timestamps, built-in Silero VAD support, pure `pip install`. No compilation, no PATH hunting, Windows-friendly.

### Decision 3 — pyannote instead of Picovoice Falcon

Falcon is closed-source and quota-metered per free account key (one key was even committed into the old repo in plaintext). pyannote is MIT-licensed, self-hosted, no quotas, configurable speaker count, best-in-class open-source accuracy on meeting audio, and it runs happily on the same GPU.

**Action item carried forward:** the committed Falcon access key must be revoked at console.picovoice.ai since it lives in git history.

### Decision 4 — Custom thin alignment instead of pulling in WhisperX

WhisperX bundles transcription + alignment + diarization, which sounds attractive until you want control: custom progress reporting, custom merging rules, fewer heavy dependencies. Our alignment logic is ~50 lines of timestamp overlap math using data both engines already output. WhisperX remains documented as a fallback option if alignment quality ever disappoints.

### Decision 5 — FastAPI backend + plain web UI (not Streamlit)

Streamlit reruns the entire script on every widget interaction, keeps state only in browser sessions, and cannot expose REST APIs properly. An organization needs: multiple users, background jobs that survive page refreshes, persistent history, and integrations via API. FastAPI gives us all four; the UI becomes a simple static page the backend serves — no Node.js toolchain required.

### Decision 6 — SQLite for storage

Zero-administration, file-based, perfect for a single-server office deployment. If usage ever outgrows it, the data layer is isolated enough to swap PostgreSQL in without touching business logic.

### Decision 7 — Background job worker with staged progress

A one-hour meeting takes minutes to process. HTTP requests must return instantly, so processing happens in a background thread pool. Each stage reports progress (`uploaded → decoding → diarizing → transcribing → aligning → summarizing → done`), which the UI polls. Server restarts don't lose completed work because state lives in SQLite.

### Decision 8 — Simple API-key authentication

One admin-configured key (`X-API-Key` header) protects everything. Right-sized for a trusted office LAN; upgradeable to real user accounts later without rearchitecting.

### Decision 9 — Ollama stays as the LLM server

Already proven in this project, dead-simple model management, OpenAI-compatible endpoint. The default model upgrades from deepseek-r1:1.5b to qwen2.5:7b; it stays switchable via config/UI dropdown.

### Decision 10 — Cap meetings at 6–7 active speakers (current stage)

pyannote's speaker-attribution accuracy stays near its peak through roughly 7 voices and only degrades meaningfully beyond ~15. Capping scope keeps diarization quality high, keeps GPU time predictable, and lets the speaker-renaming UI stay a simple per-speaker list instead of complex clustering review. Larger meetings become a later-phase concern, not a current one.

### Decision 11 — Speaker naming: manual rename now, login-voiceprints later

**Now (UI phase):** every completed meeting shows one rename field per detected speaker; names persist for that meeting and re-render into transcript and minutes.

**Later (top-priority future feature):** when the tool grows real login-based auth, each employee enrolls at first login by speaking a **5–10 second self-introduction**. That recording becomes a persistent voiceprint tied to their account+name; every meeting they attend thereafter is auto-named via embedding matching (pyannote already emits `speaker_embeddings` at no extra cost). Manual rename remains as the fallback for un-enrolled guests.

This ordering keeps the current build simple while reserving the highest-value automation for the auth milestone, where enrollment gets a natural home.

---

## 4. Model Comparison and Benchmarks

Numbers collected August 2026 from the Hugging Face Open ASR Leaderboard, published model cards, and independent benchmark papers (SDBench / Interspeech 2025; arXiv:2509.26177).

### 4.1 Speech-to-Text candidates (WER = % of words gotten wrong; lower is better)

| Model | Params | Languages | License | Avg WER | Speed (GPU) | VRAM | Verdict |
|---|---|---|---|---|---|---|---|
| **Whisper large-v3-turbo** | 809M | 99 (incl. Hindi) | MIT | ~7.75% | ~16–80x realtime | ~2–3 GB int8 | **CHOSEN** |
| Whisper large-v3 | 1.55B | 99 | MIT | ~7.4% | ~8–12x | ~3–6 GB | Config option (max accuracy) |
| Distil-Whisper large-v3 | 756M | English only | MIT | ~8.2% | ~25x | ~4 GB | Rejected: English-only |
| Parakeet TDT 0.6B v3 | 0.6B | 25 European langs, no Hindi | CC-BY-4.0 | ~6.3% | up to ~3300x | ~2–4 GB | Best raw speed+accuracy, rejected: no Hindi, NeMo dependency |
| Canary-Qwen 2.5B | 2.5B | English only | CC-BY-4.0 | ~5.63% (leaderboard #1) | slower | ~6 GB+ | Top accuracy, rejected: English-only |
| IBM Granite Speech 4.1 2B | 2B | EN + translate | Apache 2.0 | ~5.3–5.9% | moderate | ~5 GB+ | Strong newcomer, watchlist |
| Old: whisper.cpp small-q8_0 | 244M | 99 | MIT | ~14–22% | slow (CPU) | RAM-based | Replaced |

**Why turbo wins for us:** Indian office meetings mean Indian-accent English and frequent Hinglish — multilingual coverage is non-negotiable, ruling out every NVIDIA/Meta English-specialist despite their leaderboard crowns. Turbo delivers near-large-v3 accuracy at a fraction of the compute, leaving GPU room for diarization and the LLM. It also provides the word timestamps Step 5 depends on.

### 4.2 Speaker Diarization candidates (DER = % of speaking time attributed wrongly; lower is better)

Benchmarks: AMI = multi-party meeting corpus (our exact scenario); VoxConverse = wild YouTube audio; DIHARD III = adversarial stress test.

| Model | AMI | VoxConverse | DIHARD III | License | Verdict |
|---|---|---|---|---|---|
| **pyannote community-1 (v4)** | **17.0%** | **8.5%** | 20.2% | MIT | **CHOSEN** (2026 SOTA open) |
| pyannote 3.1 | 18.8% | 11.2% | 21.7% | MIT | Kept as config fallback |
| NVIDIA Sortformer v2 | good at ≤4 spk | degrades >4 spk | — | Apache 2.0 | Rejected: hard 4-speaker cap, NeMo dep |
| DiariZen | 13.3% multi-corpus avg | 5.2% | — | research | Great numbers, not turnkey on Windows |
| Picovoice Falcon (old) | closed | closed | closed | free-tier quota | Removed |
| pyannoteAI precision-2 | 12.9% | 7.4% | 14.7% | Commercial | Future paid upgrade path |

**Practical accuracy expectation:** clean 2–4 speaker recordings ≈ 90–95% correct attribution; larger/noisier rooms degrade gracefully. Speaker-count accuracy: ~93–97% with 2 speakers, dropping toward 70–85% beyond ~15.

### 4.3 Supporting cast

| Component | Choice | Role |
|---|---|---|
| VAD | Silero VAD (MIT) | Skip silence, prevent hallucinations |
| Minutes LLM | Ollama + qwen2.5:7b (Q4) default; llama3.1:8b / mistral selectable | Structured minutes generation |
| Audio decode | ffmpeg | Accept any upload format |
| Web framework | FastAPI + Uvicorn | REST APIs + serves UI |
| Database | SQLite via SQLAlchemy | Meetings, jobs, results |
| Frontend | Single-page HTML/JS served by backend | Record/upload, progress, download |

### 4.4 One-time setup requirement

pyannote models are gated on Hugging Face (free): create an account, click "agree" on both `speaker-diarization-community-1` and its `segmentation` model pages, paste your token into `.env` as `HF_TOKEN`. Every other component downloads/configures itself.

---

## 5. Your Hardware vs Requirements

Development machine measured August 2026:

| Component | Available | Needed by new stack | Fit |
|---|---|---|---|
| CPU | Intel Core Ultra 9 185H — 16 cores / 22 threads | 4+ cores for ffmpeg/API/threadpool | Huge headroom |
| RAM | 63.5 GB | ~16 GB comfortable peak | Huge headroom |
| GPU | NVIDIA RTX 2000 Ada Laptop — 8 GB VRAM | see budget below | Fits by design |
| Disk | SSD recommended | ~12 GB models + data | Fine |

**GPU memory budget (the tight resource), worst case simultaneous:**

| Resident model | Approximate VRAM |
|---|---|
| faster-whisper large-v3-turbo (int8) | ~2.5 GB |
| pyannote diarization | ~2.0 GB |
| qwen2.5:7b Q4 (partial offload to CPU is fine with 64 GB RAM) | ~2–5 GB |
| **Total** | **~7–9.5 GB → stages run sequentially, so real peak stays under 8 GB** |

The pipeline naturally staggers GPU load: diarization finishes before minutes-generation begins. Even if all three loaded at once, Ollama spills layers to system RAM transparently — slower, but never a crash.

**Expected performance on this machine** (vs the old README's "2–5 minutes per minute of audio"): a 60-minute meeting should complete end-to-end in roughly 2–4 minutes total. Exact figures get recorded during Phase 7 verification and tuned from there.

---

## 6. Target Architecture

```
                       Office LAN
                           │
        ┌──────────────────┼───────────────────┐
        │                  │                   │
   Employee PC 1      Employee PC 2       Integrations
   (browser)          (browser)           (scripts/tools)
        │                  │                   │
        └───────── X-API-Key ───────────────────┘
                           │
              ┌────────────▼────────────┐
              │      FastAPI server     │
              │  main.py (uvicorn)      │
              │                         │
              │  api/meetings.py  ──────┼── upload / list / fetch results
              │  api/jobs.py      ──────┼── progress polling
              │  api/system.py    ──────┼── health + engine status
              │  security.py      ──────┼── checks X-API-Key
              │                         │
              │  worker.py ─────────────┼── background thread pool
              │      │                  │
              │  services/              │
              │   ├ transcription.py ───┼── faster-whisper (GPU)
              │   ├ diarization.py ─────┼── pyannote (GPU)
              │   ├ alignment.py        │   words x speakers overlap
              │   └ minutes.py ─────────┼── Ollama batch+synthesize
              │                         │
              │  db.py / models.py ─────┼── SQLite: meetings, jobs
              └───────┬─────────────────┘
                      │ serves
              ┌───────▼────────┐      ┌─────────────┐
              │ frontend/      │      │ data/       │
              │ index.html     │      │ uploads/    │
              │ app.js         │      │ outputs/    │
              └────────────────┘      │ mom.db      │
                                      └─────────────┘
```

Reading the diagram as a story: browsers talk only to FastAPI. FastAPI validates the API key, stores the upload, queues a job, and returns immediately. The worker picks jobs up, drives the five-stage pipeline through the service modules, writes results to SQLite + disk, and updates job status as it goes. Browsers poll the jobs endpoint to paint progress bars, then fetch finished artifacts. Nothing blocks anything; one busy meeting never prevents another upload.

**Repository layout after rebuild:**

```
MOM-device/
├── backend/
│   ├── app/                    # everything on the server side
│   └── tests/                  # pytest suites
├── frontend/                   # static HTML/JS/CSS
├── docs/                       # this guide + future docs
├── legacy/                     # original Pi-era scripts (reference)
├── scripts/setup.ps1           # one-command environment bootstrap
├── data/                       # runtime storage (gitignored)
├── graphify-out/               # knowledge graph (committed)
├── .env.example                # template for secrets/config
└── requirements.txt            # pinned dependencies
```

---

## 7. API Surface

All endpoints require header `X-API-Key: <key>` except health.

| Method & Path | Purpose | Returns |
|---|---|---|
| `GET /api/health` | Liveness (no auth) | `{"status": "ok"}` |
| `GET /api/system/status` | Engine readiness: whisper/pyannote/Ollama/models | statuses + versions |
| `POST /api/meetings/upload` | Upload audio (multipart) or start browser-recording save | `{meeting_id, job_id}` |
| `GET /api/jobs/{job_id}` | Progress polling | `{stage, percent, message}` |
| `GET /api/meetings` | List history | array of summaries |
| `GET /api/meetings/{id}` | Full record | metadata + links |
| `GET /api/meetings/{id}/transcript` | Speaker-labeled transcript | text/plain |
| `GET /api/meetings/{id}/minutes` | Generated minutes | text/markdown |
| `DELETE /api/meetings/{id}` | Remove meeting + files | `204` |

Example — upload and poll:

```bash
curl -H "X-API-Key: $KEY" -F "file=@meeting.wav" http://localhost:8000/api/meetings/upload
# → {"meeting_id": "a1b2c3", "job_id": "j9k8l7"}

curl -H "X-API-Key: $KEY" http://localhost:8000/api/jobs/j9k8l7
# → {"stage": "transcribing", "percent": 62, "message": "segment 41/120"}
```

---

## 8. Security and Privacy

1. **Audio never leaves the machine.** STT, diarization, and LLM inference are 100% local processes. No cloud calls exist anywhere in the pipeline.
2. **API-key gate.** Every endpoint (except liveness) requires the shared key. Keys come from `.env`, never from source code.
3. **Known incident to remediate:** the old repo contains a live Picovoice access key hardcoded in `src/final.py` and `src/app.py`, permanently in git history. It must be revoked at console.picovoice.ai. The new code contains zero hardcoded secrets.
4. **Uploads validated** (size caps, extension allowlist, ffmpeg probe) before touching the pipeline.
5. **Deletion is real:** deleting a meeting removes DB rows *and* media/transcript/minutes files.
6. **Data residency ready for compliance:** since nothing egresses, this architecture passes typical internal-data policies that block Otter/Fireflies-class tools outright.

---

## 9. Glossary

| Term | Plain meaning |
|---|---|
| **ASR / STT** | Automatic Speech Recognition / Speech-to-Text: audio → words |
| **WER** | Word Error Rate — % of words transcribed wrongly. Lower = better |
| **Diarization** | Figuring out who spoke when; labels voices as Speaker 1, 2, ... |
| **DER** | Diarization Error Rate — % of speaking time attributed to the wrong person/silence. Lower = better |
| **VAD** | Voice Activity Detection — separates human speech from silence/noise |
| **Alignment** | Matching each transcribed word to its exact moment in audio |
| **RTF / RTFx** | Real-Time Factor — processing speed relative to audio length. 60x means 1 hour of audio processed in 1 minute |
| **VRAM** | Memory on the graphics card; the scarcest resource for AI models |
| **Quantization (int8/Q4)** | Compressing model numbers to smaller ones: less memory/faster, tiny accuracy cost |
| **LLM** | Large Language Model — writes the minutes from the transcript |
| **Ollama** | A local server app that runs LLMs on your own machine |
| **MoM** | Minutes of Meeting — the structured summary document |
| **FastAPI** | Python framework for building REST APIs quickly |
| **REST API** | How programs talk to the server over HTTP (upload, poll, download) |
| **Job queue / worker** | Pattern where slow work happens in the background with progress tracking |
| **SQLite** | Self-contained file database; no server software needed |
| **AST** | Abstract Syntax Tree — code parsed structurally (how graphify maps codebases free of charge) |

---

## 10. Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1. Graphify knowledge graph** | CLI installed, skill registered, `graphify-out/` built & queryable | **Done** |
| **2. PROJECT_GUIDE.md** | This document | **Done** |
| **3. Repo hygiene** | Purged hardcoded keys, legacy moved, `.gitignore` fixed, Picovoice key revocation documented | **Done** |
| **4. Backend build** | FastAPI + config + auth + SQLite + worker + services; verified through public APIs batch-by-batch | **Done** |
| **5. Web UI** | Record/upload, live progress, minutes/transcript/speakers tabs, speaker renaming (Decision 11), history | **Done** |
| **6. Tests & setup script** | 35 pytest suites + `scripts/setup.ps1` + rewritten README + helper scripts | **Done** |
| **7. End-to-end verification** | Real AMI meetings incl. diarization verified; 5.44x realtime benchmark recorded | **Done** |
| **8. Deployment (separate phase)** | Docker Compose / Windows service, LAN rollout, optional HTTPS | Deferred by decision |
| **9. Future options** | Login accounts + **voiceprint enrollment at first login** (5–10 s spoken intro -> permanent auto-naming, Decision 11), roles, cloud LLM upgrade path (pyannoteAI precision-2 class), multi-room GPU server | Ideas |

---

*Maintained alongside the codebase. When a decision changes, update section 3 — the graph in `graphify-out/` tracks the code automatically.*

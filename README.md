# MOM-device - Self-Hosted Meeting Minutes Server

A private, self-hosted meeting assistant that takes any meeting recording, figures out **who said what**, writes a **speaker-labeled transcript**, and generates **structured meeting minutes** - entirely on your own machine. No cloud, no bots joining your calls, no per-seat fees.

Built as the rebuild of an edge-device prototype into an organization-ready tool: FastAPI backend + vanilla web UI + a fully local AI pipeline.

---

## How it works

```
Upload/Record -> ffmpeg decode -> Speaker Diarization -> Transcription -> Alignment -> Meeting Minutes
  (browser)      (16kHz mono)     (pyannote community-1)   (faster-whisper)  (overlap math)   (Ollama LLM)
```

- **Speaker diarization** - pyannote community-1 (2026 open-source SOTA) labels voices as Speaker 1, 2, ...
- **Transcription** - Whisper large-v3-turbo via faster-whisper on CUDA with word-level timestamps and Silero VAD
- **Minutes** - local LLM (qwen2.5 via Ollama) with map-reduce batching for hour-long meetings
- **Measured speed** - 21-minute real meeting processed end-to-end in under 4 minutes (5.4x realtime)

Full technical walkthrough: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Features

- Drag-and-drop upload (wav/mp3/m4a/webm/ogg/flac/aac, up to 500 MB) or record directly in the browser
- Live progress: stage chips + percentage bar while processing runs in the background
- Speaker-labeled transcript with timestamps; rename speakers to real names in one click (transcript, minutes *and* downloads update instantly - no reprocessing)
- Structured markdown minutes: overview, discussion points, decisions, action items
- Meeting history that survives server restarts (SQLite)
- REST API for everything the UI does - integrate scripts or internal tools
- Graceful degradation: without a HuggingFace token you still get transcript + minutes (just without speaker labels)
- 100% local inference; audio never leaves the machine; works air-gapped

---

## Prerequisites

Install every item in this section **before** running `setup.ps1`. Nothing here is optional — skipping any step will cause the pipeline to fail.

---

### 1. Python 3.10 or newer

Download from **https://www.python.org/downloads/**.

> [!IMPORTANT]
> On the installer's first screen, tick **"Add Python to PATH"** before clicking Install Now. Without this, every `python` command will fail.

Verify after installing:

```powershell
python --version   # should print Python 3.10.x or higher
```

---

### 2. Git + Git LFS

Download Git from **https://git-scm.com/download/win** and install with default settings.

The test audio files in `data/test_audio/` are stored with **Git LFS** (Large File Storage). Without it you get pointer files instead of real audio. Install LFS after Git:

```powershell
# Download the installer from https://git-lfs.com and run it, OR:
winget install GitHub.GitLFS

# Then initialise LFS once for your user account:
git lfs install
```

Verify both:

```powershell
git --version
git lfs version   # should print git-lfs/3.x.x
```

> [!NOTE]
> If you already cloned the repo before installing LFS, run `git lfs pull` inside the repo folder to fetch the audio files.

---

### 3. NVIDIA GPU drivers + CUDA (GPU users only — skip for CPU-only)

The setup script automatically installs the **CUDA 12.1** PyTorch build when it detects `nvidia-smi`. For that to work your display driver must already support CUDA 12.1+.

- Download the latest Game Ready or Studio driver from **https://www.nvidia.com/drivers**
- Minimum driver version: **525.60** (released with CUDA 12.0)

Verify:

```powershell
nvidia-smi   # should print your GPU name and driver version
```

> [!NOTE]
> CPU-only mode still works without any GPU. Processing will be slower (~1x realtime vs ~5x on GPU) but everything runs.

---

### 4. ffmpeg

ffmpeg is required to decode every audio/video format the server accepts (wav, mp3, m4a, webm, ogg, flac, aac). Without it the server cannot process any uploaded file.

**Option A — winget (recommended, one command):**

```powershell
winget install Gyan.FFmpeg
```

After the install finishes, **close and reopen your terminal** so the new `PATH` entry takes effect.

**Option B — manual:**

1. Download the latest "essentials" build from **https://www.gyan.dev/ffmpeg/builds/** (file named `ffmpeg-release-essentials.zip`).
2. Extract the zip and copy the three files inside `bin\` (`ffmpeg.exe`, `ffprobe.exe`, `ffplay.exe`) to a permanent folder, e.g. `C:\ffmpeg\bin\`.
3. Add that folder to your system `PATH`:
   - Open **Start → "Edit the system environment variables" → Environment Variables**
   - Under *System variables*, select **Path → Edit → New**
   - Paste `C:\ffmpeg\bin` and click OK on every dialog

**Option C — Chocolatey:**

```powershell
choco install ffmpeg
```

Verify (in a fresh terminal):

```powershell
ffmpeg -version    # should print ffmpeg version 6.x or 7.x
ffprobe -version
```

---

### 5. Ollama + LLM model

Ollama serves the local LLM that writes the meeting minutes.

1. **Download and install Ollama** from **https://ollama.com/download** (Windows installer, ~100 MB).
2. After the installer finishes, Ollama runs as a background service automatically.
3. **Pull the minutes model** — this downloads ~4.4 GB the first time:

```powershell
ollama pull qwen2.5:7b
```

Verify:

```powershell
ollama list   # should show qwen2.5:7b in the table
```

> [!NOTE]
> You can substitute a different model later by changing `OLLAMA_MODEL` in `.env`. Any model listed on https://ollama.com/library works. `qwen2.5:7b` is the tested default.

---

### 6. HuggingFace account + token (for speaker diarization)

Speaker diarization (knowing *who* said *what*) uses pyannote models that are gated behind a HuggingFace license agreement. This step is **optional** — without it you get a transcript and minutes but no speaker labels.

1. Create a free account at **https://huggingface.co/join**
2. Accept the license for **pyannote/speaker-diarization-community-1**:
   `https://huggingface.co/pyannote/speaker-diarization-community-1`
   (click the "Agree and access repository" button)
3. Also accept the licence for the segmentation model linked on that page (usually `pyannote/segmentation-3.0`)
4. Create a **Read** access token at **https://huggingface.co/settings/tokens**
5. Copy the token — you will paste it as `HF_TOKEN` in `.env` in the next section

> [!TIP]
> The server logs `Diarization skipped` and continues normally when `HF_TOKEN` is empty or the models are unreachable. You can add the token later without re-running setup.

---

## Installation

### Step 1 — Clone the repository

```powershell
git clone https://github.com/<your-username>/MOM-device.git
cd MOM-device
```

### Step 2 — Run the setup script

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

The script does the following automatically:

| What | Detail |
|---|---|
| Creates `.venv` | Isolated Python virtual environment inside the project folder |
| Installs Python dependencies | `requirements.txt` + `requirements-dev.txt` via pip |
| Installs PyTorch **last** | CPU wheel first, then replaced with the CUDA 12.1 wheel if `nvidia-smi` is found — order is critical or the CPU wheel wins |
| Checks ffmpeg | Warns if not on PATH (see Prerequisites § 4 above) |
| Seeds `.env` | Copies `.env.example` → `.env` on first run |

Expected final output:

```
Setup complete. Next steps:
  1. edit .env (API_KEY, optionally HF_TOKEN)
  2. start ollama and pull a model:      ollama pull qwen2.5:7b
  3. run the server:                     .venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
  4. open http://localhost:8000 in a browser
```

### Step 3 — Configure `.env`

Open the newly created `.env` file in any text editor and fill in at minimum:

```ini
# REQUIRED — any long random string; sent by the browser as X-API-Key
API_KEY=change-me-to-a-long-random-string

# OPTIONAL — paste your HuggingFace Read token for speaker diarization
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# LLM model name (must match what you pulled with `ollama pull`)
OLLAMA_MODEL=qwen2.5:7b
```

All other values have working defaults (see Configuration reference below).

### Step 4 — Pre-download AI models (recommended)

The first time the server processes a meeting it downloads the Whisper transcription weights (~1.5 GB). Run this once to cache them ahead of time:

```powershell
.venv\Scripts\python scripts\prewarm_models.py
```

You will see a progress bar. It takes 2–5 minutes on a typical broadband connection and only runs once — subsequent starts are instant.

### Step 5 — Start Ollama and the MOM server

```powershell
# Terminal 1 — Ollama LLM server (skip if the Ollama tray icon is already running)
ollama serve

# Terminal 2 — MOM Server
.venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### Step 6 — Open the UI

Open **http://localhost:8000** in your browser. Enter the `API_KEY` you set in `.env` when prompted (stored in the browser only, never sent to any server). Then record directly or drop an audio/video file.

---

### Share with your team (LAN)

Because the server binds `0.0.0.0`, anyone on the same network can open `http://<your-pc-ip>:8000`, enter the same API key, and use the tool from their own browser — no installation on their side. Find your IP with `ipconfig` (allow port 8000 through Windows Firewall if prompted). Audio is processed only by your machine.

---

## Try it with sample audio

The repository ships six real meeting recordings from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/) (permissive research license) so you can see the full pipeline in action without recording anything yourself. They are stored under **`data/test_audio/`** and tracked with Git LFS — they download automatically when you clone.

| File | Duration | Speakers | Notes |
|---|---|---|---|
| `ES2004a.wav` | ~21 min | 4 | Scenario meeting, product design discussion |
| `IS1009a.wav` | ~18 min | 4 | Scenario meeting, project planning |
| `TS3008a.wav` | ~28 min | 4 | Scenario meeting, remote participants |
| `TS3010a.wav` | ~21 min | 4 | Scenario meeting, whiteboard session |
| `IB4011.wav` | ~49 min | 4 | Longer scenario meeting |
| `long_test.wav` | ~26 min | 4 | Internal stress-test recording |

> [!NOTE]
> Git LFS is required to get the actual audio files. If you cloned without LFS installed, run `git lfs pull` inside the repo after installing LFS and they will download.

### Upload via the UI

With the server running, open **http://localhost:8000**, drag any file from `data/test_audio/` onto the upload zone, and watch the pipeline stages tick by in real time.

### Upload via the API (curl)

```bash
# PowerShell / bash — replace with your key and chosen file
curl -H "X-API-Key: your-api-key" \
     -F "file=@data/test_audio/ES2004a.wav" \
     http://localhost:8000/api/meetings/upload
# {"meeting_id":"...","job_id":"...","status":"queued"}

# Poll until stage == "completed"
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/jobs/<job_id>
```

Expected end-to-end time on an RTX-class GPU: **under 5 minutes** for a 21-minute file.

---


## Configuration reference

Every knob lives in `.env` (see [.env.example](.env.example)):

| Key | Default | Purpose |
|---|---|---|
| `API_KEY` | - | shared secret for X-API-Key / Bearer auth |
| `HF_TOKEN` | empty | enables gated pyannote models |
| `DATA_DIR` | `data` | storage root (uploads/outputs/db/models) |
| `WHISPER_MODEL` | `large-v3-turbo` | STT model |
| `DIARIZATION_MODEL` | `community-1` | falls back to `3.1` automatically |
| `ALLOW_NO_DIARIZATION` | `true` | continue without speaker labels on failure |
| `DEVICE` / `COMPUTE_TYPE` | `auto` | force cpu/cuda or quantization if needed |
| `MAX_UPLOAD_MB` | `500` | hard upload cap |
| `WORKER_CONCURRENCY` | `1` | GPU serialization; raise at your own VRAM risk |
| `BATCH_SIZE_CHARS` / `BATCH_OVERLAP_CHARS` | `3000` / `200` | long-transcript map-reduce tuning |

---

## API

Everything the UI does is available over REST (header `X-API-Key: <key>`):

```bash
curl -H "X-API-Key: $KEY" -F "file=@meeting.wav" http://localhost:8000/api/meetings/upload
# {"meeting_id":"...","job_id":"...","status":"queued"}

curl -H "X-API-Key: $KEY" http://localhost:8000/api/jobs/<job_id>
# {"stage":"transcribing","progress":55,"message":"Transcribing (62%)"}
```

Endpoints: upload, job polling, history, meeting detail, speaker list/rename (`GET|PATCH /api/meetings/{id}/speakers`), transcript, minutes, delete. Full table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Testing

```powershell
.venv\Scripts\python -m pytest backend\tests -q
```

35 tests covering auth, upload validation, alignment math, minutes batching, and the speaker-naming flow - all CPU-only, no models needed.

---

## Benchmarks

Measured on the dev machine (RTX 2000 Ada 8 GB): real AMI corpus meetings, four speakers, correct attribution, 5.44x realtime end-to-end including LLM minutes. Comparison against Fireflies/Otter/MeetGeek/tl;dv with sources: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | beginner-friendly master guide: every technology decision with benchmark tables, hardware fit, security posture, glossary |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | system diagram, journey of a meeting mapped to code files, data model, concurrency model |
| [docs/REBUILD_LOG.md](docs/REBUILD_LOG.md) | phase-by-phase rebuild story with measured numbers |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | head-to-head vs commercial tools |
| [docs/PROBLEMS.md](docs/PROBLEMS.md) | every real problem hit during the build + solutions (12 entries) |

---

## Security notes

- All secrets live in `.env` (gitignored); nothing hardcoded
- Every endpoint except `/api/health` requires authentication
- Uploads validated by extension allowlist + streaming byte cap
- Deleting a meeting removes database rows *and* media/transcript/minutes files

---

## Credits

Built on the shoulders of:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) - CTranslate2 Whisper runtime
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT) - speaker diarization
- [Silero VAD](https://github.com/snakers4/silero-vad) (MIT) - voice activity detection
- [Ollama](https://ollama.com) (MIT) - local LLM serving
- [FastAPI](https://fastapi.tiangolo.com) (MIT), [SQLAlchemy](https://www.sqlalchemy.org) (MIT)
- OpenAI Whisper models (Apache-2.0 weights) and the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/) for test data

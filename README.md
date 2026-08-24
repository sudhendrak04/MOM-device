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

## Requirements

| Component | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11 | Linux should work; setup script is PowerShell |
| Python | 3.10+ | [python.org/downloads](https://www.python.org/downloads/) - tick "Add to PATH" |
| Git | any | [git-scm.com](https://git-scm.com/download/win) |
| NVIDIA GPU (optional but recommended) | ~8 GB VRAM | RTX 2000 Ada class verified; CPU-only mode works too |
| RAM | 16 GB+ | |
| ffmpeg / ffprobe | any recent | `winget install Gyan.FFmpeg` (or let setup.ps1 remind you) |
| Ollama | current | [ollama.com/download](https://ollama.com/download), then `ollama pull qwen2.5:7b` |
| HuggingFace account | free | only for gated pyannote diarization models |

---

## Quick start

### 1. Get the code

```powershell
git clone https://github.com/<your-username>/MOM-device.git
cd MOM-device
```

### 2. One-command setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Creates `.venv`, installs pinned dependencies, installs the CUDA PyTorch build **last** so the CPU wheel cannot override it (order matters), checks ffmpeg, and seeds `.env` from the template.

### 3. Configure `.env`

Open `.env` (created by setup) and fill in:

```ini
API_KEY=choose-a-long-random-string     # required - sent as X-API-Key
HF_TOKEN=hf_...                         # optional - enables speaker diarization
OLLAMA_MODEL=qwen2.5:7b                 # minutes LLM
```

For diarization: create a free HuggingFace account, accept the license at `https://huggingface.co/pyannote/speaker-diarization-community-1` (and its segmentation model page linked there), create a Read token at `https://huggingface.co/settings/tokens`, and paste it as `HF_TOKEN`. Skip it and the pipeline still works - just without speaker names.

### 4. Start the LLM server + MOM server

```powershell
ollama serve                                            # if not already running
.venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

First run downloads the Whisper weights (~1.5 GB, one time) on the first meeting - or run `.venv\Scripts\python scripts\prewarm_models.py` ahead of time to avoid the wait.

### 5. Use it

Open **http://localhost:8000**, enter your API key once (stored in the browser only), then record or drop a file. Watch the progress bar; download transcript and minutes when done.

Optional helpers:

```powershell
.venv\Scripts\python scripts\prewarm_models.py          # cache whisper weights before first meeting
powershell -File scripts\watch_job.ps1 <job_id>         # terminal progress bar for any job
```

### Share with your team (LAN)

Because the server binds `0.0.0.0`, anyone on the same network can open `http://<your-pc-ip>:8000`, enter the same API key, and use the tool from their own browser - no installation on their side. Find your IP with `ipconfig` (allow port 8000 through Windows Firewall if prompted). Audio is processed only by your machine either way.

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

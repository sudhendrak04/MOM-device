# MOM-device Architecture

The complete technical story of how one meeting recording becomes speaker-labeled transcript plus structured minutes - every component, every data transformation, and why each piece sits where it sits.

---

## 1. System at a glance

```
                       Office LAN
                           │
        ┌──────────────────┼───────────────────┐
        │                  │                   │
   Employee PC 1      Employee PC 2      Integrations
   (browser)          (browser)          (scripts/API clients)
        │                  │                   │
        └──────── X-API-Key ───────────────────┘
                           │
             ┌─────────────▼──────────────┐
             │       FastAPI server       │
             │   backend/app/main.py      │
             │                            │
             │  api/meetings.py  ─────────┼── upload / history / results
             │  api/jobs.py      ─────────┼── progress polling
             │  api/system.py    ─────────┼── health + engine status
             │  security.py      ─────────┼── constant-time key check
             │                            │
             │  worker.py ────────────────┼── queue.Queue + thread pool
             │      │                     │   (GPU-serialized, concurrency=1)
             │  services/                 │
             │   ├ audio.py        ffmpeg decode -> 16 kHz mono WAV
             │   ├ diarization.py  pyannote community-1 -> speaker turns
             │   ├ transcription.py faster-whisper turbo -> words+timestamps
             │   ├ alignment.py    words x turns -> labeled transcript
             │   └ minutes.py      Ollama batch -> summarize -> synthesize
             │                            │
             │  db.py / models.py ────────┼── SQLite (data/mom.db)
             └─────────┬──────────────────┘
                       │ serves static files
             ┌─────────▼─────────┐     ┌──────────────────────┐
             │ frontend/         │     │ data/                │
             │ index.html        │     │ uploads/  originals  │
             │ app.js            │     │ outputs/  wav/txt/md │
             │ styles.css        │     │ mom.db   state       │
             └───────────────────┘     │ models/   weights    │
                                       └──────────────────────┘
```

One process does everything: serves the REST API, serves the web UI, and runs the pipeline workers. Browsers never talk to the ML stack directly - only to FastAPI.

---

## 2. Journey of a meeting, mapped to code

### Step 0 - Upload (`api/meetings.py` → `upload_meeting`)
The browser POSTs multipart audio (drag-drop or an in-browser MediaRecorder blob). The endpoint validates the extension against an allowlist, streams the body to disk under `data/uploads/` with a size cap, inserts a `Meeting` row (`status="queued"`), hands the id to the worker queue, and replies instantly with `{meeting_id, job_id}`. HTTP stays fast no matter how big the meeting is.

### Step 1 - Decode (`services/audio.py` → `decode_to_wav`)
ffmpeg converts anything (wav/mp3/m4a/webm/ogg/flac/aac) into mono 16 kHz PCM WAV - the exact shape every downstream model expects - written to `data/outputs/{id}.wav`. ffprobe reports the duration used later for speedup math.

*Why 16 kHz mono? Speech models are trained on this shape; stereo music-quality audio wastes memory.*

### Step 2 - Who spoke when (`services/diarization.py` → `diarize`)
pyannote's `speaker-diarization-community-1` pipeline produces raw speaker turns `(start, end, tag)`. Two implementation notes that matter:

- **Waveform injection:** pyannote 4 normally decodes files via torchcodec, which is unusable on Windows + torch 2.5.1+cu121 (PROBLEMS P11). We decode our normalized WAV ourselves with scipy and hand the pipeline a `{"waveform": ..., "sample_rate": ...}` dict.
- **Turn hygiene:** sub-0.6 s fragments are dropped, then contiguous same-speaker turns within 0.3 s are merged (logic ported from the legacy Pi code).

If HF_TOKEN is missing or gated access fails, the whole stage degrades gracefully (see section 6).

### Step 3 - What was said (`services/transcription.py`)
faster-whisper (CTranslate2 runtime) runs Whisper large-v3-turbo on CUDA with int8_float16 quantization. Silero VAD skips silence before the model sees it (prevents hallucinations); word-level timestamps come back per segment. A lazy singleton keeps the model resident across meetings.

### Step 4 - Match words to people (`services/alignment.py`)
Pure functions overlap the two timelines: each transcribed segment lands in the speaker turn with maximum time overlap; zero-overlap segments fall back to the temporally nearest turn within a 1 s tolerance, otherwise they drop (silence artifacts). Output: `[MM:SS - MM:SS] Speaker N: text`.

### Step 5 - Write the minutes (`services/minutes.py`)
Long transcripts exceed any LLM context window, so a map-reduce runs: split into ~3000-char batches at paragraph/sentence boundaries with overlap, qwen2.5 (via local Ollama) summarizes each batch, then synthesizes all partials into final markdown minutes with fixed sections (overview, discussion, decisions, action items, next steps). Forward progress is guaranteed even when a batch has no clean break.

### Step 6 - Persist + serve (`worker.py`, `db.py`, `models.py`)
Transcript and minutes land both in SQLite and as plain files in `data/outputs/`. The job row carries stage/progress/message for live polling; completed meetings survive restarts because all state is on disk.

---

## 3. Data model

One table doubles as meeting record and job record (`models.py`):

| Column group | Fields | Purpose |
|---|---|---|
| identity | `id`, `original_filename`, `uploaded_path`, `wav_path` | what came in and where it lives |
| job state | `status`, `stage`, `stage_message`, `progress`, `error` | live polling payload |
| media facts | `duration_sec`, `num_speakers`, `language` | metadata |
| results | `transcript`, `minutes_md` | served text (raw, generic labels) |
| pipeline meta | `diarization_used`, `processing_ms` | honesty + speedup math |
| naming | `speaker_names_json` | `{ "1": "Laura", ... }` applied at serve time |
| timestamps | `created_at`, `completed_at` | history ordering |

**Speaker names are stored separately and substituted only when serving** (`api/meetings.py::_with_speaker_names`). Renaming is instant, reversible, never re-runs the GPU, and downloads get named output automatically.

---

## 4. Concurrency model

- `worker.py` runs a `queue.Queue` drained by N threads (default `WORKER_CONCURRENCY=1`).
- Concurrency 1 is deliberate: the GPU is the bottleneck resource; serializing jobs keeps latency predictable and VRAM inside budget (~2.5 GB whisper + ~2 GB pyannote peak).
- API threads stay free regardless - uploads/polls never wait behind a running meeting.
- Heavy libraries import lazily *inside* service functions, so the server boots even without torch and unit tests run without models.

---

## 5. API surface

All endpoints require `X-API-Key` except `/api/health`.

| Method & Path | Purpose |
|---|---|
| `GET /api/health` | liveness (public) |
| `GET /api/system/status` | engine readiness JSON |
| `POST /api/meetings/upload` | upload audio -> `{meeting_id, job_id}` |
| `GET /api/jobs/{job_id}` | `{stage, percent, message}` polling |
| `GET /api/meetings` | history summaries |
| `GET /api/meetings/{id}` | full record incl. saved speaker names |
| `GET /api/meetings/{id}/speakers` | detected speakers + sample quotes |
| `PATCH /api/meetings/{id}/speakers` | save display names |
| `GET /api/meetings/{id}/transcript` | text/plain (names applied) |
| `GET /api/meetings/{id}/minutes` | text/markdown (names applied) |
| `DELETE /api/meetings/{id}` | remove rows AND files |

---

## 6. Failure philosophy

Diarization is treated as an optional enhancement, not a dependency. If it fails (no token, expired license acceptance, model load error), the worker records the reason in `stage_message`, and the pipeline continues producing timestamped transcript + minutes without speaker labels. Meetings are too valuable to lose entirely to one component. Hard failures (bad ffmpeg input, LLM unreachable) mark the job `failed` with the error surfaced to the UI.

## 7. Security model

- Single admin-configured key from `.env`; constant-time comparison; router-level dependency so no endpoint silently escapes auth (PROBLEMS P9 documents exactly such a bug and its fix).
- Uploads validated twice: extension allowlist and hard byte cap while streaming.
- Delete removes database rows and every artifact file.
- Secrets exist only in `.env` (gitignored); `.env.example` documents every knob.
- Known historical incident: a Picovoice key committed in the legacy repo remains in git history by design (rewrite rejected as destructive); revocation at the provider is the remediation (PROBLEMS P4).

## 8. Frontend

Vanilla HTML/CSS/JS served by FastAPI itself - zero npm toolchain, works air-gapped (Decision 5 in PROJECT_GUIDE). Structure:

- `index.html` - semantic sections: connect -> provide audio -> progress -> results -> history, plus toast stack and delete-confirm modal
- `app.js` - fetch client, 1.5 s job polling with stage chips + percent bar, tabbed minutes/transcript/speakers views, rename panel wired to the PATCH endpoint, dependency-free markdown renderer, relative-time history
- `styles.css` - design tokens, button system, focus-visible accessibility states

## 9. Tooling

- `scripts/setup.ps1` - bootstrap venv + deps; installs requirements FIRST and the CUDA torch wheel LAST (order matters, PROBLEMS P8), verifies ffmpeg, seeds `.env`
- `scripts/prewarm_models.py` - cache whisper weights ahead of first meeting (P6 lesson)
- `scripts/watch_job.ps1` - terminal progress bar for any job id
- `backend/tests/` - pytest suites: auth, upload flow, alignment math, minutes batching, speaker naming (35 tests, no GPU needed)

---

*Maintained alongside the code. When architecture changes, update this file and re-run `graphify update .` - the knowledge graph tracks code automatically.*

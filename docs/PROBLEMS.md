# Problems Encountered & Solutions

A honest record of real problems that occurred during this project's rebuild, and how each was solved. No filler — every entry here actually happened.

---

## P1. `pipx` missing / duplicate graphify installations shadowing each other

**When:** Phase 1 (graphify setup), Aug 2026

**Symptom:** `graphify --version` resolved to an old install (`0.8.40`) living in Python 3.13's `Scripts\` folder, while a fresh uv-tool install went to `C:\Users\Admin\.local\bin` which was **not on PATH**. Running the new binary warned "skill is from graphify 0.8.40, package is 0.9.48".

**Root cause:** The package had been pip-installed into the system Python earlier; PATH resolution picked the oldest one first.

**Solution:** Locate both with `(Get-Command graphify).Source`, invoke the uv-tool binary by full path for all operations (`& "C:\Users\Admin\.local\bin\graphify.exe" ...`), and re-ran `graphify install --project` from the new binary so the project skill matched package version.

**Lesson:** On Windows, "command not found"/"wrong version" almost always means multiple installs fighting over PATH. Check `.Source` before reinstalling anything.

---

## P2. Ollama backend failed: missing `openai` package inside graphify's env

**When:** Phase 1, first semantic extraction attempt

**Symptom:** `chunk 1/1 failed: the 'openai' package is required for this backend but is not installed.`

**Root cause:** graphify talks to Ollama through its OpenAI-compatible endpoint, but the base install doesn't ship the `openai` client library.

**Solution:** `uv tool install "graphifyy[ollama]" --force` — reinstalls the tool with the Ollama extra bundled. Re-ran extraction successfully after.

**Lesson:** Optional-dependency extras exist precisely for this; when a backend errors with "package required", install the extra into *the tool's own environment*, not the project venv.

---

## P3. Semantic chunk failed: model `qwen2.5-coder:7b` not found (Ollama 404)

**When:** Phase 1, second extraction attempt

**Symptom:** `Error code: 404 - model 'qwen2.5-coder:7b' not found`, all semantic chunks failed.

**Root cause:** graphify defaults to a specific local model name that was never pulled locally (`ollama list` showed only qwen2.5:3b, mistral, deepseek-r1:1.5b, llama3).

**Solution:** Set `$env:OLLAMA_MODEL = "qwen2.5:3b"` before running extract — graphify honors it. Extraction completed at zero cost, fully offline.

**Lesson:** Local-LLM tooling must be pointed at models you actually have; check `ollama list` first, prefer env-var overrides over pulling multi-GB defaults.

---

## P4. Committed plaintext API key in legacy source (security debt)

**When:** Discovered during initial repo analysis

**Symptom:** A live Picovoice access key hardcoded in `src/final.py` line 13 (and `src/app.py` line 11), permanently part of git history across 7 commits.

**Root cause:** Prototype-era habit of pasting keys directly into code.

**Solution:** All legacy code moved out of the active codebase into `legacy/`; new architecture reads every secret from `.env` (gitignored). **Key rotation at console.picovoice.ai remains outstanding** — history rewrite was deliberately avoided as destructive; revocation achieves the same security outcome.

**Lesson:** Keys in git are compromised forever even after removal from HEAD; rotate rather than delete.

---

## P5. `Remove-Item src` errored "path does not exist" mid-cleanup

**When:** Repo cleanup phase

**Symptom:** After `git mv` of both files and `git rm` of the third, `Remove-Item src` threw PathNotFound.

**Root cause:** Not an actual failure — git removes a directory automatically once its last tracked file leaves it. The error was the script redundantly trying to delete an already-vanished folder.

**Solution:** None needed; recognized as benign. Kept `-ErrorAction SilentlyContinue` pattern for later similar steps.

**Lesson:** Git manages directories implicitly; cleanup scripts should tolerate already-clean state.

---

## P6. Job status label lied during first-run model download

**When:** First end-to-end pipeline run (Batch 2 testing)

**Symptom:** Job polled as `"status": "diarizing", "progress": 18` for minutes straight; user reasonably concluded the pipeline was stuck.

**Root cause:** Two stacked issues. (1) With no HF_TOKEN, diarization skips in milliseconds, so the worker was never really at that stage. (2) The worker only set `status="transcribing"` inside the per-segment progress callback — but faster-whisper downloads its ~1.5 GB model *before* yielding the first segment, so nothing updated the DB during the entire download window.

**Solution (two parts):** worker now sets an explicit `transcribing / Loading transcription model (first run downloads ~1.5 GB)` state before calling transcribe; and added `scripts/prewarm_models.py` so operators can cache the model ahead of time (`python scripts/prewarm_models.py`) instead of ever hitting the blind window.

**Lesson:** Progress reporting must bracket the whole slow operation, not just its productive part; long downloads belong in setup, not in a user's first request.

---

## P7. PowerShell 5.1 writes UTF-8 **with BOM**, breaking Python modules

**When:** Backend build, first syntax sweep

**Symptom:** `ast.parse` failed on `backend/app/__init__.py`: `invalid non-printable character U+FEFF`.

**Root cause:** `Set-Content -Encoding utf8` in Windows PowerShell 5.1 emits a byte-order mark. CPython rejects BOM when parsing source handed to it as text.

**Solution:** Re-wrote the two affected files with Python itself (`open(..., encoding="utf-8"`). Going forward all project files are written by tools, not shell redirection.

**Lesson:** On Windows, "UTF-8" is three different byte layouts depending on which tool wrote it; verify bytes, not labels.

---

## P8. Requirements install silently downgraded CUDA PyTorch to CPU build

**When:** Environment setup

**Symptom:** Verification printed `torch 2.13.0+cpu | cuda available: False` even though `torch==2.5.1+cu121` had been installed moments earlier and GPU is present.

**Root cause:** Installing `-r requirements.txt` afterwards re-resolved the dependency tree against PyPI, where no `+cu121` local version exists — uv/pip "helpfully" replaced torch with the newest CPU wheel, and pulled a mismatched torchaudio along.

**Solution:** Reinstalled the matched pair last: `uv pip install "torch==2.5.1+cu121" "torchaudio==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121`. `scripts/setup.ps1` now installs requirements FIRST and the GPU wheel LAST, with comments explaining why order matters. Post-install check script asserts `cuda.is_available()`.

**Lesson:** GPU wheels are second-class citizens to generic resolvers; always re-assert them after any bulk install, and always verify `cuda.is_available()` rather than trusting versions.

---

## P9. `/api/system/status` answered 200 without any API key

**When:** Batch 1 verification (user-run)

**Symptom:** `curl /api/system/status` returned full engine JSON with no `X-API-Key` header; expected 401. The same guard worked on meetings/jobs routers.

**Root cause:** Router assembly inconsistency - the system router was declared as bare `APIRouter()` (and later `APIRouter(prefix="/system")`) without `dependencies=[Depends(require_api_key)]`, while the other two routers had it. A one-line omission, invisible until exercised.

**Solution:** Added the dependency to the system router declaration; re-verified 401-without-key / 200-with-key.

**Second finding in the same batch:** user's PowerShell aliases `curl` to `Invoke-WebRequest`, which rejects `-H "Header: value"` string syntax - tests must use `curl.exe` explicitly on Windows.

**Lesson:** Security middleware applied at app level vs router level is easy to get subtly wrong; auth tests belong in every endpoint batch, not a one-time checklist item.

---

## P10. pyannote 4.x rejected `use_auth_token` — diarization skipped even with valid HF_TOKEN

**When:** Day 2 session, first run with HF_TOKEN configured (Aug 2026)

**Symptom:** Job completed normally but `stage_message` carried: `Diarization skipped: Could not load diarization model(s). Last error: Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'`. Graceful degradation kicked in: transcript + minutes produced, just without speaker labels.

**Root cause:** `diarization.py` was written against the pyannote 3.x API. Installed package is pyannote.audio 4.0.7, where huggingface_hub renamed the parameter from `use_auth_token` to `token`.

**Solution:** `Pipeline.from_pretrained(model_name, token=settings.hf_token)`.

**Second finding during the same fix:** `_load_pipeline()` called `torch.device("cuda")` without importing torch in that scope (the import lived inside another function). On this CUDA-capable machine it would have raised `NameError` the moment the keyword fix landed. Added `import torch` beside the lazy pyannote import.

**Lesson:** Version-mismatch errors travel in packs. When one API-shape error appears, audit the surrounding lines for sibling mismatches before re-running — otherwise you pay for another full upload-poll cycle per discovery.

---

## P11. pyannote unable to open any audio file: broken torchcodec on Windows + CUDA torch

**When:** Day 2 session, second run (after P10 fix)

**Symptom:** `Diarization skipped: torchcodec is not available. Cannot read audio file. Please install torchcodec or provide audio as a waveform dictionary: {'waveform': (channel, time) torch.Tensor, 'sample_rate': int}` — preceded since first import by `[WinError 127] The specified procedure could not be found`.

**Root cause:** pyannote.audio 4.x decodes media through torchcodec, which links against FFmpeg DLLs matching the exact PyTorch build. No torchcodec wheel pairs cleanly with torch 2.5.1+cu121 on Windows. Repairing it would mean upgrading torch — breaking the deliberately matched CUDA pair installed last in `scripts/setup.ps1` precisely to avoid this class of problem (P8).

**Solution:** Reroute around decoding entirely. The backend already normalizes every upload to 16 kHz mono WAV via ffmpeg (`services/audio.py`), so `diarization.py` now reads that WAV itself with `scipy.io.wavfile` (already in the venv), scales samples to float32 [-1, 1], and passes the pipeline a waveform dict — the exact workaround pyannote's own error text suggests. Note: torchaudio was attempted first and failed too — on Windows it ships without any WAV backend (`Couldn't find appropriate backend to handle uri`).

**Lesson:** A broken optional native dependency is often cheaper to route around than to repair — especially when the repair path threatens a working GPU installation. Also: trust the library's own error message; it literally documented the escape hatch.

---

## P12. pyannote 4 community-1 pipeline returns `DiarizeOutput`, not `Annotation`

**When:** Day 2 session, third run (after P11 fix)

**Symptom:** `'DiarizeOutput' object has no attribute 'itertracks'` — pipeline executed fully (models cached, ~40 s job), then crashed while reading results.

**Root cause:** The community-1 pipeline (pyannote 4's flagship) wraps its output in a new dataclass with three fields: `speaker_diarization`, `exclusive_speaker_diarization`, `speaker_embeddings`. Older pipelines returned the bare `Annotation` that downstream code expected.

**Solution:** `annotation = getattr(output, "speaker_diarization", output)` — transparently handles both the new wrapped shape and the old direct shape, keeping the 3.1 fallback config functional.

**Lesson:** Verified by introspection (`dataclasses.fields`, `typing.get_type_hints`) against the installed package rather than guessing from docs or blog posts. Third consecutive pyannote-version mismatch in one day — the whole P10–P12 series traces back to one root cause: integration code written from memory of an older major version. Introspect the installed package first, then write.

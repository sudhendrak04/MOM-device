import logging
import queue
import threading
import time
import traceback
from pathlib import Path

from .config import settings
from .db import new_session
from .models import Meeting, utcnow

logger = logging.getLogger("mom.worker")

STAGES = ["queued", "decoding", "diarizing", "transcribing", "aligning", "summarizing", "completed"]


def _update(meeting_id: str, **fields) -> None:
    session = new_session()
    try:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None:
            return
        for key, value in fields.items():
            setattr(meeting, key, value)
        session.commit()
    finally:
        session.close()


class Worker:
    """Serial GPU-bound job queue with staged progress reporting."""

    def __init__(self, concurrency: int = 1):
        self._queue: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._concurrency = max(1, concurrency)
        self._stopping = threading.Event()

    def start(self) -> None:
        for index in range(self._concurrency):
            thread = threading.Thread(target=self._run_loop, name=f"mom-worker-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stopping.set()
        for _ in self._threads:
            self._queue.put(None)

    def submit(self, meeting_id: str) -> None:
        self._queue.put(meeting_id)

    def _run_loop(self) -> None:
        while not self._stopping.is_set():
            meeting_id = self._queue.get()
            if meeting_id is None:
                break
            try:
                self.process(meeting_id)
            except Exception:
                logger.exception("Job %s crashed", meeting_id)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    def process(self, meeting_id: str) -> None:
        started = time.monotonic()
        session = new_session()
        try:
            meeting = session.get(Meeting, meeting_id)
            if meeting is None:
                logger.error("Meeting %s vanished from DB", meeting_id)
                return
            uploaded_path = Path(meeting.uploaded_path)
        finally:
            session.close()

        try:
            # ---- decode -------------------------------------------------
            from .services import audio as audio_service

            _update(meeting_id, status="decoding", stage="decoding",
                    stage_message="Converting audio to 16 kHz mono WAV", progress=5)
            wav_path = settings.data_dir / "outputs" / f"{meeting_id}.wav"
            duration = audio_service.decode_to_wav(uploaded_path, wav_path)
            _update(meeting_id, wav_path=str(wav_path), duration_sec=duration,
                    stage_message=f"Decoded {duration:.1f}s of audio", progress=12)

            # ---- diarization (optional) ----------------------------------
            turns = None
            diarization_note = ""
            try:
                from .services import diarization as diar_service

                _update(meeting_id, status="diarizing", stage="diarizing",
                        stage_message="Identifying speakers (pyannote)", progress=18)
                turns = diar_service.diarize(wav_path)
            except Exception as exc:  # noqa: BLE001
                if not settings.allow_no_diarization:
                    raise
                diarization_note = f"Diarization skipped: {exc}"

            # ---- transcription -------------------------------------------
            from .services import transcription as stt_service

            _update(
                meeting_id,
                status="transcribing",
                stage="transcribing",
                stage_message=(
                    "Loading transcription model"
                    " (first run downloads ~1.5 GB, cached afterwards)"
                ),
                progress=15,
            )

            def stt_progress(percent: int) -> None:
                mapped = 30 + int(percent * 0.4)  # 30..70
                _update(meeting_id, status="transcribing", stage="transcribing",
                        stage_message=f"Transcribing ({percent}%)",
                        progress=mapped)

            segments, language = stt_service.transcribe(wav_path, stt_progress)
            _update(meeting_id, language=language, progress=72,
                    stage_message=f"Transcribed {len(segments)} segments")

            # ---- alignment ------------------------------------------------
            _update(meeting_id, status="aligning", stage="aligning",
                    stage_message="Matching words to speakers", progress=75)
            if turns:
                from .services import alignment as align_service

                labeled = align_service.assign_segments(segments, turns)
                transcript_text = align_service.build_transcript(labeled)
                num_speakers = len({t.speaker_number for t in labeled}) or len(turns)
                used_diarization = True
            else:
                from .services import alignment as align_service

                transcript_text = align_service.build_plain_transcript(segments)
                num_speakers = None
                used_diarization = False
            if diarization_note:
                transcript_text = f"# Note\n\n{diarization_note}\n\n{transcript_text}"

            transcript_file = settings.outputs_dir / f"{meeting_id}_transcript.txt"
            transcript_file.write_text(transcript_text, encoding="utf-8")
            _update(meeting_id, transcript=transcript_text, num_speakers=num_speakers,
                    diarization_used=used_diarization)

            # ---- minutes ---------------------------------------------------
            if not transcript_text.strip():
                raise RuntimeError(
                    "No speech detected in the recording - nothing to summarize."
                )

            from .services import minutes as mom_service

            def mom_progress(percent: int) -> None:
                mapped = 80 + int(percent * 0.15)  # 80..95
                _update(meeting_id, status="summarizing", stage="summarizing",
                        stage_message=f"Writing minutes via LLM ({percent}%)",
                        progress=mapped)

            minutes_md, batches = mom_service.generate_minutes(transcript_text, mom_progress)
            minutes_file = settings.outputs_dir / f"{meeting_id}_minutes.md"
            minutes_file.write_text(minutes_md, encoding="utf-8")

            elapsed_ms = int((time.monotonic() - started) * 1000)
            speedup = round(duration / (elapsed_ms / 1000), 2) if elapsed_ms else None
            _update(
                meeting_id,
                minutes_md=minutes_md,
                status="completed",
                stage="completed",
                stage_message=(
                    f"Done. {duration:.0f}s audio -> {elapsed_ms / 1000:.0f}s processing "
                    f"({speedup}x realtime), {batches} LLM batch(es)"
                    + (f" | {diarization_note}" if diarization_note else "")
                ),
                progress=100,
                processing_ms=elapsed_ms,
                completed_at=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for %s: %s\n%s", meeting_id, exc, traceback.format_exc())
            _update(
                meeting_id,
                status="failed",
                stage="failed",
                error=str(exc)[:2000],
                stage_message=f"Failed: {exc}",
                completed_at=utcnow(),
            )


worker = Worker(settings.worker_concurrency)

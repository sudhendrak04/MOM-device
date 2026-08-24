import threading
from dataclasses import dataclass
from pathlib import Path

from ..config import settings


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


_state_lock = threading.Lock()
_model = None
_loaded_signature: str | None = None


def _resolve_device() -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_compute_type(device: str) -> str:
    if settings.compute_type != "auto":
        return settings.compute_type
    return "int8_float16" if device == "cuda" else "int8"


def get_model():
    """Lazy-loaded faster-whisper model singleton (reloads if config changed)."""
    global _model, _loaded_signature
    with _state_lock:
        device = _resolve_device()
        signature = f"{settings.whisper_model}|{device}|{_resolve_compute_type(device)}"
        if _model is not None and signature == _loaded_signature:
            return _model, device
        from faster_whisper import WhisperModel  # lazy heavy import

        compute_type = _resolve_compute_type(device)
        _model = WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=compute_type,
            download_root=str(settings.data_dir / "models"),
        )
        _loaded_signature = signature
        return _model, device


def is_loaded() -> bool:
    return _model is not None


def transcribe(
    wav_path: Path,
    progress_callback=None,
) -> tuple[list[Segment], str]:
    """Transcribe a WAV file. Returns (segments with word timestamps, language)."""
    model, _device = get_model()

    segments_iter, info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,
    )

    total_duration = max(info.duration or 0.0, 0.001)
    segments: list[Segment] = []
    for seg in segments_iter:
        words = [
            Word(start=w.start or seg.start, end=w.end or seg.end, text=w.word.strip())
            for w in (seg.words or [])
            if w.word and w.word.strip()
        ]
        segments.append(Segment(start=seg.start, end=seg.end, text=(seg.text or "").strip(), words=words))
        if progress_callback:
            percent = min(99, int(seg.end / total_duration * 100))
            progress_callback(percent)
    return segments, info.language or "unknown"

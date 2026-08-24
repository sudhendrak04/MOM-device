import threading
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

MERGE_GAP_SEC = 0.3
MIN_TURN_SEC = 0.6


class DiarizationUnavailable(RuntimeError):
    pass


@dataclass
class Turn:
    start: float
    end: float
    speaker_tag: str


_state_lock = threading.Lock()
_pipeline = None
_loaded_model_name: str | None = None


def _load_pipeline():
    """Lazy pyannote pipeline with community-1 -> 3.1 fallback."""
    global _pipeline, _loaded_model_name
    with _state_lock:
        if _pipeline is not None:
            return _pipeline, _loaded_model_name
        if not settings.hf_token:
            raise DiarizationUnavailable(
                "HF_TOKEN is not set. Speaker diarization models are gated on HuggingFace; "
                "accept the license for the pyannote models and put your token in .env"
            )
        from pyannote.audio import Pipeline  # lazy heavy import
        import torch  # noqa: F401 - needed below for pipe.to(torch.device("cuda"))

        last_error: Exception | None = None
        for model_name in (settings.diarization_model, settings.diarization_fallback_model):
            try:
                pipe = Pipeline.from_pretrained(model_name, token=settings.hf_token)
                device = _resolve_torch_device()
                if device == "cuda":
                    pipe.to(torch.device("cuda"))
                _pipeline = pipe
                _loaded_model_name = model_name
                return _pipeline, _loaded_model_name
            except Exception as exc:  # noqa: BLE001 - collect and fall through to fallback
                last_error = exc
        raise DiarizationUnavailable(
            f"Could not load diarization model(s). Last error: {last_error}"
        )


def _resolve_torch_device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def is_loaded() -> bool:
    return _pipeline is not None


def reset() -> None:
    """Drop cached pipeline (used when config/token changes)."""
    global _pipeline, _loaded_model_name
    with _state_lock:
        _pipeline = None
        _loaded_model_name = None


def _read_waveform(wav_path: Path):
    """Decode our normalized mono WAV in-process. pyannote 4.x decodes via
    torchcodec, which is unusable on this Windows + torch 2.5.1+cu121 setup,
    so we hand the pipeline a waveform dict instead of a file path."""
    import numpy as np
    import torch
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(str(wav_path))
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def diarize(wav_path: Path) -> list[Turn]:
    """Return merged speaker turns sorted by start time."""
    pipeline, _name = _load_pipeline()
    output = pipeline(_read_waveform(wav_path))
    # pyannote 4 community-1 wraps the Annotation in a DiarizeOutput; older versions return it directly
    annotation = getattr(output, "speaker_diarization", output)

    raw_turns: list[Turn] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        if segment.end - segment.start < MIN_TURN_SEC:
            continue
        raw_turns.append(Turn(start=float(segment.start), end=float(segment.end), speaker_tag=str(label)))

    raw_turns.sort(key=lambda t: t.start)
    return merge_turns(raw_turns)


def merge_turns(turns: list[Turn]) -> list[Turn]:
    """Merge contiguous same-speaker turns within MERGE_GAP_SEC (ported from legacy logic)."""
    merged: list[Turn] = []
    for turn in turns:
        if merged and turn.speaker_tag == merged[-1].speaker_tag \
                and turn.start <= merged[-1].end + MERGE_GAP_SEC:
            merged[-1].end = max(merged[-1].end, turn.end)
        else:
            merged.append(
                Turn(start=turn.start, end=turn.end, speaker_tag=turn.speaker_tag)
            )
    return merged


def speaker_labels(turns: list[Turn]) -> dict[str, int]:
    """Map pyannote tags to stable 'Speaker N" numbers ordered by first appearance."""
    mapping: dict[str, int] = {}
    for turn in turns:
        if turn.speaker_tag not in mapping:
            mapping[turn.speaker_tag] = len(mapping) + 1
    return mapping

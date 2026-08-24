from dataclasses import dataclass

from .diarization import Turn, speaker_labels
from .transcription import Segment


@dataclass
class LabeledTurn:
    start: float
    end: float
    speaker_number: int
    text: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_segments(
    segments: list[Segment],
    turns: list[Turn],
    nearest_tolerance_sec: float = 1.0,
) -> list[LabeledTurn]:
    """Assign each transcribed segment to the speaker turn with maximum time overlap.

    Segments with zero overlap fall back to the temporally nearest turn when within
    tolerance; otherwise they are dropped (usually silence artifacts).
    """
    if not segments:
        return []
    labels = speaker_labels(turns)

    buckets: dict[str, dict] = {}
    for turn in turns:
        key = f"{turn.start:.3f}-{turn.end:.3f}-{turn.speaker_tag}"
        if key not in buckets:
            buckets[key] = {
                "start": turn.start,
                "end": turn.end,
                "speaker": labels[turn.speaker_tag],
                "segments": [],
            }

    for seg in segments:
        best_key, best_score = None, 0.0
        for key, bucket in buckets.items():
            score = _overlap(seg.start, seg.end, bucket["start"], bucket["end"])
            if score > best_score:
                best_key, best_score = key, score
        if best_key is None:
            best_key = min(
                buckets,
                key=lambda k: min(
                    abs(seg.start - buckets[k]["start"]), abs(seg.end - buckets[k]["end"])
                ),
            )
            nearest_gap = min(
                abs(seg.start - buckets[best_key]["start"]),
                abs(seg.start - buckets[best_key]["end"]),
            )
            if nearest_gap > nearest_tolerance_sec:
                continue
        buckets[best_key]["segments"].append(seg)

    labeled: list[LabeledTurn] = []
    for bucket in buckets.values():
        if not bucket["segments"]:
            continue
        bucket["segments"].sort(key=lambda s: s.start)
        text = " ".join(s.text for s in bucket["segments"]).strip()
        if text:
            labeled.append(
                LabeledTurn(
                    start=bucket["start"],
                    end=bucket["end"],
                    speaker_number=bucket["speaker"],
                    text=text,
                )
            )
    labeled.sort(key=lambda t: t.start)
    return labeled


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def build_transcript(labeled_turns: list[LabeledTurn]) -> str:
    """Human-readable transcript block."""
    lines = [
        f"[{format_timestamp(t.start)} - {format_timestamp(t.end)}] Speaker {t.speaker_number}: {t.text}"
        for t in labeled_turns
    ]
    return "\n\n".join(lines)


def build_plain_transcript(segments: list[Segment]) -> str:
    """Timestamped transcript without speaker labels (no-diarization fallback)."""
    lines = [
        f"[{format_timestamp(s.start)}] {s.text}"
        for s in sorted(segments, key=lambda x: x.start)
    ]
    return "\n\n".join(lines)

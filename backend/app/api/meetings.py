import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy import select

from ..config import settings
from ..db import new_session
from ..models import Meeting
from ..schemas import (
    MeetingDetail,
    MeetingSummary,
    SpeakerNamesUpdate,
    SpeakersResponse,
    SpeakerInfo,
    UploadResult,
)
from ..security import require_api_key
from ..worker import worker

router = APIRouter(prefix="/meetings", dependencies=[Depends(require_api_key)])

_SPEAKER_LINE = re.compile(r"^\[[^\]]+\]\s+Speaker\s+(\d+):\s+(.*)$")
_ANY_SPEAKER_TOKEN = re.compile(r"\bSpeaker\s+(\d+)\b")


@router.post("/upload", response_model=UploadResult)
async def upload_meeting(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Allowed: {sorted(settings.allowed_extensions)}",
        )

    meeting_id = uuid.uuid4().hex
    dest = settings.uploads_dir / f"{meeting_id}{ext}"
    size_limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > size_limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
            out.write(chunk)

    session = new_session()
    try:
        meeting = Meeting(
            id=meeting_id,
            original_filename=Path(file.filename or "upload").name,
            uploaded_path=str(dest),
            status="queued",
            stage="queued",
            stage_message="Queued for processing",
            progress=1,
        )
        session.add(meeting)
        session.commit()
    finally:
        session.close()

    worker.submit(meeting_id)
    return UploadResult(meeting_id=meeting_id, job_id=meeting_id, status="queued")


@router.get("", response_model=list[MeetingSummary])
def list_meetings():
    session = new_session()
    try:
        meetings = session.scalars(select(Meeting).order_by(Meeting.created_at.desc())).all()
        return [MeetingSummary(**_summary(m)) for m in meetings]
    finally:
        session.close()


@router.get("/{meeting_id}", response_model=MeetingDetail)
def get_meeting(meeting_id: str):
    meeting = _get_or_404(meeting_id)
    return MeetingDetail(**_summary(meeting), transcript=meeting.transcript,
                         minutes_md=meeting.minutes_md, error=meeting.error,
                         speaker_names=meeting.speaker_names)


@router.get("/{meeting_id}/speakers", response_model=SpeakersResponse)
def list_speakers(meeting_id: str):
    meeting = _get_or_404(meeting_id)
    return SpeakersResponse(speakers=_speakers_payload(meeting))


@router.patch("/{meeting_id}/speakers", response_model=SpeakersResponse)
def rename_speakers(meeting_id: str, payload: SpeakerNamesUpdate):
    meeting = _get_or_404(meeting_id)
    cleaned = {
        key.strip(): value.strip()[:80]
        for key, value in ((str(k), str(v)) for k, v in payload.names.items())
        if key.strip().isdigit() and value.strip()
    }
    session = new_session()
    try:
        row = session.get(Meeting, meeting_id)
        row.speaker_names_json = json.dumps(cleaned)
        session.commit()
    finally:
        session.close()
    return SpeakersResponse(speakers=_speakers_payload(row))


@router.get("/{meeting_id}/transcript", response_class=PlainTextResponse)
def get_transcript(meeting_id: str):
    meeting = _get_or_404(meeting_id)
    if not meeting.transcript:
        raise HTTPException(404, "Transcript not available yet")
    return _with_speaker_names(meeting, meeting.transcript)


@router.get("/{meeting_id}/minutes", response_class=PlainTextResponse)
def get_minutes(meeting_id: str):
    meeting = _get_or_404(meeting_id)
    if not meeting.minutes_md:
        raise HTTPException(404, "Minutes not available yet")
    return _with_speaker_names(meeting, meeting.minutes_md)


@router.delete("/{meeting_id}", status_code=204)
def delete_meeting(meeting_id: str):
    meeting = _get_or_404(meeting_id)
    paths_to_delete: list[Path] = []
    attrs = ["uploaded_path", "wav_path"]
    suffixes = ["_transcript.txt", "_minutes.md"]
    for path_attr in attrs:
        path_value = getattr(meeting, path_attr)
        if path_value:
            paths_to_delete.append(Path(path_value))
    for suffix in suffixes:
        paths_to_delete.append(settings.outputs_dir / (meeting_id + suffix))
    for path in paths_to_delete:
        path.unlink(missing_ok=True)
    session = new_session()
    try:
        session.delete(session.get(Meeting, meeting_id))
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------- #
def _get_or_404(meeting_id: str) -> Meeting:
    session = new_session()
    try:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None:
            raise HTTPException(404, "Meeting not found")
        return meeting
    finally:
        session.close()


def _speakers_payload(meeting: Meeting) -> list[SpeakerInfo]:
    """Detected speakers from the stored transcript, with each speaker's
    longest utterance as a sample quote so users can identify them by eye."""
    longest: dict[int, str] = {}
    for line in (meeting.transcript or "").splitlines():
        match_ = _SPEAKER_LINE.match(line.strip())
        if match_:
            number, text_ = int(match_.group(1)), match_.group(2).strip()
            if len(text_) > len(longest.get(number, "")):
                longest[number] = text_
    names = meeting.speaker_names
    payload = []
    for number in sorted(longest):
        quote = longest[number]
        if len(quote) > 180:
            quote = quote[:177].rstrip() + "..."
        payload.append(SpeakerInfo(
            speaker_number=number,
            name=names.get(str(number)),
            sample_quote=quote,
        ))
    return payload


def _with_speaker_names(meeting: Meeting, text: str) -> str:
    """Substitute saved speaker names for generic 'Speaker N' tokens."""
    names = meeting.speaker_names
    if not names or not text:
        return text
    return _ANY_SPEAKER_TOKEN.sub(
        lambda m_: names.get(m_.group(1)) or m_.group(0), text
    )


def _summary(m: Meeting) -> dict:
    return {
        "id": m.id,
        "original_filename": m.original_filename,
        "status": m.status,
        "duration_sec": m.duration_sec,
        "num_speakers": m.num_speakers,
        "language": m.language,
        "processing_ms": m.processing_ms,
        "created_at": m.created_at,
        "has_transcript": m.has_transcript,
        "has_minutes": m.has_minutes,
    }

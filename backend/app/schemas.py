from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UploadResult(BaseModel):
    meeting_id: str
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    stage: str
    stage_message: str = ""
    progress: int = 0
    error: str | None = None


class MeetingSummary(BaseModel):
    id: str
    original_filename: str
    status: str
    duration_sec: float | None
    num_speakers: int | None
    language: str | None
    processing_ms: int | None
    created_at: datetime
    has_transcript: bool
    has_minutes: bool


class MeetingDetail(MeetingSummary):
    model_config = ConfigDict()

    transcript: str | None
    minutes_md: str | None
    error: str | None
    speaker_names: dict[str, str] = {}


class SpeakerInfo(BaseModel):
    speaker_number: int
    name: str | None = None
    sample_quote: str = ""


class SpeakersResponse(BaseModel):
    speakers: list[SpeakerInfo]


class SpeakerNamesUpdate(BaseModel):
    names: dict[str, str]


class EngineStatus(BaseModel):
    ffmpeg_available: bool
    ollama_reachable: bool
    ollama_model_available: bool
    ollama_models: list[str]
    whisper_loaded: bool
    whisper_model: str
    whisper_device: str
    diarization_configured: bool
    diarization_loaded: bool
    diarization_model: str
    allow_no_diarization: bool

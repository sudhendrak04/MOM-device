from datetime import datetime, timezone
import json

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    uploaded_path: Mapped[str] = mapped_column(String(1024))
    wav_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(24), default="queued")
    stage_message: Mapped[str] = mapped_column(Text, default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_speakers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)

    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    minutes_md: Mapped[str | None] = mapped_column(Text, nullable=True)

    diarization_used: Mapped[bool] = mapped_column(Boolean, default=False)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    speaker_names_json: Mapped[str] = mapped_column(Text, default="{}", server_default="{}")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript)

    @property
    def has_minutes(self) -> bool:
        return bool(self.minutes_md)

    @property
    def speaker_names(self) -> dict[str, str]:
        """Parsed {speaker_number: display_name} mapping; empty values ignored."""
        try:
            data = json.loads(self.speaker_names_json or "{}")
        except ValueError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): str(value).strip()
            for key, value in data.items()
            if str(value).strip()
        }

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "MOM Server"
    api_key: str
    hf_token: str = ""

    data_dir: Path = Path("data")
    frontend_dir: Path = Path("frontend")

    whisper_model: str = "large-v3-turbo"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_fallback_model: str = "pyannote/speaker-diarization-3.1"
    allow_no_diarization: bool = True

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    device: str = "auto"
    compute_type: str = "auto"

    max_upload_mb: int = 500
    worker_concurrency: int = 1
    batch_size_chars: int = 3000
    batch_overlap_chars: int = 200

    allowed_extensions: set[str] = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".opus", ".flac", ".aac"}
    cors_origins: list[str] = ["*"]

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_url(self) -> str:
        return "sqlite:///" + str(self.data_dir / "mom.db")


settings = Settings()

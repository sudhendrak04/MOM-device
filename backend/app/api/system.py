import shutil
import subprocess

import requests
from fastapi import APIRouter, Depends

from ..config import settings
from ..schemas import EngineStatus
from ..security import require_api_key
from ..services import diarization as diar_service
from ..services import transcription as stt_service

router = APIRouter(prefix="/system", dependencies=[Depends(require_api_key)])


@router.get("/status", response_model=EngineStatus)
def system_status():
    ollama_ok, ollama_models = _check_ollama()
    return EngineStatus(
        ffmpeg_available=_check_ffmpeg(),
        ollama_reachable=ollama_ok,
        ollama_model_available=settings.ollama_model in ollama_models,
        ollama_models=ollama_models,
        whisper_loaded=stt_service.is_loaded(),
        whisper_model=settings.whisper_model,
        whisper_device=settings.device,
        diarization_configured=bool(settings.hf_token),
        diarization_loaded=diar_service.is_loaded(),
        diarization_model=settings.diarization_model,
        allow_no_diarization=settings.allow_no_diarization,
    )


def _check_ollama() -> tuple[bool, list[str]]:
    try:
        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        if response.status_code != 200:
            return False, []
        models = [m.get("name", "") for m in response.json().get("models", [])]
        return True, models
    except requests.RequestException:
        return False, []


def _check_ffmpeg() -> bool:
    if shutil.which("ffprobe") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False

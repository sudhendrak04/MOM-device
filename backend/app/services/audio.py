import json
import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 16000


class FFmpegError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def decode_to_wav(source: Path, destination: Path) -> float:
    """Convert any media file to 16 kHz mono WAV. Returns duration in seconds."""
    if not ffmpeg_available():
        raise FFmpegError(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg (e.g. winget install Gyan.FFmpeg)"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(destination),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise FFmpegError(f"ffmpeg failed: {result.stderr.strip()[:500]}")
    return probe_duration(destination)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr.strip()[:300]}")
    info = json.loads(result.stdout)
    try:
        return round(float(info["format"]["duration"]), 3)
    except (KeyError, ValueError):
        return 0.0

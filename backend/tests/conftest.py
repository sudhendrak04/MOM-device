"""Shared fixtures. Env vars are set BEFORE any app import so settings bind to a temp dir."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

_tmp = tempfile.mkdtemp(prefix="mom_test_")
os.environ.setdefault("API_KEY", "test-key-123")
os.environ["DATA_DIR"] = _tmp
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_headers():
    return {"X-API-Key": os.environ["API_KEY"]}


def make_wav_bytes(seconds: float = 0.2, freq: int = 220) -> bytes:
    """Tiny valid WAV file (sine wave) built with the stdlib only."""
    import io
    import math
    import struct
    import wave

    rate = 8000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for i in range(int(rate * seconds)):
            value = int(12000 * math.sin(2 * math.pi * freq * i / rate))
            wav.writeframesraw(struct.pack("<h", value))
    return buf.getvalue()

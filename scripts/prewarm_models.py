"""Pre-download AI models so the first real meeting doesn't stall.

Usage:  .venv\\Scripts\\python scripts\\prewarm_models.py

Reads WHISPER_MODEL / DATA_DIR from .env when present, downloads the
faster-whisper weights into <DATA_DIR>/models (the exact location the
server uses), then exits. Run once after setup; rerun freely to verify.
"""
import os
import sys
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    env_file = Path(__file__).resolve().parent.parent / path
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    load_dotenv()
    model_name = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    root = Path(os.environ.get("DATA_DIR", "data")) / "models"
    root.mkdir(parents=True, exist_ok=True)

    print(f"[prewarm] model : {model_name}")
    print(f"[prewarm] cache : {root.resolve()}")

    from faster_whisper import WhisperModel

    print("[prewarm] downloading (see progress bar)...")
    WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(root))
    print("[prewarm] DONE - model cached, meetings will start fast.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

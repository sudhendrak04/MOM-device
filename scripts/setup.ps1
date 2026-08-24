# MOM Server - one-time environment bootstrap (Windows)
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== MOM Server setup ==" -ForegroundColor Cyan

# 1. Python check ----------------------------------------------------------
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Write-Host "Python not found. Install Python 3.10+ first." -ForegroundColor Red; exit 1 }
$version = (& python --version) -replace "Python ", ""
Write-Host "Python: $version"

# 2. Virtual environment ---------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    python -m venv .venv
}
$pip = Join-Path (Get-Location) ".venv\Scripts\pip.exe"

# 3. Application dependencies ----------------------------------------------
# NOTE: install these FIRST. They pull a generic (CPU) torch from PyPI.
Write-Host "Installing application dependencies ..."
& $pip install -r requirements.txt
& $pip install -r requirements-dev.txt

# 4. PyTorch GPU wheel - install LAST or the CPU wheel wins ----------------
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "NVIDIA GPU detected -> replacing torch with CUDA build (must be last) ..."
    & $pip install "torch==2.5.1+cu121" "torchaudio==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "No NVIDIA GPU detected -> keeping CPU-only PyTorch."
    & $pip install "torch==2.5.1" "torchaudio==2.5.1"
}

# 5. ffmpeg ----------------------------------------------------------------
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "ffmpeg: found" 
} else {
    Write-Host "ffmpeg NOT found. Install with:  winget install Gyan.FFmpeg  then reopen the terminal." -ForegroundColor Yellow
}

# 6. Configuration ---------------------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from template." -ForegroundColor Green
    Write-Host "  >> EDIT .env NOW: set API_KEY and HF_TOKEN before starting the server."
} else {
    Write-Host ".env already exists - leaving it untouched."
}

Write-Host ""
Write-Host "Setup complete. Next steps:" -ForegroundColor Cyan
Write-Host "  1. edit .env (API_KEY, optionally HF_TOKEN)"
Write-Host "  2. start ollama and pull a model:      ollama pull qwen2.5:7b"
Write-Host "  3. run the server:                     .venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
Write-Host "  4. open http://localhost:8000 in a browser"

# 🎙️ Meeting Transcription & Minutes Generator

A fully local, privacy-first pipeline that records meeting audio, identifies speakers, transcribes speech, and generates structured meeting minutes — all running on your own machine with no cloud dependencies.

---

## 🧠 How It Works

```
Record Audio → Noise Reduction → Speaker Diarization → Transcription → Meeting Minutes
   (mic)       (noisereduce)        (Falcon)           (Whisper.cpp)     (Ollama LLM)
```

---

## ✅ Prerequisites

Before installing, make sure you have the following:

- Windows 10/11 (64-bit)
- [Python 3.9+](https://www.python.org/downloads/) via Anaconda or standard install
- [Git](https://git-scm.com/download/win)
- [CMake](https://cmake.org/download/) (for building Whisper.cpp)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **C++ Desktop Development** workload
- A working microphone

---

## 📦 Installation

### Step 1 — Clone or place the project

Put your project files in a folder. This guide assumes:
```
D:/sudhendra/L&T/test/
```

Update the `BASE_DIR` in `final.py` to match your actual folder path:
```python
BASE_DIR = Path("D:/sudhendra/L&T/test")  # use forward slashes
```

---

### Step 2 — Install Python dependencies

Open Anaconda Prompt or Command Prompt and run:

```bash
pip install streamlit sounddevice soundfile numpy requests noisereduce pvfalcon
```

> **Note:** If `sounddevice` fails, install PortAudio first:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

---

### Step 3 — Build Whisper.cpp

Whisper.cpp is a fast, local speech-to-text engine. Build it from source:

```bash
cd D:/sudhendra/L&T/test
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

After building, confirm `whisper-cli.exe` exists at:
```
D:/sudhendra/L&T/test/whisper.cpp/build/bin/Release/whisper-cli.exe
```

---

### Step 4 — Download a Whisper model

Inside the `whisper.cpp` folder, download a model into the `models/` directory:

```bash
cd D:/sudhendra/L&T/test/whisper.cpp/models

# Small English model (recommended, ~150MB, fast)
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin -o ggml-small.en.bin

# OR quantized version (faster, slightly less accurate)
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin -o ggml-small-q8_0.bin
```

Then update the model name in `final.py` to match the file you downloaded:
```python
for model_name in ["ggml-small.en.bin", "ggml-small-q8_0.bin"]:
```

---

### Step 5 — Set up Picovoice Falcon (Speaker Diarization)

Falcon identifies who is speaking at each moment.

1. Go to [https://console.picovoice.ai/](https://console.picovoice.ai/)
2. Sign up for a free account
3. Copy your **Access Key**
4. Paste it into `final.py` at the top:

```python
PICOVOICE_ACCESS_KEY = "your-key-here"
```

> The free tier supports offline usage and does not send audio to any server.

---

### Step 6 — Install and start Ollama (Meeting Minutes LLM)

Ollama runs the language model locally to generate meeting minutes.

1. Download from [https://ollama.com](https://ollama.com) and install
2. Open a Command Prompt and pull a model:

```bash
# Lightweight and fast (recommended for low-end machines)
ollama pull deepseek-r1:1.5b

# Better quality (requires 8GB+ RAM)
ollama pull mistral:7b
```

3. Start the Ollama server (keep this running in the background):

```bash
ollama serve
```

---

## 🚀 Running the App

```bash
cd D:/sudhendra/L&T/test
streamlit run final.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🖥️ App Walkthrough

| Step | Action | What Happens |
|------|--------|--------------|
| **1** | Choose microphone & duration, click **Start Recording** | Audio is captured and saved |
| **2** | Click **Transcribe with Speaker Diarization** | Falcon identifies speakers, Whisper transcribes each turn |
| **3** | Click **Generate Meeting Minutes** | Ollama LLM produces structured minutes |
| **4** | Download transcript and/or minutes | Files saved to `outputs/` folder |

---

## 📁 Folder Structure

```
D:/sudhendra/L&T/test/
│
├── final.py                        ← Main application
│
├── whisper.cpp/
│   ├── build/bin/Release/
│   │   └── whisper-cli.exe         ← Whisper binary
│   └── models/
│       └── ggml-small.en.bin       ← Whisper model
│
├── audios/                         ← Recorded audio files (auto-created)
└── outputs/                        ← Transcripts and meeting minutes (auto-created)
```

---

## ⚙️ System Status Panel

The sidebar shows the live status of each component:

| Indicator | Meaning |
|-----------|---------|
| ✅ Ollama: Running | LLM server is active |
| ✅ Falcon: Ready | Diarization key is set |
| ✅ Whisper: `model-name` | Model file was found |
| ✅ Noise Reduction: Active | `noisereduce` is installed |

If any show ❌, refer to the relevant installation step above.

---

## 🔧 Configuration Reference

All key settings are at the top of `final.py`:

```python
# Your Picovoice key for speaker diarization
PICOVOICE_ACCESS_KEY = "your-key-here"

# Ollama model to use for minutes generation
OLLAMA_MODEL = "deepseek-r1:1.5b"

# Path to your project folder (use forward slashes on Windows)
BASE_DIR = Path("D:/sudhendra/L&T/test")

# Audio sample rate (do not change unless you know why)
SAMPLE_RATE = 16000
```

---

## 🐛 Troubleshooting

**❌ Whisper: Not Found**
- Run `dir D:\sudhendra\L&T\test\whisper.cpp\models` in Command Prompt
- Check the exact filename and update the model list in `final.py`
- Confirm `whisper-cli.exe` exists in `build/bin/Release/`

**❌ Ollama: Not Running**
- Open a new terminal and run `ollama serve`
- Make sure port 11434 is not blocked by a firewall

**❌ Diarization failed**
- Check your Picovoice access key is correctly pasted (no extra spaces)
- Ensure the recording is at least a few seconds long and has audible speech

**OSError: filename syntax incorrect**
- Never use raw backslashes in Python paths: `"D:\test"` → `"D:/test"` or `r"D:\test"`

**Audio too quiet**
- Check your microphone is set as the default recording device in Windows Sound settings
- Speak closer to the microphone or increase mic volume in system settings

---

## 📊 Expected Performance

| Stage | Time (for 1 min audio) |
|-------|------------------------|
| Recording | Real-time |
| Noise reduction | ~5 seconds |
| Speaker diarization | ~15–30 seconds |
| Transcription | ~1–3 minutes |
| Minutes generation | ~10–30 seconds |
| **Total** | **~2–5 minutes** |

Performance varies based on CPU speed, model size, and number of speakers.

---

## 📄 License

This project uses the following open-source components:
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) — MIT License
- [Ollama](https://github.com/ollama/ollama) — MIT License
- [Picovoice Falcon](https://picovoice.ai/) — Free tier available
- [Streamlit](https://streamlit.io/) — Apache 2.0 License
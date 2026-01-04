import streamlit as st
import subprocess
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import platform

# ================= CONFIG - ADD YOUR PICOVOICE KEY HERE =================
PICOVOICE_ACCESS_KEY = "aQ6jUlCc9R/WlaUTRWle5ARsTAO7c3LqR3RE1aZrjoUJm/dgUKc4+Q=="  # 🔑 Replace with your key from console.picovoice.ai

# Try to import pvfalcon
try:
    import pvfalcon
    FALCON_AVAILABLE = True
except ImportError:
    FALCON_AVAILABLE = False
    st.warning("⚠️ pvfalcon not installed. Install with: pip install pvfalcon")

# ================= AUTO-DETECT OS & PATHS =================
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# ⚙️ YOUR EXACT PATH CONFIGURATION ⚙️
BASE_DIR = Path("C:/Users/Admin/Desktop/test")

if IS_WINDOWS:
    possible_bins = [
        BASE_DIR / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe",
        BASE_DIR / "whisper.cpp" / "build" / "bin" / "Release" / "main.exe",
        BASE_DIR / "whisper.cpp" / "build" / "Release" / "whisper-cli.exe",
        BASE_DIR / "whisper.cpp" / "build" / "Release" / "main.exe",
        BASE_DIR / "whisper.cpp" / "whisper-cli.exe",
        BASE_DIR / "whisper.cpp" / "main.exe",
    ]
    WHISPER_BIN = None
    for bin_path in possible_bins:
        if bin_path.exists():
            WHISPER_BIN = bin_path
            break
    if WHISPER_BIN is None:
        WHISPER_BIN = BASE_DIR / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe"
else:
    BASE_DIR = Path.home() / "test"
    WHISPER_BIN = BASE_DIR / "whisper.cpp" / "main"

# Create directories
AUDIO_DIR = BASE_DIR / "audios"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "whisper.cpp" / "models"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= CONFIG =================
SAMPLE_RATE = 16000
MIN_CHUNK_SEC = 0.6  # Minimum segment duration
MERGE_GAP_SEC = 0.3  # Gap to merge segments from same speaker

# Auto-detect available model
WHISPER_MODEL = None
for model_name in ["ggml-base.en.bin", "ggml-tiny.en.bin"]:
    model_path = MODELS_DIR / model_name
    if model_path.exists() and model_path.stat().st_size > 10_000_000:
        WHISPER_MODEL = model_path
        break

# ================= SESSION STATE =================
if "recorded_audio_path" not in st.session_state:
    st.session_state.recorded_audio_path = None
if "recorded_audio_data" not in st.session_state:
    st.session_state.recorded_audio_data = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "diarized_transcript" not in st.session_state:
    st.session_state.diarized_transcript = None

# ================= AUDIO RECORDING =================
def list_audio_devices():
    """List all available audio input devices"""
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append({
                'index': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate']
            })
    return input_devices

def record_audio(duration_sec: int, device_index: int = None) -> np.ndarray:
    """Record audio from selected device"""
    try:
        audio = sd.rec(
            int(duration_sec * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=device_index
        )
        sd.wait()
        return audio.flatten()
    except Exception as e:
        st.error(f"Recording failed: {str(e)}")
        return np.array([], dtype=np.int16)

def check_audio_quality(audio: np.ndarray) -> dict:
    """Validate audio quality"""
    if audio.size == 0:
        return {"valid": False, "reason": "Empty audio"}
    
    max_amp = np.max(np.abs(audio))
    if max_amp < 100:
        return {"valid": False, "reason": "Audio too quiet"}
    
    rms = np.sqrt(np.mean(audio.astype(float)**2))
    
    return {
        "valid": True,
        "max_amplitude": int(max_amp),
        "rms": int(rms)
    }

# ================= DIARIZATION =================
def diarize_audio(audio_int16: np.ndarray) -> list:
    """
    Perform speaker diarization using Picovoice Falcon
    Returns list of segments with speaker tags and timestamps
    """
    if not FALCON_AVAILABLE:
        st.error("Falcon not available. Install with: pip install pvfalcon")
        return []
    
    try:
        # Initialize Falcon
        falcon = pvfalcon.create(access_key=PICOVOICE_ACCESS_KEY)
        
        # Process audio
        segments = falcon.process(audio_int16)
        
        # Clean up
        falcon.delete()
        
        return segments
        
    except Exception as e:
        st.error(f"Diarization failed: {str(e)}")
        return []

def merge_segments(segments: list) -> list:
    """
    Merge contiguous segments from the same speaker
    """
    if not segments:
        return []
    
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        
        last = merged[-1]
        # Merge if same speaker and within gap threshold
        if (seg.speaker_tag == last.speaker_tag and 
            seg.start_sec <= last.end_sec + MERGE_GAP_SEC):
            # Extend the last segment
            last.end_sec = seg.end_sec
        else:
            merged.append(seg)
    
    return merged

def get_speaker_labels(segments: list) -> dict:
    """
    Create readable speaker labels (Speaker 1, Speaker 2, etc.)
    in order of appearance
    """
    speaker_map = {}
    speaker_id = 1
    
    for seg in segments:
        if seg.speaker_tag not in speaker_map:
            speaker_map[seg.speaker_tag] = f"Speaker {speaker_id}"
            speaker_id += 1
    
    return speaker_map

# ================= WHISPER TRANSCRIPTION =================
def transcribe_audio_segment(audio_segment: np.ndarray, temp_path: Path) -> str:
    """Transcribe a single audio segment"""
    
    # Save segment
    sf.write(temp_path, audio_segment, SAMPLE_RATE)
    
    output_txt = Path(str(temp_path) + ".txt")
    
    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        "-f", str(temp_path),
        "-otxt",
        "-of", str(temp_path),
        "--language", "en",
        "--threads", "4" if IS_WINDOWS else "2",  # 4 for PC, 2 for Pi
        "--no-timestamps"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )
        
        if output_txt.exists():
            transcript = output_txt.read_text(encoding="utf-8").strip()
            output_txt.unlink(missing_ok=True)
            return transcript
        
        return ""
            
    except Exception as e:
        return f"[Error: {str(e)}]"

def transcribe_with_diarization(audio_int16: np.ndarray) -> str:
    """
    Full pipeline: Diarization + Transcription
    Returns formatted transcript with speaker labels
    """
    
    # Step 1: Diarize
    with st.spinner("🔍 Step 1/2: Identifying speakers..."):
        segments = diarize_audio(audio_int16)
    
    if not segments:
        return "[ERROR] Diarization failed or no speech detected"
    
    st.success(f"✅ Found {len(segments)} segments")
    
    # Step 2: Merge segments
    merged_segments = merge_segments(segments)
    st.info(f"📊 Merged into {len(merged_segments)} speaker turns")
    
    # Step 3: Get speaker labels
    speaker_map = get_speaker_labels(merged_segments)
    st.info(f"👥 Detected {len(speaker_map)} unique speakers")
    
    # Step 4: Transcribe each segment
    transcript_lines = []
    temp_chunk_path = AUDIO_DIR / "temp_chunk.wav"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, seg in enumerate(merged_segments):
        # Extract audio segment
        start_sample = int(seg.start_sec * SAMPLE_RATE)
        end_sample = int(seg.end_sec * SAMPLE_RATE)
        chunk = audio_int16[start_sample:end_sample]
        
        duration = len(chunk) / SAMPLE_RATE
        
        # Skip very short segments
        if duration < MIN_CHUNK_SEC:
            continue
        
        # Update progress
        progress = (idx + 1) / len(merged_segments)
        progress_bar.progress(progress)
        status_text.text(f"🔄 Step 2/2: Transcribing segment {idx+1}/{len(merged_segments)}...")
        
        # Transcribe
        text = transcribe_audio_segment(chunk, temp_chunk_path)
        
        if text:
            speaker_label = speaker_map[seg.speaker_tag]
            timestamp = f"[{seg.start_sec:.1f}s - {seg.end_sec:.1f}s]"
            transcript_lines.append(f"{speaker_label} {timestamp}: {text}")
    
    # Cleanup
    temp_chunk_path.unlink(missing_ok=True)
    progress_bar.empty()
    status_text.empty()
    
    if not transcript_lines:
        return "[No speech detected in segments]"
    
    return "\n\n".join(transcript_lines)

def transcribe_audio_simple(audio_path: Path) -> str:
    """Simple transcription without diarization"""
    
    if not WHISPER_MODEL or not WHISPER_MODEL.exists():
        return "[ERROR] Whisper model not found!"
    
    if not WHISPER_BIN or not WHISPER_BIN.exists():
        return f"[ERROR] Whisper binary not found at: {WHISPER_BIN}"
    
    output_txt = Path(str(audio_path) + ".txt")
    
    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        "-f", str(audio_path),
        "-otxt",
        "-of", str(audio_path),
        "--language", "en",
        "--threads", "4" if IS_WINDOWS else "2",
        "--no-timestamps"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )
        
        if output_txt.exists():
            transcript = output_txt.read_text(encoding="utf-8").strip()
            output_txt.unlink(missing_ok=True)
            return transcript if transcript else "[No speech detected]"
        else:
            return f"[ERROR] Output file not created\nSTDERR: {result.stderr[:200]}"
            
    except subprocess.TimeoutExpired:
        return "[ERROR] Transcription timeout (>2 minutes)"
    except Exception as e:
        return f"[ERROR] {str(e)}"

# ================= UI =================
st.set_page_config(page_title="Transcription + Diarization", layout="wide")
st.title("🎙️ Speech Transcription with Speaker Diarization")
st.caption("Identify different speakers and transcribe their speech accurately")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ System Information")
st.sidebar.text(f"OS: {platform.system()}")
st.sidebar.text(f"Python: {platform.python_version()}")

# Falcon status
if FALCON_AVAILABLE:
    if PICOVOICE_ACCESS_KEY != "YOUR_ACCESS_KEY_HERE":
        st.sidebar.success("✅ Falcon Diarization: Ready")
    else:
        st.sidebar.warning("⚠️ Falcon: Add Access Key")
else:
    st.sidebar.error("❌ Falcon: Not Installed")

# Model status
if WHISPER_MODEL and WHISPER_MODEL.exists():
    st.sidebar.success(f"✅ Model: {WHISPER_MODEL.name}")
    size_mb = WHISPER_MODEL.stat().st_size / (1024*1024)
    st.sidebar.text(f"   Size: {size_mb:.0f} MB")
else:
    st.sidebar.error("❌ No Whisper model found")

# Binary status
if WHISPER_BIN and WHISPER_BIN.exists():
    st.sidebar.success(f"✅ Whisper: Found")
else:
    st.sidebar.error("❌ Whisper: Not Found")

# Audio devices
st.sidebar.subheader("🎤 Audio Devices")
devices = list_audio_devices()
if devices:
    for dev in devices:
        st.sidebar.text(f"[{dev['index']}] {dev['name'][:25]}")

# ================= MAIN APP =================

# Device Selection
st.header("1️⃣ Select Microphone")
device_options = {f"[{d['index']}] {d['name']}": d['index'] for d in devices}

if device_options:
    selected_device_name = st.selectbox(
        "Choose your microphone:",
        options=list(device_options.keys())
    )
    selected_device = device_options[selected_device_name]
else:
    st.error("❌ No microphone detected!")
    selected_device = None

# Recording
st.header("2️⃣ Record Audio")

col1, col2 = st.columns(2)
with col1:
    duration = st.number_input("Duration (seconds)", 10, 120, 30, 5)
with col2:
    st.metric("Sample Rate", f"{SAMPLE_RATE} Hz")

st.info("💡 Tip: For best diarization results, record conversations with 2-4 speakers, each speaking clearly.")

if st.button("🎙️ Start Recording", disabled=selected_device is None):
    with st.spinner(f"Recording for {duration} seconds..."):
        progress_bar = st.progress(0)
        
        audio = record_audio(duration, selected_device)
        progress_bar.progress(100)
        
        quality = check_audio_quality(audio)
        
        if not quality["valid"]:
            st.error(f"⚠️ {quality['reason']}")
        else:
            # Save audio
            audio_path = AUDIO_DIR / f"recording_{int(time.time())}.wav"
            sf.write(audio_path, audio, SAMPLE_RATE)
            
            st.session_state.recorded_audio_path = audio_path
            st.session_state.recorded_audio_data = audio
            
            st.success("✅ Recording completed!")
            st.info(f"📊 Max: {quality['max_amplitude']} | RMS: {quality['rms']}")
            
            # Play audio
            st.audio(str(audio_path))

# Transcription Options
if st.session_state.recorded_audio_path:
    st.header("3️⃣ Transcription Options")
    
    st.info(f"📁 Audio file: {st.session_state.recorded_audio_path.name}")
    
    col1, col2 = st.columns(2)
    
    # Option 1: Simple transcription
    with col1:
        st.subheader("Simple Transcription")
        st.caption("Fast transcription without speaker identification")
        
        if st.button("🧠 Transcribe (No Diarization)"):
            with st.spinner("Transcribing..."):
                start_time = time.time()
                
                transcript = transcribe_audio_simple(st.session_state.recorded_audio_path)
                
                elapsed = time.time() - start_time
                st.session_state.transcript = transcript
                
                if not transcript.startswith("[ERROR]"):
                    st.success(f"✅ Completed in {elapsed:.1f}s")
                    st.text_area("Transcript:", transcript, height=200, key="simple_transcript")
                else:
                    st.error(transcript)
    
    # Option 2: Diarization + Transcription
    with col2:
        st.subheader("With Speaker Diarization")
        st.caption("Identifies different speakers (slower)")
        
        if st.button("👥 Transcribe + Diarize", disabled=not FALCON_AVAILABLE):
            start_time = time.time()
            
            transcript = transcribe_with_diarization(st.session_state.recorded_audio_data)
            
            elapsed = time.time() - start_time
            st.session_state.diarized_transcript = transcript
            
            if not transcript.startswith("[ERROR]"):
                st.success(f"✅ Completed in {elapsed:.1f}s")
                st.text_area("Diarized Transcript:", transcript, height=200, key="diarized_transcript")
                
                # Save
                transcript_file = OUTPUT_DIR / f"diarized_{int(time.time())}.txt"
                transcript_file.write_text(transcript)
                
                # Download
                st.download_button(
                    "💾 Download Diarized Transcript",
                    transcript,
                    file_name="diarized_transcript.txt",
                    mime="text/plain"
                )
            else:
                st.error(transcript)

# Reset
st.divider()
if st.button("🔄 Reset & New Recording"):
    st.session_state.clear()
    st.rerun()

# ================= SETUP GUIDE =================
with st.expander("🔧 Setup Guide"):
    st.markdown(f"""
    ## Setup Picovoice Falcon:
    
    ### 1. Install pvfalcon
    ```bash
    pip install pvfalcon
    ```
    
    ### 2. Get Access Key
    - Go to: https://console.picovoice.ai/
    - Sign up (free tier available)
    - Copy your Access Key
    - Add it to line 11 in the code:
    ```python
    PICOVOICE_ACCESS_KEY = "your_actual_key_here"
    ```
    
    ### 3. Test
    - Record audio with 2+ people talking
    - Click "Transcribe + Diarize"
    - Should see: Speaker 1, Speaker 2, etc.
    
    ## Current Status:
    - Falcon Available: {"✅ Yes" if FALCON_AVAILABLE else "❌ No"}
    - Access Key Set: {"✅ Yes" if PICOVOICE_ACCESS_KEY != "YOUR_ACCESS_KEY_HERE" else "❌ No"}
    - Whisper Model: {"✅ Found" if WHISPER_MODEL and WHISPER_MODEL.exists() else "❌ Not Found"}
    
    ## Expected Performance:
    - **PC**: Diarization + Transcription ~2-5x realtime
    - **Pi 4B**: Diarization + Transcription ~3-8x realtime
    """)

with st.expander("📊 How It Works"):
    st.markdown("""
    ## Pipeline:
    
    1. **Record Audio** → Captures conversation
    2. **Diarization (Falcon)** → Identifies speaker segments
       - Detects who spoke when
       - Groups segments by speaker
    3. **Merge Segments** → Combines nearby segments from same speaker
    4. **Transcription (Whisper)** → Converts each segment to text
    5. **Format Output** → Shows "Speaker 1: text", "Speaker 2: text"
    
    ## Tips for Best Results:
    - Clear audio with minimal background noise
    - 2-4 speakers work best
    - Speakers should not overlap too much
    - Minimum 10 seconds of audio
    - Each speaker segment > 0.6 seconds
    """)
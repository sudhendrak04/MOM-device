import streamlit as st
import subprocess
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import platform
import requests
import json

# ================= CONFIG - ADD YOUR PICOVOICE KEY HERE =================
PICOVOICE_ACCESS_KEY = "aQ6jUlCc9R/WlaUTRWle5ARsTAO7c3LqR3RE1aZrjoUJm/dgUKc4+Q=="

# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"
BATCH_SIZE = 3000
OVERLAP = 200

# Try to import pvfalcon
try:
    import pvfalcon
    FALCON_AVAILABLE = True
except ImportError:
    FALCON_AVAILABLE = False

# ================= AUTO-DETECT OS & PATHS =================
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

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

AUDIO_DIR = BASE_DIR / "audios"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "whisper.cpp" / "models"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= AUDIO CONFIG =================
SAMPLE_RATE = 16000
MIN_CHUNK_SEC = 0.6
MERGE_GAP_SEC = 0.3

WHISPER_MODEL = None
for model_name in ["ggml-small-q8_0.bin"]:
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
if "meeting_minutes" not in st.session_state:
    st.session_state.meeting_minutes = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# ================= OLLAMA FUNCTIONS =================
def check_ollama_status():
    """Check if Ollama is running and model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            return True, model_names
        return False, []
    except Exception as e:
        return False, []

def generate_with_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send prompt to Ollama and get response"""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 2000
            }
        }
        
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return f"[ERROR] API returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "[ERROR] Request timeout - try a smaller batch size"
    except Exception as e:
        return f"[ERROR] {str(e)}"

def split_text_into_batches(text: str, batch_size: int, overlap: int) -> list:
    """Split transcript into overlapping batches"""
    batches = []
    start = 0
    
    while start < len(text):
        end = start + batch_size
        
        if end < len(text):
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + batch_size // 2:
                end = para_break
            else:
                sent_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                if sent_break > start + batch_size // 2:
                    end = sent_break + 1
        
        batch = text[start:end].strip()
        if batch:
            batches.append(batch)
        
        start = end - overlap if end < len(text) else end
    
    return batches

def create_batch_prompt(batch_text: str, batch_num: int, total_batches: int) -> str:
    """Create prompt for processing a batch"""
    
    if total_batches == 1:
        prompt = f"""You are a professional meeting transcriber. Convert the following meeting transcript into well-formatted meeting minutes.

TRANSCRIPT:
{batch_text}

Generate structured meeting minutes with these sections:
1. **Meeting Overview** - Date, attendees, purpose
2. **Key Discussion Points** - Main topics discussed with bullet points
3. **Decisions Made** - Clear list of decisions
4. **Action Items** - Tasks assigned with owners (if mentioned)
5. **Next Steps** - Follow-up actions and next meeting details

Format using markdown with clear headings and bullet points. Be concise but comprehensive."""
    
    else:
        prompt = f"""You are processing part {batch_num} of {total_batches} of a meeting transcript.

TRANSCRIPT SEGMENT:
{batch_text}

Extract and summarize:
1. Key discussion points and topics covered
2. Any decisions mentioned
3. Action items or tasks mentioned
4. Important context or information

Be concise. Focus on factual information. Use bullet points."""
    
    return prompt

def create_final_synthesis_prompt(batch_summaries: list) -> str:
    """Create prompt to synthesize all batch summaries into final minutes"""
    
    combined = "\n\n---BATCH SEPARATOR---\n\n".join(batch_summaries)
    
    prompt = f"""You are synthesizing multiple summaries from a meeting transcript into final meeting minutes.

BATCH SUMMARIES:
{combined}

Create comprehensive meeting minutes with these sections:
1. **Meeting Overview** - Infer date, attendees, and purpose from context
2. **Key Discussion Points** - Consolidate all topics discussed, organized logically
3. **Decisions Made** - List all decisions clearly
4. **Action Items** - All tasks with assigned owners (if mentioned)
5. **Next Steps** - Follow-up actions

Format using markdown. Eliminate redundancy. Be professional and concise."""
    
    return prompt

def generate_meeting_minutes(transcript_text: str) -> str:
    """Process transcript and generate meeting minutes"""
    
    batches = split_text_into_batches(transcript_text, BATCH_SIZE, OVERLAP)
    total_batches = len(batches)
    
    st.info(f"📊 Processing transcript in {total_batches} batch(es)")
    
    if total_batches == 1:
        st.write("Processing single batch...")
        prompt = create_batch_prompt(batches[0], 1, 1)
        
        with st.spinner("🤖 Generating meeting minutes..."):
            result = generate_with_ollama(prompt)
        
        return result
    
    else:
        batch_summaries = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, batch in enumerate(batches):
            status_text.text(f"🔄 Processing batch {i+1}/{total_batches}...")
            progress_bar.progress((i + 1) / (total_batches + 1))
            
            prompt = create_batch_prompt(batch, i+1, total_batches)
            summary = generate_with_ollama(prompt)
            
            if summary.startswith("[ERROR]"):
                st.error(f"Batch {i+1} failed: {summary}")
                continue
            
            batch_summaries.append(summary)
            time.sleep(0.5)
        
        status_text.text(f"🔄 Synthesizing final meeting minutes...")
        progress_bar.progress(1.0)
        
        final_prompt = create_final_synthesis_prompt(batch_summaries)
        final_minutes = generate_with_ollama(final_prompt)
        
        progress_bar.empty()
        status_text.empty()
        
        return final_minutes

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

# ================= AUDIO PREPROCESSING =================
def preprocess_audio(audio: np.ndarray) -> np.ndarray:
    """Enhance audio quality before transcription"""
    try:
        import noisereduce as nr
        
        audio_float = audio.astype(np.float32) / 32768.0
        reduced_noise = nr.reduce_noise(
            y=audio_float,
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=0.8
        )
        
        max_val = np.max(np.abs(reduced_noise))
        if max_val > 0:
            reduced_noise = reduced_noise / max_val * 0.95
        
        audio_clean = (reduced_noise * 32768.0).astype(np.int16)
        
        return audio_clean
    except ImportError:
        return audio
    except Exception as e:
        return audio

# ================= DIARIZATION =================
def diarize_audio(audio_int16: np.ndarray) -> list:
    """Perform speaker diarization using Picovoice Falcon"""
    if not FALCON_AVAILABLE:
        st.error("Falcon not available. Install with: pip install pvfalcon")
        return []
    
    try:
        falcon = pvfalcon.create(access_key=PICOVOICE_ACCESS_KEY)
        segments = falcon.process(audio_int16)
        falcon.delete()
        return segments
        
    except Exception as e:
        st.error(f"Diarization failed: {str(e)}")
        return []

def merge_segments(segments: list) -> list:
    """Merge contiguous segments from the same speaker"""
    if not segments:
        return []
    
    merged = []
    
    for seg in segments:
        if not merged:
            merged.append({
                'speaker_tag': seg.speaker_tag,
                'start_sec': seg.start_sec,
                'end_sec': seg.end_sec
            })
            continue
        
        last = merged[-1]
        
        if (seg.speaker_tag == last['speaker_tag'] and 
            seg.start_sec <= last['end_sec'] + MERGE_GAP_SEC):
            last['end_sec'] = seg.end_sec
        else:
            merged.append({
                'speaker_tag': seg.speaker_tag,
                'start_sec': seg.start_sec,
                'end_sec': seg.end_sec
            })
    
    return merged

def get_speaker_labels(segments: list) -> dict:
    """Create readable speaker labels"""
    speaker_map = {}
    speaker_id = 1
    
    for seg in segments:
        speaker_tag = seg['speaker_tag'] if isinstance(seg, dict) else seg.speaker_tag
        
        if speaker_tag not in speaker_map:
            speaker_map[speaker_tag] = f"Speaker {speaker_id}"
            speaker_id += 1
    
    return speaker_map

# ================= WHISPER TRANSCRIPTION =================
def transcribe_audio_segment(audio_segment: np.ndarray, temp_path: Path) -> str:
    """Transcribe a single audio segment"""
    
    sf.write(temp_path, audio_segment, SAMPLE_RATE)
    
    output_txt = Path(str(temp_path) + ".txt")
    
    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        "-f", str(temp_path),
        "-otxt",
        "-of", str(temp_path),
        "--language", "en",
        "--threads", "4" if IS_WINDOWS else "2",
        "--no-timestamps",
        "--best-of", "5",
        "--beam-size", "5",
        "--temperature", "0.0"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            input=""
        )
        
        if output_txt.exists():
            transcript = output_txt.read_text(encoding="utf-8").strip()
            output_txt.unlink(missing_ok=True)
            return transcript
        
        return ""
            
    except Exception as e:
        return f"[Error: {str(e)}]"

def transcribe_with_diarization(audio_int16: np.ndarray) -> str:
    """Full pipeline: Preprocessing → Diarization → Transcription"""
    
    st.info("🔧 Preprocessing audio for better accuracy...")
    audio_int16 = preprocess_audio(audio_int16)
    
    with st.spinner("🔍 Step 1/2: Identifying speakers..."):
        segments = diarize_audio(audio_int16)
    
    if not segments:
        return "[ERROR] Diarization failed or no speech detected"
    
    st.success(f"✅ Found {len(segments)} segments")
    
    merged_segments = merge_segments(segments)
    st.info(f"📊 Merged into {len(merged_segments)} speaker turns")
    
    speaker_map = get_speaker_labels(merged_segments)
    st.info(f"👥 Detected {len(speaker_map)} unique speakers")
    
    transcript_lines = []
    temp_chunk_path = AUDIO_DIR / "temp_chunk.wav"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, seg in enumerate(merged_segments):
        start_sec = seg['start_sec']
        end_sec = seg['end_sec']
        speaker_tag = seg['speaker_tag']
        
        start_sample = int(start_sec * SAMPLE_RATE)
        end_sample = int(end_sec * SAMPLE_RATE)
        chunk = audio_int16[start_sample:end_sample]
        
        duration = len(chunk) / SAMPLE_RATE
        
        if duration < MIN_CHUNK_SEC:
            continue
        
        progress = (idx + 1) / len(merged_segments)
        progress_bar.progress(progress)
        status_text.text(f"🔄 Step 2/2: Transcribing segment {idx+1}/{len(merged_segments)}...")
        
        text = transcribe_audio_segment(chunk, temp_chunk_path)
        
        if text and not text.startswith("[Error"):
            speaker_label = speaker_map[speaker_tag]
            timestamp = f"[{start_sec:.1f}s - {end_sec:.1f}s]"
            transcript_lines.append(f"{speaker_label} {timestamp}: {text}")
    
    temp_chunk_path.unlink(missing_ok=True)
    progress_bar.empty()
    status_text.empty()
    
    if not transcript_lines:
        return "[No speech detected in segments]"
    
    return "\n\n".join(transcript_lines)

# ================= UI =================
st.set_page_config(page_title="Meeting Transcription & Minutes", layout="wide")
st.title("🎙️ Complete Meeting Solution")
st.caption("Record → Transcribe → Diarize → Generate Meeting Minutes")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ System Status")

# Ollama Status
ollama_running, available_models = check_ollama_status()
if ollama_running:
    st.sidebar.success("✅ Ollama: Running")
    if available_models:
        selected_model = st.sidebar.selectbox(
            "Model:",
            options=available_models,
            index=available_models.index(OLLAMA_MODEL) if OLLAMA_MODEL in available_models else 0
        )
        OLLAMA_MODEL = selected_model
else:
    st.sidebar.error("❌ Ollama: Not Running")

# Falcon Status
if FALCON_AVAILABLE:
    if PICOVOICE_ACCESS_KEY != "YOUR_ACCESS_KEY_HERE":
        st.sidebar.success("✅ Falcon: Ready")
    else:
        st.sidebar.warning("⚠️ Falcon: Add Key")
else:
    st.sidebar.error("❌ Falcon: Not Installed")

# Whisper Status
if WHISPER_MODEL and WHISPER_MODEL.exists():
    st.sidebar.success(f"✅ Whisper: {WHISPER_MODEL.name}")
else:
    st.sidebar.error("❌ Whisper: Not Found")

# Audio Processing
st.sidebar.divider()
st.sidebar.subheader("🎛️ Audio Processing")
try:
    import noisereduce
    st.sidebar.success("✅ Noise Reduction: Active")
except ImportError:
    st.sidebar.warning("⚠️ Noise Reduction: Disabled")

# ================= MAIN WORKFLOW =================

# Step 1: Record Audio
st.header("1️⃣ Record Meeting Audio")

devices = list_audio_devices()
device_options = {f"[{d['index']}] {d['name']}": d['index'] for d in devices}

if device_options:
    selected_device_name = st.selectbox(
        "Choose microphone:",
        options=list(device_options.keys())
    )
    selected_device = device_options[selected_device_name]
else:
    st.error("❌ No microphone detected!")
    selected_device = None

col1, col2 = st.columns(2)
with col1:
    duration = st.number_input("Duration (seconds)", 10, 300, 60, 10)
with col2:
    st.metric("Sample Rate", f"{SAMPLE_RATE} Hz")

if st.button("🎙️ Start Recording", disabled=selected_device is None):
    with st.spinner(f"Recording for {duration} seconds..."):
        progress_bar = st.progress(0)
        
        audio = record_audio(duration, selected_device)
        progress_bar.progress(100)
        
        quality = check_audio_quality(audio)
        
        if not quality["valid"]:
            st.error(f"⚠️ {quality['reason']}")
        else:
            audio_path = AUDIO_DIR / f"recording_{int(time.time())}.wav"
            sf.write(audio_path, audio, SAMPLE_RATE)
            
            st.session_state.recorded_audio_path = audio_path
            st.session_state.recorded_audio_data = audio
            
            st.success("✅ Recording completed!")
            st.audio(str(audio_path))

# Step 2: Transcribe with Diarization
if st.session_state.recorded_audio_path:
    st.header("2️⃣ Transcribe & Diarize")
    
    st.info(f"📁 Audio: {st.session_state.recorded_audio_path.name}")
    
    if st.button("👥 Transcribe with Speaker Diarization", disabled=not FALCON_AVAILABLE):
        start_time = time.time()
        
        transcript = transcribe_with_diarization(st.session_state.recorded_audio_data)
        
        elapsed = time.time() - start_time
        st.session_state.diarized_transcript = transcript
        
        if not transcript.startswith("[ERROR]"):
            st.success(f"✅ Completed in {elapsed:.1f}s")
            st.text_area("Diarized Transcript:", transcript, height=300)
            
            # Save transcript
            transcript_file = OUTPUT_DIR / f"transcript_{int(time.time())}.txt"
            transcript_file.write_text(transcript)
            
            st.download_button(
                "💾 Download Transcript",
                transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )
        else:
            st.error(transcript)

# Step 3: Generate Meeting Minutes
if st.session_state.diarized_transcript and not st.session_state.diarized_transcript.startswith("[ERROR]"):
    st.header("3️⃣ Generate Meeting Minutes")
    
    if not ollama_running:
        st.error("❌ Ollama is not running. Start with: `ollama serve`")
    else:
        col1, col2 = st.columns(3)[:2]
        with col1:
            st.metric("Characters", f"{len(st.session_state.diarized_transcript):,}")
        with col2:
            estimated_batches = max(1, len(st.session_state.diarized_transcript) // BATCH_SIZE)
            st.metric("Est. Batches", estimated_batches)
        
        if st.button("🚀 Generate Meeting Minutes", type="primary"):
            st.session_state.processing = True
            
            start_time = time.time()
            
            try:
                result = generate_meeting_minutes(st.session_state.diarized_transcript)
                elapsed = time.time() - start_time
                
                if not result.startswith("[ERROR]"):
                    st.session_state.meeting_minutes = result
                    st.success(f"✅ Meeting minutes generated in {elapsed:.1f}s")
                else:
                    st.error(result)
            
            finally:
                st.session_state.processing = False

# Step 4: Display Meeting Minutes
if st.session_state.meeting_minutes:
    st.header("4️⃣ Meeting Minutes")
    
    st.markdown(st.session_state.meeting_minutes)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "💾 Download Meeting Minutes",
            st.session_state.meeting_minutes,
            file_name=f"meeting_minutes_{int(time.time())}.md",
            mime="text/markdown"
        )
    with col2:
        # Save both transcript and minutes
        minutes_file = OUTPUT_DIR / f"minutes_{int(time.time())}.md"
        minutes_file.write_text(st.session_state.meeting_minutes)
        st.success(f"✅ Saved to: {minutes_file.name}")

# Reset
st.divider()
if st.button("🔄 Start New Meeting"):
    st.session_state.clear()
    st.rerun()

# ================= SETUP GUIDE =================
with st.expander("🔧 Complete Setup Guide"):
    st.markdown("""
    ## Complete Setup Instructions:
    
    ### 1. Install Dependencies
    ```bash
    pip install streamlit sounddevice soundfile numpy pvfalcon requests noisereduce
    ```
    
    ### 2. Setup Whisper.cpp
    ```bash
    # Clone and build whisper.cpp
    cd C:/Users/Admin/Desktop/test/MOM-device
    git clone https://github.com/ggerganov/whisper.cpp
    cd whisper.cpp
    
    # Download model
    cd models
    curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin -o ggml-small.en.bin
    ```
    
    ### 3. Get Picovoice Falcon Key
    - Visit: https://console.picovoice.ai/
    - Sign up (free tier available)
    - Copy Access Key
    - Add to line 12 of code
    
    ### 4. Install & Start Ollama
    ```bash
    # Install Ollama from ollama.com
    
    # Pull a model
    ollama pull deepseek-r1:1.5b
    # OR for better quality:
    ollama pull mistral:7b
    
    # Start Ollama server
    ollama serve
    ```
    
    ### 5. Run the App
    ```bash
    streamlit run complete_meeting_app.py
    ```
    
    ## Complete Pipeline:
    
    1. **Record** → Capture meeting audio
    2. **Preprocess** → Noise reduction
    3. **Diarize** → Identify speakers (Falcon)
    4. **Transcribe** → Convert speech to text (Whisper)
    5. **Generate** → Create meeting minutes (Ollama)
    6. **Export** → Download transcript & minutes
    
    ## Expected Performance:
    - Recording: Real-time
    - Diarization + Transcription: 2-5x real-time
    - Minutes Generation: 10-30 seconds
    - Total: ~3-6 minutes for 1 minute meeting
    """)

with st.expander("📊 Sample Output"):
    st.markdown("""
    ## Sample Meeting Minutes:
    
    ```markdown
    # Meeting Minutes
    
    ## Meeting Overview
    - **Date:** January 5, 2026
    - **Attendees:** Speaker 1, Speaker 2
    - **Duration:** 2 minutes
    
    ## Key Discussion Points
    
    ### Project Status
    - Speaker 1 provided update on Q4 deliverables
    - Team exceeded targets by 15%
    - Technical challenges were resolved ahead of schedule
    
    ### Budget Planning
    - Speaker 2 presented Q1 budget proposal
    - Requested additional resources for new initiatives
    - Discussion on cost optimization strategies
    
    ## Decisions Made
    
    1. ✅ Approve Q1 budget with modifications
    2. ✅ Proceed with technical upgrade plan
    3. ✅ Schedule follow-up in 2 weeks
    
    ## Action Items
    
    - [ ] **Speaker 1** - Finalize technical specifications by Jan 12
    - [ ] **Speaker 2** - Revise budget proposal by Jan 10
    - [ ] **All** - Review documentation before next meeting
    
    ## Next Steps
    
    - Next meeting: January 19, 2026
    - Focus: Technical review and final budget approval
    ```
    """)
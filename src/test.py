import streamlit as st
import subprocess
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import platform
import re

# ================= CONFIG =================
PICOVOICE_ACCESS_KEY = "aQ6jUlCc9R/WlaUTRWle5ARsTAO7c3LqR3RE1aZrjoUJm/dgUKc4+Q=="

try:
    import pvfalcon
    FALCON_AVAILABLE = True
except ImportError:
    FALCON_AVAILABLE = False

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

BASE_DIR = Path("C:/Users/Admin/Desktop/test")

if IS_WINDOWS:
    # Whisper binary
    possible_bins = [
        BASE_DIR / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe",
        BASE_DIR / "whisper.cpp" / "build" / "bin" / "Release" / "main.exe",
        BASE_DIR / "whisper.cpp" / "whisper-cli.exe",
        BASE_DIR / "whisper.cpp" / "main.exe",
    ]
    WHISPER_BIN = next((b for b in possible_bins if b.exists()), possible_bins[0])
    
    # Llama binary
    possible_llama_bins = [
        BASE_DIR / "llama.cpp" / "llama-cli.exe",
        BASE_DIR / "llama.cpp" / "main.exe",
        BASE_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe",
        BASE_DIR / "llama.cpp" / "build" / "bin" / "Release" / "main.exe",
    ]
    LLAMA_BIN = next((b for b in possible_llama_bins if b.exists()), possible_llama_bins[0])
else:
    BASE_DIR = Path.home() / "test"
    WHISPER_BIN = BASE_DIR / "whisper.cpp" / "main"
    LLAMA_BIN = BASE_DIR / "llama.cpp" / "llama-cli"

AUDIO_DIR = BASE_DIR / "audios"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "whisper.cpp" / "models"
LLAMA_MODELS_DIR = BASE_DIR / "llama.cpp" / "models"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
MIN_CHUNK_SEC = 0.6
MERGE_GAP_SEC = 0.3

# Auto-detect Whisper model
WHISPER_MODEL = None
for model_name in ["ggml-base.en.bin", "ggml-tiny.en.bin"]:
    model_path = MODELS_DIR / model_name
    if model_path.exists() and model_path.stat().st_size > 10_000_000:
        WHISPER_MODEL = model_path
        break

# Auto-detect TinyLlama model
LLAMA_MODEL = None
for model_name in ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"]:
    model_path = LLAMA_MODELS_DIR / model_name
    if model_path.exists() and model_path.stat().st_size > 10_000_000:
        LLAMA_MODEL = model_path
        break

# ================= SESSION STATE =================
for key in ["recorded_audio_path", "recorded_audio_data", "transcript", "diarized_transcript", "minutes_of_meeting"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ================= AUDIO FUNCTIONS =================
def list_audio_devices():
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
    try:
        audio = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device_index)
        sd.wait()
        return audio.flatten()
    except Exception as e:
        st.error(f"Recording failed: {str(e)}")
        return np.array([], dtype=np.int16)

def check_audio_quality(audio: np.ndarray) -> dict:
    if audio.size == 0:
        return {"valid": False, "reason": "Empty audio"}
    max_amp = np.max(np.abs(audio))
    if max_amp < 100:
        return {"valid": False, "reason": "Audio too quiet"}
    rms = np.sqrt(np.mean(audio.astype(float)**2))
    return {"valid": True, "max_amplitude": int(max_amp), "rms": int(rms)}

# ================= DIARIZATION =================
def diarize_audio(audio_int16: np.ndarray) -> list:
    if not FALCON_AVAILABLE:
        st.error("Falcon not available")
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
    """Merge contiguous segments from the same speaker - creates new segment objects"""
    if not segments:
        return []
    
    # Convert to mutable dictionaries
    merged = []
    for seg in segments:
        seg_dict = {
            'speaker_tag': seg.speaker_tag,
            'start_sec': seg.start_sec,
            'end_sec': seg.end_sec
        }
        
        if not merged:
            merged.append(seg_dict)
            continue
        
        last = merged[-1]
        # Merge if same speaker and within gap threshold
        if (seg_dict['speaker_tag'] == last['speaker_tag'] and 
            seg_dict['start_sec'] <= last['end_sec'] + MERGE_GAP_SEC):
            # Extend the last segment
            last['end_sec'] = seg_dict['end_sec']
        else:
            merged.append(seg_dict)
    
    return merged

def get_speaker_labels(segments: list) -> dict:
    """Create readable speaker labels - works with both objects and dicts"""
    speaker_map = {}
    speaker_id = 1
    for seg in segments:
        # Handle both object and dict format
        speaker_tag = seg['speaker_tag'] if isinstance(seg, dict) else seg.speaker_tag
        if speaker_tag not in speaker_map:
            speaker_map[speaker_tag] = f"Speaker {speaker_id}"
            speaker_id += 1
    return speaker_map

# ================= WHISPER TRANSCRIPTION =================
def transcribe_audio_segment(audio_segment: np.ndarray, temp_path: Path) -> str:
    sf.write(temp_path, audio_segment, SAMPLE_RATE)
    output_txt = Path(str(temp_path) + ".txt")
    
    cmd = [str(WHISPER_BIN), "-m", str(WHISPER_MODEL), "-f", str(temp_path), "-otxt", "-of", str(temp_path),
           "--language", "en", "--threads", "4" if IS_WINDOWS else "2", "--no-timestamps"]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if output_txt.exists():
            transcript = output_txt.read_text(encoding="utf-8").strip()
            output_txt.unlink(missing_ok=True)
            return transcript
        return ""
    except Exception as e:
        return f"[Error: {str(e)}]"

def transcribe_with_diarization(audio_int16: np.ndarray) -> str:
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
        # Handle both dict and object format
        start_sec = seg['start_sec'] if isinstance(seg, dict) else seg.start_sec
        end_sec = seg['end_sec'] if isinstance(seg, dict) else seg.end_sec
        speaker_tag = seg['speaker_tag'] if isinstance(seg, dict) else seg.speaker_tag
        
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
        if text:
            speaker_label = speaker_map[speaker_tag]
            timestamp = f"[{start_sec:.1f}s - {end_sec:.1f}s]"
            transcript_lines.append(f"{speaker_label} {timestamp}: {text}")
    
    temp_chunk_path.unlink(missing_ok=True)
    progress_bar.empty()
    status_text.empty()
    
    return "\n\n".join(transcript_lines) if transcript_lines else "[No speech detected in segments]"

def transcribe_audio_simple(audio_path: Path) -> str:
    if not WHISPER_MODEL or not WHISPER_MODEL.exists():
        return "[ERROR] Whisper model not found!"
    if not WHISPER_BIN or not WHISPER_BIN.exists():
        return f"[ERROR] Whisper binary not found at: {WHISPER_BIN}"
    
    output_txt = Path(str(audio_path) + ".txt")
    cmd = [str(WHISPER_BIN), "-m", str(WHISPER_MODEL), "-f", str(audio_path), "-otxt", "-of", str(audio_path),
           "--language", "en", "--threads", "4" if IS_WINDOWS else "2", "--no-timestamps"]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
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

# ================= ENHANCED MOM GENERATION =================
def clean_llm_output(raw_output: str) -> str:
    """Clean TinyLlama output - enhanced to remove ALL metadata"""
    lines = raw_output.split('\n')
    clean_lines = []
    
    # Start capturing after we see "assistant" or actual content
    capture_started = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip completely empty lines at start
        if not capture_started and not line_stripped:
            continue
        
        # Skip ALL metadata lines - expanded list
        if any(keyword in line_stripped.lower() for keyword in [
            'loading model', 'llama', 'ggml', 'build', 'model', 'modalities',
            'available commands', '/exit', '/clear', '/read', '/regen',
            'ctrl+c', 'exiting', 'prompt:', 'generation:', 't/s', 'eval time',
            'seed', 'temp =', 'repeat_penalty', 'top_k', 'sampling', 'stop or exit',
            'loading', 'backend'
        ]):
            continue
        
        # Skip lines that are just ASCII art or symbols
        if re.match(r'^[▄█▀▌▐░▒▓│─├┤┬┴┼═║╔╗╚╝>\s]+$', line_stripped):
            continue
        
        # Skip special tokens and chat markers
        chat_markers = ['<|system|>', '<|user|>', '<|assistant|>', '</s>', '<s>', '###']
        if any(marker in line for marker in chat_markers):
            continue
        
        # Start capturing when we see actual content (bullet or substantial text)
        if not capture_started:
            has_bullet = line_stripped.startswith('-') or line_stripped.startswith('•') or line_stripped.startswith('1.')
            has_content = len(line_stripped) > 20 and not any(c in line_stripped for c in ['▄', '█', '▀'])
            if has_bullet or has_content:
                capture_started = True
            else:
                continue
        
        # Skip the original prompt echoing back
        if 'summarize the following meeting' in line_stripped.lower():
            continue
        if 'meeting summary:' in line_stripped.lower() and len(line_stripped) < 30:
            continue
        
        # Add valid lines
        if line_stripped:
            clean_lines.append(line_stripped)
    
    # Join and do final cleanup
    output = '\n'.join(clean_lines)
    
    # Remove any remaining special characters
    output = re.sub(r'[▄█▀▌▐░▒▓]', '', output)
    output = re.sub(r'\s+', ' ', output)  # Normalize whitespace
    output = '\n'.join([l.strip() for l in output.split('\n') if l.strip()])
    
    # Ensure proper bullet formatting
    if output:
        lines = [l.strip() for l in output.split('\n')]
        formatted = []
        for line in lines:
            # If line doesn't start with bullet/number, add bullet
            if line and not line.startswith('-') and not line.startswith('•') and not re.match(r'^\d+\.', line):
                formatted.append(f"- {line}")
            else:
                formatted.append(line)
        output = '\n'.join(formatted)
    
    return output.strip() if output.strip() else "[No meaningful output generated]"

def generate_minutes_of_meeting(transcript: str) -> str:
    """FIXED: Generate Minutes of Meeting using TinyLlama with improved prompt"""
    if not LLAMA_MODEL or not LLAMA_MODEL.exists():
        return "[ERROR] TinyLlama model not found!"
    if not LLAMA_BIN or not LLAMA_BIN.exists():
        return f"[ERROR] Llama binary not found at: {LLAMA_BIN}"
    
    # Truncate transcript
    max_length = 700  # Shorter to prevent hallucination
    if len(transcript) > max_length:
        transcript = transcript[:max_length]
        last_period = transcript.rfind('.')
        if last_period > max_length - 200:
            transcript = transcript[:last_period + 1]
        transcript += "..."
    
    # Improved prompt - more explicit instructions to prevent hallucination
    prompt = f"""<|system|>
You are a meeting minutes assistant. You ONLY summarize what was actually said in the transcript. Do NOT add information that is not in the transcript.</s>
<|user|>
Read this meeting transcript and create ONLY 3-5 bullet points summarizing what was discussed. Base your summary ONLY on the information provided below. Do not make up or assume any information.

TRANSCRIPT:
{transcript}

Create 3-5 bullet points covering the key topics discussed:</s>
<|assistant|>
Based on the transcript, here are the key points:
- """
    
    # More conservative parameters to reduce hallucination
    cmd = [
        str(LLAMA_BIN),
        "-m", str(LLAMA_MODEL),
        "-p", prompt,
        "-n", "150",           # Reduced tokens to prevent rambling
        "-c", "2048",
        "-ngl", "0",
        "-t", "4",
        "-b", "256",           # Smaller batch
        "--temp", "0.3",       # Lower temperature = less creative/hallucination
        "--repeat-penalty", "1.3",  # Higher penalty to avoid repetition
        "--top-k", "20",       # More focused (was 40)
        "--top-p", "0.85",     # More focused (was 0.95)
        "--no-display-prompt", # Don't echo the prompt back
        "-cnv",
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        )
        
        try:
            stdout, stderr = process.communicate(timeout=180)  # 3 min timeout
            
            if process.returncode == 0 or stdout:
                output = clean_llm_output(stdout)
                
                # Validate output quality
                if len(output) < 15:
                    st.warning("⚠️ Very short output. Raw output:")
                    st.code(stdout[:800])
                    return output
                
                # Check if output seems to be hallucinating timestamps/speakers
                if 'speaker 1' in output.lower() and 'speaker 1' not in transcript.lower():
                    st.warning("⚠️ Model may be hallucinating. Using simpler extraction...")
                    # Fallback: just extract first few sentences
                    sentences = transcript.split('.')[:3]
                    return "- " + "\n- ".join([s.strip() for s in sentences if s.strip()])
                
                return output
            else:
                return f"[ERROR] TinyLlama failed (exit code {process.returncode})\n\nSTDERR:\n{stderr[:400]}"
        
        except subprocess.TimeoutExpired:
            process.kill()
            return "[ERROR] Timeout after 3 minutes."
    
    except Exception as e:
        return f"[ERROR] Exception: {type(e).__name__}: {str(e)}"

# ================= UI =================
st.set_page_config(page_title="Transcription + Diarization", layout="wide")
st.title("🎙️ Speech Transcription with Speaker Diarization")
st.caption("Identify different speakers and transcribe their speech accurately")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ System Information")
st.sidebar.text(f"OS: {platform.system()}")
st.sidebar.text(f"Python: {platform.python_version()}")

if FALCON_AVAILABLE:
    st.sidebar.success("✅ Falcon Diarization: Ready")
else:
    st.sidebar.error("❌ Falcon: Not Installed")

if WHISPER_MODEL and WHISPER_MODEL.exists():
    st.sidebar.success(f"✅ Whisper: {WHISPER_MODEL.name}")
    size_mb = WHISPER_MODEL.stat().st_size / (1024*1024)
    st.sidebar.text(f"   Size: {size_mb:.0f} MB")
else:
    st.sidebar.error("❌ No Whisper model found")

if WHISPER_BIN and WHISPER_BIN.exists():
    st.sidebar.success(f"✅ Whisper Binary: Found")
else:
    st.sidebar.error("❌ Whisper Binary: Not Found")

if LLAMA_BIN and LLAMA_BIN.exists():
    st.sidebar.success(f"✅ Llama Binary: Found")
else:
    st.sidebar.error("❌ Llama Binary: Not Found")

if LLAMA_MODEL and LLAMA_MODEL.exists():
    st.sidebar.success(f"✅ TinyLlama Model: Found")
    st.sidebar.text(f"   {LLAMA_MODEL.name[:30]}")
else:
    st.sidebar.warning("⚠️ TinyLlama model not found")

st.sidebar.subheader("🎤 Audio Devices")
devices = list_audio_devices()
if devices:
    for dev in devices:
        st.sidebar.text(f"[{dev['index']}] {dev['name'][:25]}")

# ================= MAIN APP =================
st.header("1️⃣ Select Microphone")
device_options = {f"[{d['index']}] {d['name']}": d['index'] for d in devices}

if device_options:
    selected_device_name = st.selectbox("Choose your microphone:", options=list(device_options.keys()))
    selected_device = device_options[selected_device_name]
else:
    st.error("❌ No microphone detected!")
    selected_device = None

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
            audio_path = AUDIO_DIR / f"recording_{int(time.time())}.wav"
            sf.write(audio_path, audio, SAMPLE_RATE)
            st.session_state.recorded_audio_path = audio_path
            st.session_state.recorded_audio_data = audio
            st.success("✅ Recording completed!")
            st.info(f"📊 Max: {quality['max_amplitude']} | RMS: {quality['rms']}")
            st.audio(str(audio_path))

# Transcription Options
if st.session_state.recorded_audio_path:
    st.header("3️⃣ Transcription Options")
    st.info(f"📁 Audio file: {st.session_state.recorded_audio_path.name}")
    
    col1, col2 = st.columns(2)
    
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
                
                transcript_file = OUTPUT_DIR / f"diarized_{int(time.time())}.txt"
                transcript_file.write_text(transcript)
                
                st.download_button("💾 Download Diarized Transcript", transcript, 
                                 file_name="diarized_transcript.txt", mime="text/plain")
            else:
                st.error(transcript)

# Minutes of Meeting Generation
if st.session_state.diarized_transcript and not st.session_state.diarized_transcript.startswith("[ERROR]"):
    st.header("4️⃣ Generate Minutes of Meeting")
    st.info("💡 Using TinyLlama for summary generation (optimized for Pi)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test TinyLlama", disabled=not (LLAMA_BIN and LLAMA_BIN.exists() and LLAMA_MODEL)):
            with st.spinner("Testing TinyLlama..."):
                try:
                    test_cmd = [
                        str(LLAMA_BIN), 
                        "-m", str(LLAMA_MODEL), 
                        "-p", "Hello", 
                        "-n", "10",
                        "-ngl", "0",
                        "-t", "4",
                        "-c", "512",
                        "--temp", "0.1",
                        "--log-disable",
                        "-cnv",
                    ]
                    
                    process = subprocess.Popen(
                        test_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
                    )
                    
                    try:
                        stdout, stderr = process.communicate(timeout=30)
                        
                        if process.returncode == 0 or stdout:
                            st.success("✅ TinyLlama is working!")
                            if stdout:
                                st.code(stdout[:500])
                            st.info("🎉 Model loaded successfully!")
                        else:
                            st.warning(f"⚠️ Process exited with code {process.returncode}")
                            if stderr:
                                st.code(f"STDERR:\n{stderr[:400]}")
                            if stdout:
                                st.code(f"STDOUT:\n{stdout[:400]}")
                    
                    except subprocess.TimeoutExpired:
                        process.kill()
                        st.error("❌ Test timed out after 30 seconds")
                        
                except FileNotFoundError:
                    st.error(f"❌ Binary not found at: {LLAMA_BIN}")
                except PermissionError:
                    st.error(f"❌ Permission denied. Check if binary is executable.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        if st.button("📝 Generate Meeting Minutes", disabled=not (LLAMA_BIN and LLAMA_BIN.exists() and LLAMA_MODEL)):
            with st.spinner("🤖 Generating Minutes... This may take 2-4 minutes"):
                start_time = time.time()
                minutes = generate_minutes_of_meeting(st.session_state.diarized_transcript)
                elapsed = time.time() - start_time
                st.session_state.minutes_of_meeting = minutes
                
                if not minutes.startswith("[ERROR]"):
                    st.success(f"✅ Minutes generated in {elapsed:.1f}s")
                    st.subheader("📋 Minutes of Meeting")
                    st.text_area("Meeting Minutes:", minutes, height=300, key="mom_output")
                    
                    mom_file = OUTPUT_DIR / f"mom_{int(time.time())}.txt"
                    mom_file.write_text(minutes)
                    
                    st.download_button("💾 Download Minutes", minutes, 
                                     file_name="minutes_of_meeting.txt", mime="text/plain")
                else:
                    st.error(minutes)
                    with st.expander("🔍 Show Full Debug Output"):
                        st.code(minutes)
    
    if st.session_state.minutes_of_meeting and not st.session_state.minutes_of_meeting.startswith("[ERROR]"):
        st.subheader("📋 Saved Minutes of Meeting")
        st.text_area("Minutes:", st.session_state.minutes_of_meeting, height=250, key="mom_display")

st.divider()
if st.button("🔄 Reset & New Recording"):
    st.session_state.clear()
    st.rerun()


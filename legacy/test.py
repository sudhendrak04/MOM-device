import streamlit as st
import requests
import json
from pathlib import Path
import time

# ================= CONFIG =================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"  # Change to your preferred model

# Batch processing settings
BATCH_SIZE = 3000  # Characters per batch (adjust based on model context)
OVERLAP = 200  # Character overlap between batches for context

# ================= SESSION STATE =================
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
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
                "temperature": 0.3,  # Lower temperature for more focused output
                "top_p": 0.9,
                "num_predict": 2000  # Max tokens to generate
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

# ================= TEXT PROCESSING =================
def split_text_into_batches(text: str, batch_size: int, overlap: int) -> list:
    """Split transcript into overlapping batches"""
    batches = []
    start = 0
    
    while start < len(text):
        end = start + batch_size
        
        # Try to break at a sentence/paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + batch_size // 2:
                end = para_break
            else:
                # Look for sentence break
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
        
        # Move start position with overlap
        start = end - overlap if end < len(text) else end
    
    return batches

def create_batch_prompt(batch_text: str, batch_num: int, total_batches: int) -> str:
    """Create prompt for processing a batch"""
    
    if total_batches == 1:
        # Single batch - generate complete minutes
        prompt = f""" \no_think 
        You are a professional meeting transcriber. Convert the following meeting transcript into well-formatted meeting minutes.

TRANSCRIPT:
{batch_text}
\no_think 
Generate structured meeting minutes with these sections:
1. **Meeting Overview** - Date, attendees, purpose
2. **Key Discussion Points** - Main topics discussed with bullet points
3. **Decisions Made** - Clear list of decisions
4. **Action Items** - Tasks assigned with owners (if mentioned)
5. **Next Steps** - Follow-up actions and next meeting details

Format using markdown with clear headings and bullet points. Be concise but comprehensive."""
    
    else:
        # Multi-batch processing
        prompt = f"""\no_think 
        You are processing part {batch_num} of {total_batches} of a meeting transcript.

TRANSCRIPT SEGMENT:
{batch_text}
\no_think 
Extract and summarize:
1. Key discussion points and topics covered
2. Any decisions mentioned
3. Action items or tasks mentioned
4. Important context or information

Be concise. Focus on factual information. Use bullet points."""
    
    return prompt

def create_final_synthesis_prompt(batch_summaries: list) -> str:
    """\no_think 
    Create prompt to synthesize all batch summaries into final minutes"""
    
    combined = "\n\n---BATCH SEPARATOR---\n\n".join(batch_summaries)
    
    prompt = f"""\no_think 
    You are synthesizing multiple summaries from a meeting transcript into final meeting minutes.

BATCH SUMMARIES:
{combined}
\no_think 
Create comprehensive meeting minutes with these sections:
1. **Meeting Overview** - Infer date, attendees, and purpose from context
2. **Key Discussion Points** - Consolidate all topics discussed, organized logically
3. **Decisions Made** - List all decisions clearly
4. **Action Items** - All tasks with assigned owners (if mentioned)
5. **Next Steps** - Follow-up actions

Format using markdown. Eliminate redundancy. Be professional and concise."""
    
    return prompt

# ================= MAIN PROCESSING FUNCTION =================
def generate_meeting_minutes(transcript_text: str) -> str:
    """Process transcript and generate meeting minutes"""
    
    # Split into batches
    batches = split_text_into_batches(transcript_text, BATCH_SIZE, OVERLAP)
    total_batches = len(batches)
    
    st.info(f"📊 Processing transcript in {total_batches} batch(es)")
    
    if total_batches == 1:
        # Single batch - direct processing
        st.write("Processing single batch...")
        prompt = create_batch_prompt(batches[0], 1, 1)
        
        with st.spinner("🤖 Generating meeting minutes..."):
            result = generate_with_ollama(prompt)
        
        return result
    
    else:
        # Multi-batch processing
        batch_summaries = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each batch
        for i, batch in enumerate(batches):
            status_text.text(f"🔄 Processing batch {i+1}/{total_batches}...")
            progress_bar.progress((i + 1) / (total_batches + 1))
            
            prompt = create_batch_prompt(batch, i+1, total_batches)
            summary = generate_with_ollama(prompt)
            
            if summary.startswith("[ERROR]"):
                st.error(f"Batch {i+1} failed: {summary}")
                continue
            
            batch_summaries.append(summary)
            time.sleep(0.5)  # Brief pause between requests
        
        # Synthesize final minutes
        status_text.text(f"🔄 Synthesizing final meeting minutes...")
        progress_bar.progress(1.0)
        
        final_prompt = create_final_synthesis_prompt(batch_summaries)
        final_minutes = generate_with_ollama(final_prompt)
        
        progress_bar.empty()
        status_text.empty()
        
        return final_minutes

# ================= UI =================
st.set_page_config(page_title="Meeting Minutes Generator", layout="wide")
st.title("📝 Meeting Minutes Generator")
st.caption("Convert speaker-diarized transcripts into professional meeting minutes using Ollama")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Configuration")

# Check Ollama status
ollama_running, available_models = check_ollama_status()

if ollama_running:
    st.sidebar.success("✅ Ollama: Running")
    
    if available_models:
        st.sidebar.write("**Available Models:**")
        for model in available_models:
            st.sidebar.text(f"  • {model}")
        
        # Model selector
        selected_model = st.sidebar.selectbox(
            "Select Model:",
            options=available_models,
            index=available_models.index(OLLAMA_MODEL) if OLLAMA_MODEL in available_models else 0
        )
        OLLAMA_MODEL = selected_model
    else:
        st.sidebar.warning("No models found")
else:
    st.sidebar.error("❌ Ollama: Not Running")
    st.sidebar.code("ollama serve")

# Batch settings
st.sidebar.subheader("📦 Batch Settings")
BATCH_SIZE = st.sidebar.slider("Batch Size (chars)", 1000, 5000, 3000, 500)
OVERLAP = st.sidebar.slider("Overlap (chars)", 0, 500, 200, 50)

st.sidebar.info(f"💡 Smaller batches = more API calls but better handling of long transcripts")

# ================= MAIN APP =================

# Step 1: Upload transcript
st.header("1️⃣ Upload Transcript")

uploaded_file = st.file_uploader(
    "Upload your transcript (.txt file from Whisper + Falcon)",
    type=['txt'],
    help="Upload the diarized transcript file generated from your audio"
)

if uploaded_file is not None:
    # Read transcript
    transcript_text = uploaded_file.read().decode('utf-8')
    st.session_state.transcript_text = transcript_text
    
    # Display preview
    st.success("✅ Transcript loaded!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Characters", f"{len(transcript_text):,}")
    with col2:
        st.metric("Words", f"{len(transcript_text.split()):,}")
    with col3:
        estimated_batches = max(1, len(transcript_text) // BATCH_SIZE)
        st.metric("Est. Batches", estimated_batches)
    
    with st.expander("📄 Preview Transcript"):
        st.text_area("", transcript_text[:2000] + ("..." if len(transcript_text) > 2000 else ""), height=200)

# Step 2: Generate Minutes
if st.session_state.transcript_text:
    st.header("2️⃣ Generate Meeting Minutes")
    
    if not ollama_running:
        st.error("❌ Ollama is not running. Please start Ollama first:")
        st.code("ollama serve")
    else:
        if st.button("🚀 Generate Meeting Minutes", type="primary", disabled=st.session_state.processing):
            st.session_state.processing = True
            
            start_time = time.time()
            
            try:
                result = generate_meeting_minutes(st.session_state.transcript_text)
                elapsed = time.time() - start_time
                
                if not result.startswith("[ERROR]"):
                    st.session_state.meeting_minutes = result
                    st.success(f"✅ Meeting minutes generated in {elapsed:.1f}s")
                else:
                    st.error(result)
            
            finally:
                st.session_state.processing = False

# Step 3: Display and Download
if st.session_state.meeting_minutes:
    st.header("3️⃣ Meeting Minutes")
    
    # Display minutes
    st.markdown(st.session_state.meeting_minutes)
    
    # Download button
    st.download_button(
        "💾 Download Meeting Minutes",
        st.session_state.meeting_minutes,
        file_name=f"meeting_minutes_{int(time.time())}.md",
        mime="text/markdown"
    )
    
    # Copy to clipboard
    st.code(st.session_state.meeting_minutes, language="markdown")

# Reset
st.divider()
if st.button("🔄 Start New Session"):
    st.session_state.clear()
    st.rerun()

# ================= SETUP GUIDE =================
with st.expander("🔧 Setup Guide"):
    st.markdown("""
    ## Quick Setup:
    
    ### 1. Install Ollama
    ```bash
    # Linux/Mac
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Windows - Download from ollama.com
    ```
    
    ### 2. Pull a Model
    ```bash
    # Lightweight (1.5GB) - Fast but basic
    ollama pull deepseek-r1:1.5b
    
    # Recommended (4GB) - Better quality
    ollama pull mistral:7b
    
    # Best quality (8GB+)
    ollama pull llama3:8b
    ```
    
    ### 3. Start Ollama
    ```bash
    ollama serve
    ```
    
    ### 4. Run This App
    ```bash
    streamlit run meeting_minutes_generator.py
    ```
    
    ## How It Works:
    
    1. **Upload** - Load your diarized transcript (.txt)
    2. **Batch Processing** - Splits long transcripts into manageable chunks
    3. **AI Analysis** - Each batch is processed by Ollama
    4. **Synthesis** - Combines batch summaries into final minutes
    5. **Download** - Export as markdown
    
    ## Tips:
    
    - Use **smaller batches** (1500-2500 chars) for longer transcripts
    - Use **larger models** (7B+) for better quality minutes
    - Adjust **overlap** to maintain context between batches
    - For meetings under 5000 chars, single batch is fastest
    """)

with st.expander("📊 Expected Output Format"):
    st.markdown("""
    ## Sample Meeting Minutes:
    
    ```markdown
    # Meeting Minutes
    
    ## Meeting Overview
    - **Date:** January 5, 2026
    - **Attendees:** Speaker 1, Speaker 2, Speaker 3
    - **Purpose:** Project status review and planning
    
    ## Key Discussion Points
    
    ### Project Timeline
    - Current phase completion ahead of schedule
    - Resource allocation concerns raised
    - Need for additional testing identified
    
    ### Budget Review
    - Q1 spending within limits
    - Request for additional budget in Q2
    
    ## Decisions Made
    
    1. ✅ Approve additional testing phase
    2. ✅ Schedule follow-up meeting for budget discussion
    3. ✅ Assign Sarah to lead testing coordination
    
    ## Action Items
    
    - [ ] **John** - Prepare detailed budget proposal by Jan 15
    - [ ] **Sarah** - Set up testing environment by Jan 10
    - [ ] **Team** - Review technical specs by Jan 8
    
    ## Next Steps
    
    - Next meeting: January 15, 2026
    - Focus: Budget approval and testing results
    ```
    """)
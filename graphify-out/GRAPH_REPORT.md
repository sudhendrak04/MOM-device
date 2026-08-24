# Graph Report - MOM-device  (2026-08-23)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 84 nodes · 91 edges · 21 communities (5 shown, 16 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 545 input · 170 output

## Graph Freshness
- Built from commit: `c2702865`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Audio Processing
- Meeting Minutes Generation
- Audio Diarization and Transcription
- Meeting Minutes Generation
- README
- System Status Panel
- Configuration Reference
- Troubleshooting
- Expected Performance
- License
- Step 1 — Clone or place the project
- Step 2 — Install Python dependencies
- Step 3 — Build Whisper.cpp
- Step 4 — Download a Whisper model
- Step 5 — Set up Picovoice Falcon (Speaker Diarization)
- LLM Integration
- App Operations
- App Walkthrough
- Project Structure
- Prerequisites

## God Nodes (most connected - your core abstractions)
1. `transcribe_with_diarization()` - 8 edges
2. `transcribe_with_diarization()` - 7 edges
3. `generate_meeting_minutes()` - 6 edges
4. `generate_meeting_minutes()` - 6 edges
5. `transcribe_audio_segment()` - 5 edges
6. `transcribe_audio_segment()` - 5 edges
7. `diarize_audio()` - 4 edges
8. `diarize_audio()` - 4 edges
9. `preprocess_audio()` - 4 edges
10. `check_audio_quality()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `transcribe_with_diarization()` --calls--> `merge_segments()`  [EXTRACTED]
  src/final.py → src/final.py  _Bridges community 1 → community 2_

## Import Cycles
- None detected.

## Communities (21 total, 16 thin omitted)

### Community 0 - "Audio Processing"
Cohesion: 0.14
Nodes (20): check_audio_quality(), diarize_audio(), get_speaker_labels(), list_audio_devices(), merge_segments(), ndarray, Path, Validate audio quality (+12 more)

### Community 1 - "Meeting Minutes Generation"
Cohesion: 0.15
Nodes (16): check_ollama_status(), create_batch_prompt(), create_final_synthesis_prompt(), generate_meeting_minutes(), generate_with_ollama(), list_audio_devices(), merge_segments(), Send prompt to Ollama and get response (+8 more)

### Community 2 - "Audio Diarization and Transcription"
Cohesion: 0.15
Nodes (16): check_audio_quality(), diarize_audio(), get_speaker_labels(), preprocess_audio(), ndarray, Path, Record audio from selected device, Validate audio quality (+8 more)

### Community 3 - "Meeting Minutes Generation"
Cohesion: 0.21
Nodes (12): check_ollama_status(), create_batch_prompt(), create_final_synthesis_prompt(), generate_meeting_minutes(), generate_with_ollama(), Create prompt for processing a batch, \no_think Create prompt to synthesize all batch summaries into final minutes, Process transcript and generate meeting minutes (+4 more)

## Knowledge Gaps
- **16 isolated node(s):** `Step 1 — Clone or place the project`, `Step 2 — Install Python dependencies`, `Step 3 — Build Whisper.cpp`, `Step 4 — Download a Whisper model`, `Step 5 — Set up Picovoice Falcon (Speaker Diarization)` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `transcribe_audio_segment()` connect `Audio Diarization and Transcription` to `Meeting Minutes Generation`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Why does `transcribe_with_diarization()` connect `Audio Diarization and Transcription` to `Meeting Minutes Generation`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **What connects `Step 1 — Clone or place the project`, `Step 2 — Install Python dependencies`, `Step 3 — Build Whisper.cpp` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Audio Processing` be split into smaller, more focused modules?**
  _Cohesion score 0.1380952380952381 - nodes in this community are weakly interconnected._
- **Should `Meeting Minutes Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.14705882352941177 - nodes in this community are weakly interconnected._
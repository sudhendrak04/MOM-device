# MOM-device vs Commercial Meeting Tools - Benchmark

_Collected August 2026. Our numbers are **measured on the development machine** (Intel Ultra 9 185H, RTX 2000 Ada 8 GB, warm models); competitor figures come from their official pricing pages and published documentation as of that date. Closed-source engines only publish marketing claims, so accuracy rows compare our *published model benchmarks* against their *claims* - noted honestly below._

---

## Our measured performance

| Run | Audio | Processing | Speed | Notes |
|---|---|---|---|---|
| Sample fixture | 20 s | 38 s | 0.53x | fixed costs dominate short audio |
| AMI ES2002a (real meeting) | 1273 s / 21.2 min | 234 s | **5.44x realtime** | 4 speakers correctly labeled, 6 LLM batches |

Underlying model quality (published benchmarks, see PROJECT_GUIDE section 4):

- STT: Whisper large-v3-turbo - ~7.75% avg WER, 99 languages incl. Hindi/Hinglish
- Diarization: pyannote community-1 - 17.0% DER on AMI corpus (2026 open-source SOTA)
- Minutes: local LLM via Ollama, batched map-reduce for long transcripts

---

## Feature comparison

| Dimension | **MOM-device (ours)** | **Fireflies.ai** | **Otter.ai** | **MeetGeek** | **tl;dv** |
|---|---|---|---|---|---|
| **Deployment** | Self-hosted, one PC | Cloud | Cloud | Cloud | Cloud |
| **Audio leaves office?** | **Never** (air-gap capable) | Yes | Yes | Yes | Yes |
| **Measured speed** | **5.44x realtime** | not disclosed | near-realtime (live captions) | not disclosed | not disclosed |
| **Transcription engine** | Whisper large-v3-turbo (~7.75% WER class) | proprietary | proprietary | proprietary | proprietary |
| **Diarization** | pyannote community-1 (17% DER, published SOTA open) | proprietary | proprietary | proprietary | proprietary |
| **Languages** | 99 incl. Hindi/Hinglish | English-first, 100+ claimed | ~6 | 100+ | English-primary |
| **Minutes generation** | Local LLM (qwen2.5-class), map-reduce batching | AI credits metered | yes | yes, templates | yes |
| **Meeting bot** | None - files or browser recording only | Bot joins calls | Bot joins calls | Bot joins calls | Bot joins calls |
| **Free tier** | Unlimited everything | 800 min storage, credit-metered AI | 300 min/mo | 3 h/mo transcription | limited |
| **Paid (per user/mo)** | **$0** | $10-39 | $8.33-30 | $6-17 | ~$29 |
| **Cost, 10-person team/yr** | **$0** | ~$1,200+ | ~$1,000+ | ~$720+ | ~$3,480+ |
| **Data retention control** | Total (own SQLite + files) | Per plan tier | Per plan tier | Per plan tier | Per plan tier |
| **Works without internet** | **Yes** | No | No | No | No |
| **API / integrations** | REST API (self-owned, extensible) | Rich CRM sync | Moderate | Rich | Moderate |

---

## Where MOM-device genuinely wins

1. **Privacy & compliance by design.** Confidential discussions never touch a third-party cloud. Passes internal-data policies that ban Otter/Fireflies outright; nothing to leak because nothing egresses.
2. **Zero marginal cost at scale.** Unlimited minutes, unlimited users, no seats. Competitors bill per user per month forever.
3. **No meeting bot.** Nothing silently joins the call - audio is provided deliberately by a human.
4. **Multilingual edge.** Whisper-class models handle Indian-accent English and Hinglish natively; several incumbents are English-first.

## Where commercial tools honestly win

1. **Auto-join bots + calendar sync** for Zoom/Teams/Meet - zero-friction capture (a possible future integration for us).
2. **Conversation intelligence** - talk-time analytics, sales scorecards, org-wide search across thousands of meetings.
3. **Zero setup/maintenance** - sign up and go vs operating one Windows box (acceptable trade for the target use case).

## Fair-comparison caveats

- Cloud competitors do not publish WER/DER for their stacks; industry-wide cloud transcription accuracy is quoted at 90-95% on clean audio. Our chosen model stack lands in the same band by construction (92%+ accuracy class STT + SOTA-open diarization).
- Our speed figure is single-machine and GPU-dependent; CPU-only fallback is slower but functional.
- Single-server capacity: designed for an office LAN (6-7 speakers per meeting at this stage, Decision 10), not a global SaaS workload.

## Sources

- Fireflies.ai pricing: fireflies.ai pricing page mirrors (aitoolsatlas, costbench, sonix.ai resources), checked Aug 2026
- Otter.ai / MeetGeek / tl;dv: vendor pricing pages and 2026 comparisons (thebusinessdive, screenapp.io blog, thetoolsverse), checked Aug 2026
- Model benchmarks: Hugging Face Open ASR Leaderboard, pyannote model cards, AMI/VoxConverse DER tables (details in PROJECT_GUIDE section 4)
- Own measurements: REBUILD_LOG Phase 7

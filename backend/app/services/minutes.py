import re

import requests

from ..config import settings

REQUEST_TIMEOUT = 600


class LLMError(RuntimeError):
    pass


def split_text_into_batches(text: str, batch_size: int, overlap: int) -> list[str]:
    """Split transcript into batches preferring paragraph then sentence boundaries.

    Ported from legacy final.py; adds hard guarantee of forward progress.
    """
    if not text:
        return []
    batches: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + batch_size, text_length)
        if end < text_length:
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + batch_size // 2:
                end = para_break
            else:
                sentence_break = max(
                    text.rfind(". ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("? ", start, end),
                )
                if sentence_break > start + batch_size // 2:
                    end = sentence_break + 1
                elif end == start:
                    end = start + batch_size
        chunk = text[start:end].strip()
        if chunk:
            batches.append(chunk)
        if end >= text_length:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return batches


def generate(prompt: str) -> str:
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 2000},
    }
    try:
        response = requests.post(
            f"{settings.ollama_url}/api/generate", json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        raise LLMError(f"Ollama unreachable at {settings.ollama_url}: {exc}") from exc
    if response.status_code != 200:
        raise LLMError(f"Ollama returned {response.status_code}: {response.text[:300]}")
    return _strip_reasoning(response.json().get("response", "").strip())


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. deepseek-r1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _batch_prompt(batch_text: str, index: int, total: int) -> str:
    if total == 1:
        return (
            "You are a professional meeting secretary. Convert this meeting transcript "
            "into well-formatted meeting minutes.\n\nTRANSCRIPT:\n"
            f"{batch_text}\n\n"
            "Produce structured minutes in markdown with exactly these sections:\n"
            "# Meeting Minutes\n## Meeting Overview\n## Key Discussion Points\n"
            "## Decisions Made\n## Action Items\n## Next Steps\n"
            "Use bullet points. Include owners and dates for action items when mentioned. "
            "Be concise but complete."
        )
    return (
        f"You are processing part {index} of {total} of a long meeting transcript.\n\n"
        f"TRANSCRIPT SEGMENT:\n{batch_text}\n\n"
        "Extract concisely as bullets:\n1. Key discussion points and topics\n"
        "2. Any decisions mentioned\n3. Action items or tasks (with owners)\n"
        "4. Important context. Facts only."
    )


def _synthesis_prompt(summaries: list[str]) -> str:
    combined = "\n\n---\n\n".join(summaries)
    return (
        "Synthesize these section summaries from one meeting into final meeting minutes.\n\n"
        f"SECTION SUMMARIES:\n{combined}\n\n"
        "Create comprehensive minutes in markdown with exactly these sections:\n"
        "# Meeting Minutes\n## Meeting Overview\n## Key Discussion Points\n"
        "## Decisions Made\n## Action Items\n## Next Steps\n"
        "Consolidate duplicates, keep every decision and action item with its owner."
    )


def generate_minutes(transcript_text: str, progress_callback=None) -> tuple[str, int]:
    """Batched map-reduce minutes generation. Returns (markdown, batches_used)."""
    batches = split_text_into_batches(
        transcript_text, settings.batch_size_chars, settings.batch_overlap_chars
    )
    summaries: list[str] = []
    for i, batch in enumerate(batches, start=1):
        result = generate(_batch_prompt(batch, i, len(batches)))
        if result.startswith("[ERROR]"):
            raise LLMError(f"Batch {i} failed: {result}")
        summaries.append(result)
        if progress_callback:
            progress_callback(int(i / (len(batches) + 1) * 100))

    if len(summaries) == 1:
        if progress_callback:
            progress_callback(100)
        return summaries[0], 1

    final = generate(_synthesis_prompt(summaries))
    if final.startswith("[ERROR]"):
        raise LLMError(f"Synthesis failed: {final}")
    if progress_callback:
        progress_callback(100)
    return final, len(batches)

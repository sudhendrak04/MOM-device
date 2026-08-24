from app.services.minutes import split_text_into_batches, _strip_reasoning, _batch_prompt


class TestSplitBatches:
    def test_empty_text(self):
        assert split_text_into_batches("", 100, 10) == []

    def test_short_text_single_batch(self):
        text = "Hello world."
        assert split_text_into_batches(text, 100, 10) == [text]

    def test_splits_long_text_with_overlap(self):
        text = " ".join(f"sentence number {i}." for i in range(400))  # ~7600 chars
        batches = split_text_into_batches(text, 3000, 200)
        assert len(batches) >= 2
        # overlap: tail of batch N appears at the head of batch N+1
        tail = batches[0][-80:]
        assert any(tail.strip()[:40] in b for b in batches[1:])

    def test_progress_guaranteed_when_no_boundaries(self):
        # no spaces/periods -> must still advance and terminate
        text = "x" * 2500
        batches = split_text_into_batches(text, 1000, 900)  # overlap nearly == size
        assert all(len(b) <= 1000 for b in batches)
        assert sum(len(b.replace("x", "")) == 0 for b in batches) == len(batches)

    def test_prefers_paragraph_breaks(self):
        para = "A" * 1200
        text = f"{para}\n\n{para}\n\n{para}"
        batch = split_text_into_batches(text, 1500, 50)[0]
        assert batch.endswith(para)  # broke at the paragraph boundary


class TestPromptsAndCleanup:
    def test_think_blocks_stripped(self):
        raw = "<think>reasoning steps here</think>The answer is 4."
        assert _strip_reasoning(raw) == "The answer is 4."

    def test_single_batch_uses_full_minutes_prompt(self):
        prompt = _batch_prompt("transcript", 1, 1)
        assert "# Meeting Minutes" in prompt
        assert "Action Items" in prompt

    def test_multi_batch_uses_extraction_prompt(self):
        prompt = _batch_prompt("transcript", 2, 5)
        assert "part 2 of 5" in prompt

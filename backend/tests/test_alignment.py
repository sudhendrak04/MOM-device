from app.services.alignment import (
    LabeledTurn,
    assign_segments,
    build_plain_transcript,
    build_transcript,
    format_timestamp,
)
from app.services.diarization import Turn, merge_turns, speaker_labels
from app.services.transcription import Segment, Word


def _seg(start, end, text):
    return Segment(start=start, end=end, text=text,
                   words=[Word(start=start, end=end, text=text)])


def _turn(start, end, tag):
    return Turn(start=start, end=end, speaker_tag=tag)


class TestMergeTurns:
    def test_merges_same_speaker_within_gap(self):
        turns = [_turn(0.0, 2.0, "A"), _turn(2.2, 4.0, "A"), _turn(5.0, 6.0, "B")]
        merged = merge_turns(turns)
        assert len(merged) == 2
        assert merged[0].end == 4.0

    def test_does_not_merge_different_speakers(self):
        turns = [_turn(0.0, 2.0, "A"), _turn(2.1, 4.0, "B")]
        assert len(merge_turns(turns)) == 2

    def test_gap_beyond_threshold_not_merged(self):
        turns = [_turn(0.0, 2.0, "A"), _turn(3.0, 4.0, "A")]  # gap 1.0 > 0.3
        assert len(merge_turns(turns)) == 2


class TestSpeakerLabels:
    def test_labels_ordered_by_first_appearance(self):
        labels = speaker_labels([_turn(0, 1, "C"), _turn(1, 2, "A"), _turn(2, 3, "C")])
        assert labels == {"C": 1, "A": 2}


class TestAssignSegments:
    def test_exact_overlap_assignment(self):
        segments = [_seg(0.5, 1.5, "hello world")]
        turns = [_turn(0.0, 2.0, "A"), _turn(2.5, 4.0, "B")]
        labeled = assign_segments(segments, turns)
        assert len(labeled) == 1
        assert labeled[0].speaker_number == 1
        assert labeled[0].text == "hello world"

    def test_multiple_segments_concatenate_in_order(self):
        segments = [_seg(0.0, 1.0, "first part"), _seg(1.2, 2.0, "second part")]
        turns = [_turn(0.0, 2.5, "A")]
        labeled = assign_segments(segments, turns)
        assert labeled[0].text == "first part second part"

    def test_distant_segment_dropped(self):
        segments = [_seg(100.0, 101.0, "orphan")]
        turns = [_turn(0.0, 2.0, "A")]
        assert assign_segments(segments, turns) == []


class TestTranscriptBuilding:
    def test_timestamp_formatting(self):
        assert format_timestamp(65) == "01:05"
        assert format_timestamp(9.7) == "00:09"

    def test_labeled_lines(self):
        result = build_transcript([
            LabeledTurn(start=10, end=12, speaker_number=2, text="hi there")
        ])
        assert result.startswith("[00:10 - 00:12] Speaker 2: hi there")

    def test_plain_fallback(self):
        result = build_plain_transcript([_seg(30, 31, "no speakers here")])
        assert result == "[00:30] no speakers here"

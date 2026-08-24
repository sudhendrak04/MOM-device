"""Speaker naming (Option A): storage, listing with sample quotes, name substitution."""
import uuid


def _make_meeting(client, auth_headers, transcript=None, minutes=None):
    """Insert a meeting row directly (no pipeline) and return its id."""
    from app.db import new_session
    from app.models import Meeting

    meeting_id = uuid.uuid4().hex
    session = new_session()
    try:
        row = Meeting(
            id=meeting_id,
            original_filename='fake.wav',
            uploaded_path='nowhere',
            status='completed',
            stage='completed',
            progress=100,
        )
        if transcript is not None:
            row.transcript = transcript
        if minutes is not None:
            row.minutes_md = minutes
        session.add(row)
        session.commit()
    finally:
        session.close()
    return meeting_id


TRANSCRIPT = (
    "[00:00 - 00:05] Speaker 1: Good morning everyone, let us start.\n"
    "[00:05 - 00:09] Speaker 2: Sure. The budget variance is within two percent.\n"
    "[00:09 - 00:12] Speaker 1: Great, moving on to action items then."
)
MINUTES = "## Decisions\n- Speaker 1 approved the budget\n- Speaker 2 will circulate the report"


class TestSpeakersListing:
    def test_unknown_meeting_404(self, client, auth_headers):
        response = client.get('/api/meetings/nope/speakers', headers=auth_headers)
        assert response.status_code == 404

    def test_lists_detected_speakers_with_quotes(self, client, auth_headers):
        meeting_id = _make_meeting(client, auth_headers, transcript=TRANSCRIPT)
        response = client.get(f'/api/meetings/{meeting_id}/speakers', headers=auth_headers)
        assert response.status_code == 200
        speakers = response.json()['speakers']
        assert [s['speaker_number'] for s in speakers] == [1, 2]
        assert speakers[1]['sample_quote'].startswith('Sure. The budget')

    def test_empty_when_no_transcript(self, client, auth_headers):
        meeting_id = _make_meeting(client, auth_headers)
        response = client.get(f'/api/meetings/{meeting_id}/speakers', headers=auth_headers)
        assert response.status_code == 200
        assert response.json()['speakers'] == []


class TestRenameFlow:
    def test_rename_persists_and_substitutes_everywhere(self, client, auth_headers):
        meeting_id = _make_meeting(
            client, auth_headers, transcript=TRANSCRIPT, minutes=MINUTES
        )

        patch = client.patch(
            f'/api/meetings/{meeting_id}/speakers',
            json={'names': {'1': 'Laura', '2': 'Andrew'}},
            headers=auth_headers,
        )
        assert patch.status_code == 200
        speakers = {s['speaker_number']: s['name'] for s in patch.json()['speakers']}
        assert speakers == {1: 'Laura', 2: 'Andrew'}

        transcript = client.get(
            f'/api/meetings/{meeting_id}/transcript', headers=auth_headers
        ).text
        assert 'Speaker 1' not in transcript and 'Laura' in transcript
        assert '[00:05 - 00:09] Andrew:' in transcript

        minutes = client.get(
            f'/api/meetings/{meeting_id}/minutes', headers=auth_headers
        ).text
        assert 'Laura approved' in minutes and 'Andrew will circulate' in minutes

        detail = client.get(f'/api/meetings/{meeting_id}', headers=auth_headers).json()
        assert detail['speaker_names'] == {'1': 'Laura', '2': 'Andrew'}

    def test_raw_storage_untouched_by_renaming(self, client, auth_headers):
        """Renaming must never rewrite stored artifacts - it is a serve-time view."""
        from app.db import new_session
        from app.models import Meeting

        meeting_id = _make_meeting(client, auth_headers, transcript=TRANSCRIPT)
        client.patch(
            f'/api/meetings/{meeting_id}/speakers',
            json={'names': {'1': 'Laura'}},
            headers=auth_headers,
        )
        session = new_session()
        try:
            row = session.get(Meeting, meeting_id)
            assert 'Speaker 1:' in row.transcript
        finally:
            session.close()

    def test_clearing_name_falls_back_to_speaker_label(self, client, auth_headers):
        meeting_id = _make_meeting(
            client, auth_headers, transcript=TRANSCRIPT, minutes=MINUTES
        )
        url = f'/api/meetings/{meeting_id}/speakers'
        client.patch(url, json={'names': {'1': 'Laura'}}, headers=auth_headers)
        client.patch(url, json={'names': {'1': '', '2': 'Andrew'}}, headers=auth_headers)
        transcript = client.get(
            f'/api/meetings/{meeting_id}/transcript', headers=auth_headers
        ).text
        assert 'Speaker 1:' in transcript and 'Andrew:' in transcript

    def test_blank_and_invalid_entries_are_dropped(self, client, auth_headers):
        meeting_id = _make_meeting(client, auth_headers, transcript=TRANSCRIPT)
        patch = client.patch(
            f'/api/meetings/{meeting_id}/speakers',
            json={'names': {'1': '   ', 'x': 'Bob', '2': 'Greg'}},
            headers=auth_headers,
        )
        assert patch.status_code == 200
        # '   ' dropped as blank, key 'x' not a speaker number, only detected
        # speakers (1 and 2) appear in the listing even if extra keys were sent
        names = {s['speaker_number']: s['name'] for s in patch.json()['speakers']}
        assert names == {1: None, 2: 'Greg'}

    def test_name_over_80_chars_truncated(self, client, auth_headers):
        meeting_id = _make_meeting(client, auth_headers, transcript=TRANSCRIPT)
        patch = client.patch(
            f'/api/meetings/{meeting_id}/speakers',
            json={'names': {'1': 'N' * 200}},
            headers=auth_headers,
        )
        assert patch.status_code == 200
        saved = patch.json()['speakers'][0]['name']
        assert len(saved) <= 80

    def test_patch_unknown_meeting_404(self, client, auth_headers):
        response = client.patch(
            '/api/meetings/nope/speakers', json={'names': {'1': 'X'}},
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_auth_required_for_both_endpoints(self, client):
        assert client.get('/api/meetings/x/speakers').status_code == 401
        assert client.patch(
            '/api/meetings/x/speakers', json={'names': {}}
        ).status_code == 401

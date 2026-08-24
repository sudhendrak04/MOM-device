import io

from conftest import make_wav_bytes

KEY = "test-key-123"


def _upload(client, headers, filename='sample.wav'):
    wav_bytes = io.BytesIO(make_wav_bytes())
    return client.post(
        '/api/meetings/upload',
        files={'file': (filename, wav_bytes, 'audio/wav')},
        headers=headers,
    )


class TestAuth:
    def test_health_is_public(self, client):
        assert client.get('/api/health').status_code == 200

    def test_protected_endpoint_rejects_missing_key(self, client):
        assert client.get('/api/meetings').status_code == 401

    def test_protected_endpoint_rejects_wrong_key(self, client):
        response = client.get('/api/meetings', headers={'X-API-Key': 'nope'})
        assert response.status_code == 401

    def test_bearer_alternative_accepted(self, client, monkeypatch):
        from app.worker import worker as real_worker

        monkeypatch.setattr(real_worker, 'submit', lambda _id: None)
        response = client.post(
            '/api/meetings/upload',
            files={'file': ('t.wav', io.BytesIO(make_wav_bytes()), 'audio/wav')},
            headers={'Authorization': 'Bearer ' + KEY},
        )
        assert response.status_code == 200


class TestUploadFlow:
    def test_upload_returns_ids_and_queues(self, client, auth_headers, monkeypatch):
        from app import worker as worker_module

        submitted = []
        monkeypatch.setattr(
            worker_module.worker, 'submit', lambda mid: submitted.append(mid)
        )

        response = _upload(client, auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body['meeting_id'] == body['job_id']
        assert body['status'] == 'queued'
        assert submitted == [body['meeting_id']]

        listing = client.get('/api/meetings', headers=auth_headers).json()
        mine = [m for m in listing if m['id'] == body['meeting_id']]
        assert len(mine) == 1
        assert mine[0]['original_filename'] == 'sample.wav'

    def test_upload_rejects_bad_extension(self, client, auth_headers):
        bad_file = {'file': ('evil.exe', b'MZ', 'application/x-msdownload')}
        response = client.post(
            '/api/meetings/upload', files=bad_file, headers=auth_headers
        )
        assert response.status_code == 415

    def test_unknown_meeting_404(self, client, auth_headers):
        missing_jobs = client.get('/api/jobs/does-not-exist', headers=auth_headers)
        missing_meetings = client.get('/api/meetings/nope', headers=auth_headers)
        assert missing_jobs.status_code == 404
        assert missing_meetings.status_code == 404

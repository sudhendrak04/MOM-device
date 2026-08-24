from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import new_session
from ..models import Meeting
from ..schemas import JobStatus
from ..security import require_api_key

router = APIRouter(prefix="/jobs", dependencies=[Depends(require_api_key)])


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    session: Session = new_session()
    try:
        meeting = session.get(Meeting, job_id)
        if meeting is None:
            raise HTTPException(404, "Job not found")
        return JobStatus(
            job_id=meeting.id,
            status=meeting.status,
            stage=meeting.stage,
            stage_message=meeting.stage_message or "",
            progress=meeting.progress,
            error=meeting.error,
        )
    finally:
        session.close()

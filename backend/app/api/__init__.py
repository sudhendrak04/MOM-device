from fastapi import APIRouter

from .jobs import router as jobs_router
from .meetings import router as meetings_router
from .system import router as system_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(jobs_router)
api_router.include_router(meetings_router)

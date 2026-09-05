import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from ingestion.pipeline import run_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()
_status = {"status": "idle", "result": None}

class IngestRequest(BaseModel):
    data_dir: Optional[str] = None

class IngestResponse(BaseModel):
    status: str
    message: str

def _run_task(data_dir):
    global _status
    _status = {"status": "running", "result": None}
    try:
        result = run_ingestion(data_dir)
        _status = {"status": "complete", "result": result}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        _status = {"status": "error", "result": {"error": str(e)}}

@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(request: IngestRequest, background_tasks: BackgroundTasks):
    if _status["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already in progress")
    background_tasks.add_task(_run_task, request.data_dir)
    return IngestResponse(status="accepted", message="Ingestion started in background")

@router.get("/ingest/status")
async def get_ingestion_status():
    return _status

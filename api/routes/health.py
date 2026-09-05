import logging
from fastapi import APIRouter
from config.database import check_db_health
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "document-search-platform"}

@router.get("/health/dependencies")
async def dependency_health():
    db_healthy = check_db_health()
    return {
        "status": "ok" if db_healthy else "degraded",
        "dependencies": {
            "postgres": "healthy" if db_healthy else "unhealthy",
            "ollama": settings.ollama_base_url,
            "phoenix": f"http://{settings.phoenix_host}:{settings.phoenix_port}",
        },
    }

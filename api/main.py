import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import search, ingest, health
from tracing.phoenix_setup import instrument_all
from config.settings import settings

logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Document Search Platform API...")
    instrument_all()
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Document Search Platform API",
    description="Agentic RAG Document Search Platform — LlamaIndex + CrewAI + PGVector + Arize Phoenix",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(health.router, tags=["Health"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])

@app.get("/")
async def root():
    return {"service": "Document Search Platform", "version": "1.0.0", "docs": "/docs"}

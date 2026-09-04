"""
CrewAI Agent Server — runs in its own container.
Exposes OpenAI-compatible API that OpenWebUI connects to.
The Retriever agent calls the RAG API container for document search.
"""
import uuid
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any

from agents.crew_config import CrewAIRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Agent Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipeline: Optional[CrewAIRAGPipeline] = None

def get_pipeline() -> CrewAIRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CrewAIRAGPipeline()
    return _pipeline


class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: Optional[str] = "document-search"
    messages: List[ChatMessage]
    stream: Optional[bool] = False


@app.get("/api/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "document-search", "object": "model", "owned_by": "crewai-rag"}]
    }


@app.post("/api/v1/chat/completions")
async def chat(request: ChatRequest):
    last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
    if not last_user:
        return {"error": "No user message"}
    content = last_user.content
    query = " ".join([c.get("text","") for c in content if isinstance(c,dict)]) if isinstance(content, list) else str(content)
    if query.strip().startswith("###"):
        return {"id":"chatcmpl-skip","object":"chat.completion","model":"document-search","choices":[{"index":0,"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
    logger.info(f"CrewAI processing: {query[:80]}")
    result = get_pipeline().run(query)
    return {"id":f"chatcmpl-{uuid.uuid4().hex[:8]}","object":"chat.completion","model":"document-search","choices":[{"index":0,"message":{"content":result.get("answer","")},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}


@app.post("/api/v1/responses")
async def responses(request: dict):
    input_data = request.get("input", "")
    if isinstance(input_data, list):
        last_user = next((m for m in reversed(input_data) if m.get("role") == "user"), None)
        content = last_user.get("content","") if last_user else ""
        query = " ".join([c.get("text","") for c in content if isinstance(c,dict)]) if isinstance(content,list) else str(content)
    else:
        query = str(input_data)
    if query.strip().startswith("###") or not query.strip():
        return {"id":"resp-skip","object":"response","model":"document-search","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":""}]}],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}
    result = get_pipeline().run(query)
    return {"id":f"resp-{uuid.uuid4().hex[:8]}","object":"response","model":"document-search","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":result.get("answer","")}]}],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "crewai-agent"}

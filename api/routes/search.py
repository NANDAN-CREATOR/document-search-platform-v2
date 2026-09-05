import logging, uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any
from agents.rag_pipeline import AgenticRAGPipeline
import json

logger = logging.getLogger(__name__)
router = APIRouter()
_pipeline: Optional[AgenticRAGPipeline] = None

def get_pipeline() -> AgenticRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgenticRAGPipeline()
    return _pipeline

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SourceReference(BaseModel):
    filename: str
    score: float

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceReference]
    chunks_retrieved: int

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: Optional[str] = "document-search"
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str
    model: str
    choices: List[ChatChoice]

# --- Models endpoint ---
@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "document-search",
                "object": "model",
                "created": 1700000000,
                "owned_by": "document-search-platform",
            }
        ]
    }

# --- Search endpoint ---
@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = get_pipeline().run(request.query)
        return SearchResponse(
            query=result["query"],
            answer=result["answer"],
            sources=[SourceReference(**s) for s in result["sources"]],
            chunks_retrieved=result["chunks_retrieved"],
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _build_answer(query: str) -> tuple:
    """Run pipeline and return (full_answer, model_name)."""
    result = get_pipeline().run(query)
    sources_text = "\n".join(
        [f"- {s['filename']} (score: {s['score']:.3f})" for s in result["sources"]]
    )
    full_answer = (
        f"{result['answer']}\n\n**Sources:**\n{sources_text}"
        if result["sources"] else result["answer"]
    )
    return full_answer, result

# --- OpenWebUI chat/completions endpoint ---
@router.post("/chat/completions")
async def openwebui_chat(request: ChatRequest):
    last_user_msg = next(
        (m for m in reversed(request.messages) if m.role == "user"), None
    )
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Extract text content (handle both string and list formats)
    content = last_user_msg.content
    if isinstance(content, list):
        query = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
    else:
        query = str(content)

    try:
        full_answer, result = _build_answer(query)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": request.model or "document-search",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_answer},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- OpenWebUI /responses endpoint (newer API format) ---
@router.post("/responses")
async def openwebui_responses(request: dict):
    """Handle OpenWebUI responses API format."""
    try:
        # Extract query from input
        input_data = request.get("input", "")
        if isinstance(input_data, list):
            # Get ONLY the last user message — not entire chat history
            last_user = next(
                (m for m in reversed(input_data) if m.get("role") == "user"), None
            )
            if last_user:
                content = last_user.get("content", "")
                if isinstance(content, list):
                    query = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
                else:
                    query = str(content)
            else:
                query = ""
        else:
            query = str(input_data)

        # Block OpenWebUI internal system prompts
        if query.strip().startswith("###"):
            return {
                "id": f"resp-skip",
                "object": "response",
                "model": request.get("model", "document-search"),
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": ""}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            }

        if not query.strip():
            query = "Hello"

        full_answer, result = _build_answer(query)

        return {
            "id": f"resp-{uuid.uuid4().hex[:8]}",
            "object": "response",
            "model": request.get("model", "document-search"),
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": full_answer}
                    ]
                }
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
    except Exception as e:
        logger.error(f"Responses endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
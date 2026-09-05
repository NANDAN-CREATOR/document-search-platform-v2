# Document Search Platform v2 &mdash; Solution Architecture

## Overview

Fully containerised Agentic RAG platform using a two-service microservices architecture.

---

## System Architecture

```
+-------------------------------------------------------------------+
|                Docker Compose Stack                                  |
|                                                                      |
|  +OpenWebUI+                                                         |
|  |  :8080 |                                                          |
|  +--------+                                                          |
|      |                                                               |
|      v OpenAI API                                                    |
|  +CrewAI Agent+                                                      |
|  |    :8001    |                                                     |
|  | CrewAI 1.0  |---------------------------+                         |
|  | 3 Agents    |                           |                         |
|  +-------------+                           v HTTP POST               |
|      | HTTP POST                    +-----------------+              |
|      v /api/v1/search               | Ollama          |              |
|        +RAG API+                    |   :11434        |              |
|  |  :8000    |                      | llama3.2:3b     |              |
|  | LlamaIndex|--------------------> | nomic-embed-text|              |
|  | Docling   |                      +-----------------+              |
|  | RAGAs     |                                                       |
|  | Phoenix   |                       +-----------------+             |
|  +----------+                       | PostgreSQL       |             |
|      |                              |   :5432          |             |
|      v----------------------------> | PGVector         |             |
|                                     +-----------------+              |
|                                                                      |
|  +--------------------+                                              |
|  | Arize Phoenix      |                                              |
|  |   :6006 (UI)       |  <-- OTLP gRPC traces from RAG API           |
|  |   :4317 (gRPC)     |                                              |
|  +--------------------+                                              |
+-------------------------------------------------------------------+
```

---

## Two-Container Design Decision

### Problem
`LlamaIndex 0.11` requires `httpx<0.28` but `CrewAI 1.0` requires `httpx>=0.28.1`. These cannot coexist in one Python environment.

### Solution
Two separate Linux containers communicating over HTTP:

- **api` (8000)** -- LlamaIndex + Docling + Arize Phoenix + RAGAs
- **`crewai-agent` (8001)** -- CrewAI 1.0 only

---

## Ingestion Pipeline

```
PDF Files
    | Docling 2.5.0 (OCR + table extraction on Linux)
    v
Raw Text
    | LlamaIndex SentenceSplitter (512 tokens, 50 overlap)
    v
Chunks (TextNodes)
    | Ollama nomic-embed-text (768 dimensions)
    v
Vector Embeddings
    | LlamaIndex PGVectorStore
    v
PostgreSQL data_document_embeddings table
```

---

## Query Pipeline (CrewAI 3-Agent)

```
User Question (OpenWebUI)
    |
    v OpenAI API format
CrewAI Agent Server (8001)
    |
    v Python HTTP POST
 +---------------------------------------+
| RAG API (8000)                          |
|  -- Nomic embed query -> PGVector search|
|  -- Retrieve top-5 chunks               |
|  -- Llama3.2:3b reasoning               |
|  -- Validator agent                     |
+----------------------------------------+
    |
    v context + answer
Agent 1 (Retriever) -- summarises context
    |
Agent 2 (Reasoner) -- generates final answer
    |
Agent 3 (Validator) -- verifies groundedness
    |
    v
Answer -> OpenWebUI
```

---

## Technology Stack

| Layer | Technology | Version |
|------|-----------|--------|
| Document Processing | Docling | 2.5.0 |
| Vector Database | PostgreSQL + PGVector | pg16 |
| RAG Framework | LlamaIndex | 0.11.0 |
| Multi-Agent | CrewAI | 1.0.0 |
| LLM | Ollama llama3.2:3b | latest |
| Embeddings | Ollama nomic-embed-text | latest |
| Tracing | Arize Phoenix | 4.29.0 |
| Evaluation | RAGAs | 0.1.21 |
| Frontend | OpenWebUI | latest |
| API | FastAPI | 0.115.0 |
| Container Orchestration | Docker Compose | v2+ |

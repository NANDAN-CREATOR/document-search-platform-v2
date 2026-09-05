# Document Search Platform v2 -- Presentation

## Executive Summary

Agentic RAG platform that ingests PDF documents and enables conversational search via OpenWebUI. Built with all mandatory tools running natively on Linux in Docker.

---

## Problem Statement

Organisations have large PDF knowledge bases that are:
- Hard to search manually
- Not accessible conversationally
- Sensitive (cannot use external APIs)

Solution: fully local Agentic RAG with no external API calls.

---

## Architecture Highlights

### Two-Container Design

Key innovation: separating incompatible ecosystems:

- **RAG API container**: LlamaIndex 0.11 + Docling + Arize Phoenix + RAGAs
- **CrewAI container**: CrewAI 1.0 in isolated environment

Communication: HTTP REST between containers

### 3-Agent Pipeline

1. **RetrieverAgent** -- fetches relevant document chunks via semantic search
2. **ReasonerAgent** -- generates grounded answer from context
3. **ValidatorAgent** -- verifies accuracy and groundedness

---

## Mandatory Tools Implemented

| Tool | Implementation |
|-----|-----------------|
| Docling 2.5.0 | Full OCR + table extraction on Linux |
| PostgreSQL + PGVector | HNSW index, 768-dim vectors |
| LlamaIndex 0.11 | SentenceSplitter + PGVectorStore |
| CrewAI 1.0.0 | Sequential 3-agent pipeline |
| Ollama | llama3.2:3b + nomic-embed-text |
| Arize Phoenix 4.29 | Full UI, OTLP gRPC traces |
| RAGAs 0.1.21 | Official package with Ollama scorer |
| OpenWebUI | CrewAI agent via OpenAI compatible API |

---

## Evaluation Results (RAGAs)

| Metric | Score |
|------|------|
| Faithfulness | **0.80** |
| Answer Relevancy | 0.38 |
| Context Utilization | **0.69** |

Faithfulness of 0.8 is strong for a 3.2B local model.

---

## Key Challenges Solved

1 **httpx dependency conflict** -- Resolved by separate Docker containers
2 **CrewAI tool calling broken on small models** -- Resolved by direct HTTP call from RetrieverAgent
3 **Docling PyTorch CUDA download** -- Resolved by installing torch CPU-wheel first
4 **RAGAs langchain conflict** -- Resolved by separate api container with compatible versions

---

## Deployment

One command deployment:

```bash
docker-compose up --build
```

All 6 services start automatically.

---

## GitHub

https://github.com/NANDAN-CREATOR/document-search-platform-v2

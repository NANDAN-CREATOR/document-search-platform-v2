# Document Search Platform v2 &mdash; Full Docker Deployment

Agentic RAG platform with all mandatory tools running natively on Linux in Docker.

![Architecture](doc/architecture.md)

---

## Mandatory Tools

| Tool | Version | Status |
|----|--------|--------|
| Docling | 2.5.0 | ✅ Real OCR + table extraction on Linux |
| PostgreSQL + PGVector | pg16 | ✅ Docker container |
| LlamaIndex | 0.11 | ✅ RAG pipeline |
| CrewAI | 1.0.0 | ✅ 3-agent pipeline |
| Ollama | latest | ✅ llama3.2:3b + nomic-embed-text |
| Arize Phoenix | 4.29.0 | ✅ Full UI at localhost:6006 |
| RAGAs | 0.1.21 | ✅ Official package, faithfulness 0.80 |
| OpenWebUI | latest | ✅ Full chat interface |

---

## Architecture

Two-service microservices architecture to resolve `httpx`/`pydantic` dependency conflicts between LlamaIndex 0.11 and CrewAI 1.0:

```
 OpenWebUI (8080)
     ↓ CrewAI Agent (8001)              ↓ Ollama (11434)
          ↓ RAG API (8000)            ↓ PostgreSQL (5432)
               ↓ Phoenix (6006)
```

---

## Quick Start

### Prerequisites

- Docker Desktop or Docker Engine
- Docker Compose v2+
- 16GB RAM recommended
- 20GB+ free disk space

### 1. Clone the repository

```bash
git clone https://github.com/NANDAN-CREATOR/document-search-platform-v2
cd document-search-platform-v2
```

### 2. Configure environment

```bash
cp .env.example .env
```

### 3. Start all services

```bash
docker-compose up --build
```

This starts 6 containers: PostgreSQL, Ollama, Arize Phoenix, RAG API, CrewAI Agent, OpenWebUI.

### 4. Pull Ollama models

```bash
docker exec dsp_ollama ollama pull llama3.2:3b
docker exec dsp_ollama ollama pull nomic-embed-text
```

### 5. Add PDFs and ingest

```bash
# Copy PDFs to data/ folder
cp your_documents.pdf data/

# Trigger ingestion
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "./data"}'
```

### 6. Check ingestion status

```bash
curl http://localhost:8000/api/v1/ingest/status
```

---

## Service URLs

| Service | URL | Purpose |
|------|-----|--------|
| OpenWebUI | http://localhost:8080 | Chat interface |
| RAG API | http://localhost:8000/docs | Swagger UI |
| CrewAI Agent | http://localhost:8001 | Multi-agent API |
| Arize Phoenix | http://localhost:6006 | Tracing dashboard |
| Ollama | http://localhost:11434 | LMM API |
| PostgreSQL | localhost:5432 | Vector database |

---

## Run RAGAs Evaluation

```bash
docker exec dsp_api python -m evaluation.ragas_eval
```

Results saved to `evaluation/ragas_report.json`.

Current results (llama3.2:3b, 5 questions):

### RAGAs Results

| Metric | Score |
|------|------|
| Faithfulness | 0.80 |
| Answer Relevancy | 0.38 |
| Context Utilization | 0.69 |

---

## Project Structure

```
document-search-platform-v2/
|-- Dockerfile.api            → RAG API container (LlamaIndex ecosystem)
|-- Dockerfile.crewai        → CrewAI container (isolated ecosystem)
|-- docker-compose.yml      → 6-service orchestration
|-- requirements.api.txt    → LlamaIndex + Docling + Phoenix + RAGAs
|-- requirements.crewai.txt → CrewAI 1.0 (isolated)
|-- crewai_server.py
                             → FastAPI server for CrewAI container
|-- agents/
|   |-- crew_config.py      → CrewAI 3-agent pipeline
|   |-- rag_pipeline.py     → LlamaIndex RAG pipeline
|-- api/                    → FastAPI routes (search, ingest, health)
|-- config/                 → Settings + database
|-- ingestion/              → Docling + embedder + PGVector
|-- prompts/                → Externalized YAML prompts
|-- tracing/                → Arize Phoenix OTLP setup
|-- evaluation/             → RAGAs evaluation script + report
|-- doc/                    → Architecture, Swagger, presentation
```

---

## Documentation

See the [/doc directory](doc/) for:

- [Solution Architecture](doc/architecture.md)
- [API Swagger Spec](doc/api_swagger.yaml)
- [Presentation Deck](doc/presentation.md)
- [RAGAs Evaluation Report](doc/ragas_evaluation.md)

---

## Key Technical Decisions

### Why Two Containers?

`LlamaIndex 0.11` requires `httpx<<0.28` but `CrewAI 1.0` requires `httpx<>=0.28.1`. These are fundamentally incompatible in a single environment. Solution: separate Docker containers communicating over HTTP.

### Why no CrewAI Tool Calling?

`llama3.2:3b` generates malformed JSON for tool arguments. The Retriever agent calls the RAG API directly via HTTP POST instead -- more reliable and architecturally correct.

---

## GitHub

- v2 (Docker, all mandatory tools): https://github.com/NANDAN-CREATOR/document-search-platform-v2
- v1 (Windows ARM64, working): https://github.com/NANDAN-CREATOR/document-search-platform

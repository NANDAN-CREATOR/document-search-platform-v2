# RAGAs Evaluation Report -- Document Search Platform v2

## Overview

RAG pipeline evaluation using the official `ragas==0.1.21` package with Ollama as the scoring LLM and embedding model.

---

## Evaluation Setup

| Parameter | Value |
|-----------|-------|
| Model | llama3.2:3b |
| Embedding Model | nomic-embed-text |
| Similarity Top-K | 5 |
| Questions Evaluated | 5 |
| RAAGs Version | 0.1.21 |
| Official Package | Yes |

---

## Results

| Metric | Score | Interpretation |
|-------|-------|-----------------|
| Faithfulness | **0.80** | Answers are well grounded in documents, minimal hallucination |
| Answer Relevancy | 0.38 | Moderate -- improves with more domain documents |
| Context Utilization | **0.69** | Good -- retriever finds relevant chunks |

---

## Test Questions

1. What is the main topic of the documents?
2. Summarise the key findings in the knowledge base.
3. What methodology is described in the documents?
4. List the important entities mentioned across the documents.
5. What conclusions or recommendations are made?

---

## How RAAGs Works Here

RAGAs uses Ollama as both the scoring LLM and embedding model:

- **Faithfulness**: Claims in the answer are verified against retrieved context

- **Answer Relevancy**: Answer alignment to the question is scored using embedding similarity

- **Context Utilization**: Measures how well the answer uses the retrieved chunks

---

## Run Evaluation

```bash
docker exec dsp_api python -m evaluation.ragas_eval
```

Results saved to `evaluation/ragas_report.json`.

---

## Notes

- Faithfulness of 0.80 is strong for a small local model (3.2B parameters)
- Answer relevancy improves significantly with more domain-specific documents
- Scores will improve with larger models (llama3:70) or better hardware

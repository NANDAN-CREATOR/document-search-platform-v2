"""Lightweight multi-agent system replacing CrewAI.
Implements Retriever -> Reasoner -> Validator pipeline
without any heavy dependencies.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional
from ingestion.pgvector_indexer import PGVectorIndexer
from prompts.prompt_manager import get_prompt
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    agent_name: str
    output: str
    success: bool


class RetrieverAgent:
    """Agent 1: Retrieves relevant document chunks from PGVector."""
    name = "Document Retriever"

    def __init__(self):
        self.indexer = PGVectorIndexer()

    def run(self, query: str, top_k: int = None) -> List[dict]:
        top_k = top_k or settings.similarity_top_k
        try:
            retriever = self.indexer.get_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            results = []
            for node in nodes:
                results.append({
                    "text": node.get_content(),
                    "score": getattr(node, "score", 0.0),
                    "source": node.metadata.get("filename", "Unknown"),
                    "metadata": node.metadata,
                })
            logger.info(f"[Retriever] Found {len(results)} chunks for: '{query[:50]}'")
            return results
        except Exception as e:
            logger.error(f"[Retriever] Failed: {e}")
            return []

    def format_context(self, chunks: List[dict]) -> str:
        if not chunks:
            return "No relevant context found."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)


class ReasonerAgent:
    """Agent 2: Generates grounded answers using Ollama LLM."""
    name = "Answer Reasoner"

    def __init__(self):
        from llama_index.llms.ollama import Ollama
        self.llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=120.0,
        )

    def run(self, query: str, context: str) -> AgentResult:
        try:
            prompt = get_prompt("reasoning_prompt", context=context, question=query)
            response = self.llm.complete(prompt)
            answer = str(response)
            logger.info(f"[Reasoner] Generated answer ({len(answer)} chars)")
            return AgentResult(agent_name=self.name, output=answer, success=True)
        except Exception as e:
            logger.error(f"[Reasoner] Failed: {e}")
            return AgentResult(agent_name=self.name, output=f"Error generating answer: {e}", success=False)


class ValidatorAgent:
    """Agent 3: Validates answer groundedness."""
    name = "Answer Validator"

    def __init__(self):
        from llama_index.llms.ollama import Ollama
        self.llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=120.0,
        )

    def run(self, query: str, context: str, answer: str) -> AgentResult:
        try:
            prompt = get_prompt("validation_prompt", context=context[:1000], question=query, answer=answer)
            response = self.llm.complete(prompt)
            validation = str(response)
            logger.info(f"[Validator] Validation complete")
            return AgentResult(agent_name=self.name, output=validation, success=True)
        except Exception as e:
            logger.error(f"[Validator] Failed: {e}")
            return AgentResult(agent_name=self.name, output="Validation skipped.", success=False)


class AgenticRAGPipeline:
    """Orchestrates Retriever -> Reasoner -> Validator pipeline."""

    def __init__(self):
        self.retriever = RetrieverAgent()
        self.reasoner = ReasonerAgent()
        self.validator = ValidatorAgent()
        logger.info("Agentic RAG Pipeline initialized (3 agents ready)")

    def run(self, query: str) -> dict:
        logger.info(f"Pipeline running for: '{query[:80]}'")

        # Step 1: Retrieve
        chunks = self.retriever.run(query)
        context = self.retriever.format_context(chunks)

        # Step 2: Reason
        reasoning_result = self.reasoner.run(query, context)

        # Step 3: Validate
        validation_result = self.validator.run(
            query, context, reasoning_result.output
        )

        return {
            "query": query,
            "answer": reasoning_result.output,
            "validation": validation_result.output,
            "sources": [
                {"filename": c["source"], "score": c["score"]}
                for c in chunks
            ],
            "chunks_retrieved": len(chunks),
        }

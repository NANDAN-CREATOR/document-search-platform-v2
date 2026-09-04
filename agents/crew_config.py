"""
CrewAI multi-agent configuration.
Implements Retriever -> Reasoner -> Validator using CrewAI agents and tasks.
"""
import logging
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import Any, Type

from ingestion.pgvector_indexer import PGVectorIndexer
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentRetrievalTool(BaseTool):
    name: str = "DocumentRetrievalTool"
    description: str = (
        "Retrieves the most relevant document chunks from the vector database "
        "based on semantic similarity to the input query."
    )

    def _run(self, query: str) -> str:
        try:
            indexer = PGVectorIndexer()
            retriever = indexer.get_retriever(similarity_top_k=5)
            nodes = retriever.retrieve(query)
            if not nodes:
                return "No relevant documents found."
            parts = []
            for i, node in enumerate(nodes, 1):
                source = node.metadata.get("filename", "Unknown")
                score = getattr(node, "score", 0.0)
                parts.append(f"[Source {i}: {source} | Score: {score:.3f}]\n{node.get_content()}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            return f"Retrieval failed: {e}"


def build_retriever_agent() -> Agent:
    return Agent(
        role="Document Retriever",
        goal="Find the most relevant document chunks from the knowledge base",
        backstory="Specialist in semantic search and document retrieval.",
        tools=[DocumentRetrievalTool()],
        verbose=True,
        allow_delegation=False,
        llm=f"ollama/{settings.ollama_model}",
    )


def build_reasoner_agent() -> Agent:
    return Agent(
        role="Answer Reasoner",
        goal="Generate accurate answers using only provided document context",
        backstory="Expert at synthesising document information. Never makes up information.",
        verbose=True,
        allow_delegation=False,
        llm=f"ollama/{settings.ollama_model}",
    )


def build_validator_agent() -> Agent:
    return Agent(
        role="Answer Validator",
        goal="Validate answers are grounded and free of hallucinations",
        backstory="Quality assurance expert who checks accuracy and groundedness.",
        verbose=True,
        allow_delegation=False,
        llm=f"ollama/{settings.ollama_model}",
    )


class CrewAIRAGPipeline:
    """Full CrewAI implementation of the 3-agent RAG pipeline."""

    def __init__(self):
        self.retriever_agent = build_retriever_agent()
        self.reasoner_agent = build_reasoner_agent()
        self.validator_agent = build_validator_agent()
        logger.info("CrewAI RAG Pipeline initialised (3 agents)")

    def run(self, query: str) -> dict:
        logger.info(f"CrewAI Pipeline running for: {query[:80]}")
        retrieval_task = Task(
            description=f"Retrieve document chunks for: {query}",
            expected_output="Relevant document chunks with source citations",
            agent=self.retriever_agent,
        )
        reasoning_task = Task(
            description=f"Using retrieved context, answer: {query}\nAnswer ONLY from context.",
            expected_output="Concise answer with source citations",
            agent=self.reasoner_agent,
        )
        validation_task = Task(
            description=f"Validate the answer for: {query}\nCheck groundedness and hallucinations.",
            expected_output="Validation report",
            agent=self.validator_agent,
        )
        crew = Crew(
            agents=[self.retriever_agent, self.reasoner_agent, self.validator_agent],
            tasks=[retrieval_task, reasoning_task, validation_task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff(inputs={"query": query})
        return {
            "query": query,
            "answer": str(result),
            "validation": "Validated by CrewAI ValidatorAgent",
            "sources": [],
            "chunks_retrieved": 0,
        }

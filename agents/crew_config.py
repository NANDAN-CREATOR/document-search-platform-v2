"""
CrewAI multi-agent RAG pipeline.
"""
import os
import logging
import requests

RAG_API_URL = os.getenv("RAG_API_URL", "http://api:8000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Set litellm Ollama base URL before importing crewai
os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL

import litellm
litellm.ollama_api_base = OLLAMA_BASE_URL

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


def get_llm() -> LLM:
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        api_base=OLLAMA_BASE_URL,
    )


class DocumentRetrievalTool(BaseTool):
    name: str = "DocumentRetrievalTool"
    description: str = "Retrieves relevant document chunks from the knowledge base for a given query."

    def _run(self, query: str) -> str:
        try:
            resp = requests.post(
                f"{RAG_API_URL}/api/v1/search",
                json={"query": query, "top_k": 5},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            sources_text = "\n".join([f"- {s.get('filename', '?')} (score: {s.get('score', 0):.3f})" for s in sources])
            return f"{answer}\n\nSources:\n{sources_text}" if sources_text else answer
        except Exception as e:
            logger.error(f"DocumentRetrievalTool failed: {e}")
            return f"Retrieval failed: {e}"


def build_retriever_agent() -> Agent:
    return Agent(
        role="Document Retriever",
        goal="Find the most relevant document chunks",
        backstory="Specialist in semantic search.",
        tools=[DocumentRetrievalTool()],
        verbose=True,
        allow_delegation=False,
        llm=get_llm(),
    )


def build_reasoner_agent() -> Agent:
    return Agent(
        role="Answer Reasoner",
        goal="Generate accurate answers from document context",
        backstory="Expert at synthesising documents. Never makes up information.",
        verbose=True,
        allow_delegation=False,
        llm=get_llm(),
    )


def build_validator_agent() -> Agent:
    return Agent(
        role="Answer Validator",
        goal="Validate answers are grounded and free of hallucinations",
        backstory="Quality assurance expert who checks accuracy.",
        verbose=True,
        allow_delegation=False,
        llm=get_llm(),
    )


class CrewAIRAGPipeline:
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
            description=f"Validate answer for: {query}\nCheck groundedness and hallucinations.",
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
        return {"query": query, "answer": str(result), "validation": "Validated by CrewAI", "sources": []}

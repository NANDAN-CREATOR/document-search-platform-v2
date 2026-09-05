"""
CrewAI multi-agent RAG pipeline.
Runs in its own container -- NO LlamaIndex imports.

Architecture:
  OpenWebUI -> CrewAI Server -> RetrieverAgent (HTTP call to RAG API)
                             -> ReasonerAgent (Ollama LLM)
                             -> ValidatorAgent (Ollama LLM)
"""
import os
import logging
import requests

RAG_API_URL = os.getenv("RAG_API_URL", "http://api:8000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

from crewai import Agent, Task, Crew, Process, LLM

logger = logging.getLogger(__name__)

# Module-level LLM instance with increased timeout for slow validator
OLLAMA_LLM = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    timeout=300,
)


def _retrieve_from_api(query: str) -> str:
    """
    Direct Python HTTP call to the RAG API -- no LLM tool calling needed.
    This avoids the broken JSON tool call generation from llama3.2:3b.
    """
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
        sources_text = "\n".join(
            [f"- {s.get('filename', '?')} (score: {s.get('score', 0):.3f})" for s in sources]
        )
        context = f"{answer}\n\nSources:\n{sources_text}" if sources_text else answer
        logger.info(f"Retrieved context for: {query[:50]}")
        return context
    except Exception as e:
        logger.error(f"RAG API call failed: {e}")
        return f"Could not retrieve context: {e}"


class CrewAIRAGPipeline:
    """
    3-agent RAG pipeline without tool calling.
    Retriever calls FastAPI directly in Python.
    Reasoner and Validator use Ollama LLM via CrewAI.
    """

    def __init__(self):
        # Agent 1: Retriever -- no tools, gets context via direct HTTP
        self.retriever_agent = Agent(
            role="Document Retriever",
            goal="Summarise the retrieved document context for the reasoner",
            backstory="Specialist in semantic search. You receive pre-retrieved context and summarise it.",
            verbose=True,
            allow_delegation=False,
            llm=OLLAMA_LLM,
        )
        # Agent 2: Reasoner
        self.reasoner_agent = Agent(
            role="Answer Reasoner",
            goal="Generate accurate answers from document context",
            backstory="Expert at synthesising documents. Never makes up information.",
            verbose=True,
            allow_delegation=False,
            llm=OLLAMA_LLM,
        )
        # Agent 3: Validator
        self.validator_agent = Agent(
            role="Answer Validator",
            goal="Validate answers are grounded and free of hallucinations",
            backstory="Quality assurance expert who checks accuracy.",
            verbose=True,
            allow_delegation=False,
            llm=OLLAMA_LLM,
        )
        logger.info("CrewAI RAG Pipeline initialised (3 agents, no tool calling)")

    def run(self, query: str) -> dict:
        logger.info(f"CrewAI Pipeline running for: {query[:80]}")

        # Step 1: Direct Python call to RAG API -- no LLM tool call
        context = _retrieve_from_api(query)
        logger.info(f"Context retrieved: {len(context)} chars")

        # Step 2: Retriever agent summarises the context
        retrieval_task = Task(
            description=f"""You have received the following document context for the question: {query}

CONTEXT:
{context}

Summarise the key information relevant to answering the question.""",
            expected_output="A clear summary of the relevant context",
            agent=self.retriever_agent,
        )

        # Step 3: Reasoner agent generates the answer
        reasoning_task = Task(
            description=f"Using the summarised context, answer: {query}\nAnswer ONLY from the provided context.",
            expected_output="Concise answer with source citations",
            agent=self.reasoner_agent,
        )

        # Step 4: Validator agent checks the answer
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
        result = crew.kickoff()
        return {"query": query, "answer": str(result), "validation": "Validated by CrewAI", "sources": []}

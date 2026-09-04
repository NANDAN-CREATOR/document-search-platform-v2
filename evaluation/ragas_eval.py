"""
RAGAs evaluation using the official ragas package.
Runs on Linux/Docker where all dependencies install cleanly.

Run:
    docker-compose exec api python -m evaluation.ragas_eval
"""
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from config.settings import settings
from agents.rag_pipeline import AgenticRAGPipeline
from ingestion.pgvector_indexer import PGUctorIndexer

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS: List[str] = [
    "What is the main topic of the documents?",
    "Summarise the key findings in the knowledge base.",
    "What methodology is described in the documents?",
    "List the important entities mentioned across the documents.",
    "What conclusions or recommendations are made?",
]
DEFAULT_GROUND_TRUTHS: Optional[List[List[str]]] = None


class RAGEvaluator:
    def __init__(self):
        self.pipeline = AgenticRAGPipeline()
        self.indexer = PGVectorIndexer()
        self.ragas_llm = LangchainLLMWrapper(Ollama(model=settings.ollama_model, base_url=settings.ollama_base_url))
        self.ragas_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url))

    def evaluate_pipeline(self, questions, ground_truths=None, output_path=None):
        data = {"question": [], "answer": [], "contexts": [], "ground_truths": ground_truths or [["N/A"]] * len(questions)}
        for idx, q in enumerate(questions, 1):
            logger.info(f"[{idx}/{len(questions)}] {q}")
            try:
                result = self.pipeline.run(q)
                nodes = self.indexer.get_retriever(similarity_top_k=settings.similarity_top_k).retrieve(q)
                data["question"].append(q)
                data["answer"].append(result.get("answer", ""))
                data["contexts"].append([n.get_content() for n in nodes] or [""])
            except Exception as e:
                logger.error(f"Failed: {e}")
                data["question"].append(q)
                data["answer"].append("")
                data["contexts"].append([""])
        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths: metrics.append(context_recall)
        for m in metrics:
            if hasattr(m, "llm"): m.llm = self.ragas_llm
            if hasattr(m, "embeddings"): m.embeddings = self.ragas_embeddings
        results = evaluate(dataset=dataset, metrics=metrics)
        output = {"faithfulness": float(results["faithfulness"]), "answer_relevancy": float(results["answer_relevancy"]), "context_precision": float(results["context_precision"]), "num_questions": len(questions), "settings": {"model": settings.ollama_model, "embedding_model": settings.ollama_embedding_model, "similarity_top_k": settings.similarity_top_k, "data_dir": settings.data_dir}}
        if ground_truths: output["context_recall"] = float(results["context_recall"])
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(output, indent=2))
        return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    RAGEvaluator().evaluate_pipeline(DEFAULT_QUESTIONS, output_path="evaluation/ragas_report.json")

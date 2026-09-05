"""
RAGAS evaluation for the Agentic RAG pipeline.

RAGAS version:
    0.1.21

Run inside Docker:

    docker exec dsp_api python -m evaluation.ragas_eval

or:

    docker compose exec api python -m evaluation.ragas_eval

Evaluation modes:

1. WITHOUT ground truth:
       - faithfulness
       - answer_relevancy
       - context_utilization

2. WITH ground truth:
       - faithfulness
       - answer_relevancy
       - context_precision
       - context_recall

Ollama is used as the RAGAS evaluator LLM and embedding model.
RAGAS concurrency is intentionally limited to one worker because
Ollama is running locally.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,
    context_precision,
    context_recall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from config.settings import settings
from agents.rag_pipeline import AgenticRAGPipeline
from ingestion.pgvector_indexer import PGVectorIndexer


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# DEFAULT QUESTIONS
# ============================================================

DEFAULT_QUESTIONS: List[str] = [
    "What is the main topic of the documents?",
    "Summarise the key findings in the knowledge base.",
    "What methodology is described in the documents?",
    "List the important entities mentioned across the documents.",
    "What conclusions or recommendations are made?",
]


# ============================================================
# RAG EVALUATOR
# ============================================================

class RAGEvaluator:

    def __init__(self):

        logger.info("Initializing RAGAS evaluator...")

        # ----------------------------------------------------
        # Actual Agentic RAG pipeline
        # ----------------------------------------------------

        self.pipeline = AgenticRAGPipeline()

        # ----------------------------------------------------
        # PGVector indexer
        # ----------------------------------------------------

        self.indexer = PGVectorIndexer()

        # ----------------------------------------------------
        # RAGAS LLM
        #
        # This is the LLM used by RAGAS to evaluate the
        # generated answers.
        # ----------------------------------------------------

        logger.info(
            "Configuring RAGAS LLM: %s",
            settings.ollama_model,
        )

        self.ragas_llm_client = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

        self.ragas_llm = LangchainLLMWrapper(
            self.ragas_llm_client
        )

        # ----------------------------------------------------
        # RAGAS embedding model
        # ----------------------------------------------------

        logger.info(
            "Configuring RAGAS embedding model: %s",
            settings.ollama_embedding_model,
        )

        self.ragas_embeddings_client = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

        self.ragas_embeddings = LangchainEmbeddingsWrapper(
            self.ragas_embeddings_client
        )

        logger.info(
            "RAGAS evaluator initialized successfully."
        )

    # ========================================================
    # BUILD DATASET
    # ========================================================

    def build_dataset(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> Dataset:

        questions_data: List[str] = []
        answers_data: List[str] = []
        contexts_data: List[List[str]] = []

        # ----------------------------------------------------
        # Validate ground truths
        # ----------------------------------------------------

        if ground_truths is not None:

            if len(ground_truths) != len(questions):

                raise ValueError(
                    "Number of ground truths must match "
                    "number of questions."
                )

        # ----------------------------------------------------
        # Process questions
        # ----------------------------------------------------

        for idx, question in enumerate(questions, start=1):

            logger.info(
                "[%d/%d] %s",
                idx,
                len(questions),
                question,
            )

            try:

                # =================================================
                # RUN ACTUAL RAG PIPELINE
                # =================================================

                result = self.pipeline.run(question)

                answer = result.get("answer", "")

                if not answer:

                    logger.warning(
                        "RAG pipeline returned an empty answer "
                        "for question: %s",
                        question,
                    )

                # =================================================
                # RETRIEVE CONTEXTS FOR RAGAS
                # =================================================

                retriever = self.indexer.get_retriever(
                    similarity_top_k=settings.similarity_top_k
                )

                nodes = retriever.retrieve(question)

                contexts = []

                for node in nodes:

                    try:

                        content = node.get_content()

                    except Exception:

                        content = ""

                    if content:

                        contexts.append(content)

                # RAGAS requires at least a context value.

                if not contexts:

                    contexts = [""]

                # =================================================
                # STORE RESULT
                # =================================================

                questions_data.append(question)

                answers_data.append(answer)

                contexts_data.append(contexts)

                logger.info(
                    "Retrieved %d contexts; answer length=%d",
                    len(contexts),
                    len(answer),
                )

            except Exception as exc:

                logger.exception(
                    "Failed processing question '%s': %s",
                    question,
                    exc,
                )

                # Keep dataset aligned.

                questions_data.append(question)

                answers_data.append("")

                contexts_data.append([""])

        # ========================================================
        # CREATE DATASET
        # ========================================================

        data = {
            "question": questions_data,
            "answer": answers_data,
            "contexts": contexts_data,
        }

        # --------------------------------------------------------
        # Add real ground truth only when provided.
        #
        # RAGAS 0.1.21 expects:
        #
        #     ground_truth
        #
        # NOT:
        #
        #     ground_truths
        # --------------------------------------------------------

        if ground_truths is not None:

            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        logger.info(
            "Dataset created successfully with %d questions.",
            len(dataset),
        )

        logger.info(
            "Dataset columns: %s",
            dataset.column_names,
        )

        return dataset

    # ========================================================
    # CONFIGURE METRICS
    # ========================================================

    def configure_metrics(
        self,
        ground_truths: Optional[List[str]] = None,
    ):

        # ----------------------------------------------------
        # WITHOUT GROUND TRUTH
        #
        # context_precision/context_recall cannot be used
        # correctly without actual reference answers.
        #
        # Therefore use context_utilization.
        # ----------------------------------------------------

        if ground_truths is None:

            metrics = [
                faithfulness,
                answer_relevancy,
                context_utilization,
            ]

            logger.info(
                "No ground truths supplied."
            )

            logger.info(
                "Using metrics: "
                "faithfulness, answer_relevancy, "
                "context_utilization"
            )

        # ----------------------------------------------------
        # WITH GROUND TRUTH
        # ----------------------------------------------------

        else:

            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]

            logger.info(
                "Ground truths supplied."
            )

            logger.info(
                "Using metrics: "
                "faithfulness, answer_relevancy, "
                "context_precision, context_recall"
            )

        # ----------------------------------------------------
        # Attach Ollama LLM / embeddings
        # ----------------------------------------------------

        for metric in metrics:

            if hasattr(metric, "llm"):

                metric.llm = self.ragas_llm

            if hasattr(metric, "embeddings"):

                metric.embeddings = self.ragas_embeddings

        return metrics

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    def evaluate_pipeline(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ):

        # ====================================================
        # BUILD DATASET
        # ====================================================

        dataset = self.build_dataset(
            questions=questions,
            ground_truths=ground_truths,
        )

        # ====================================================
        # CONFIGURE METRICS
        # ====================================================

        metrics = self.configure_metrics(
            ground_truths=ground_truths,
        )

        # ====================================================
        # RAGAS RUN CONFIGURATION
        #
        # RAGAS 0.1.21 defaults:
        #
        # timeout = 180
        # max_retries = 10
        # max_wait = 60
        # max_workers = 16
        #
        # max_workers=16 is too aggressive for a local
        # Ollama server.
        #
        # We deliberately use ONE worker.
        # ====================================================

        run_config = RunConfig(
            timeout=300,
            max_retries=2,
            max_wait=30,
            max_workers=1,
        )

        logger.info(
            "=============================================="
        )

        logger.info(
            "Starting RAGAS evaluation"
        )

        logger.info(
            "=============================================="
        )

        logger.info(
            "Questions: %d",
            len(questions),
        )

        logger.info(
            "Metrics: %s",
            [
                getattr(metric, "name", str(metric))
                for metric in metrics
            ],
        )

        logger.info(
            "Ollama model: %s",
            settings.ollama_model,
        )

        logger.info(
            "Ollama embedding model: %s",
            settings.ollama_embedding_model,
        )

        logger.info(
            "RAGAS timeout: 300 seconds"
        )

        logger.info(
            "RAGAS max workers: 1"
        )

        logger.info(
            "RAGAS max retries: 2"
        )

        # ====================================================
        # EXECUTE RAGAS
        # ====================================================

        try:

            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                run_config=run_config,
            )

        except Exception as exc:

            logger.exception(
                "RAGAS evaluation failed: %s",
                exc,
            )

            raise

        # ====================================================
        # BUILD OUTPUT
        # ====================================================

        output = {
            "num_questions": len(questions),

            "metrics": {},

            "settings": {
                "ragas_version": "0.1.21",
                "model": settings.ollama_model,
                "embedding_model": settings.ollama_embedding_model,
                "similarity_top_k": settings.similarity_top_k,
                "data_dir": settings.data_dir,
                "ragas_max_workers": 1,
                "ragas_timeout_seconds": 300,
                "ragas_max_retries": 2,
            },
        }

        # ====================================================
        # EXTRACT METRICS
        # ====================================================

        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_utilization",
            "context_precision",
            "context_recall",
        ]

        for metric_name in metric_names:

            if metric_name in results:

                try:

                    value = float(
                        results[metric_name]
                    )

                except (TypeError, ValueError):

                    value = None

                output["metrics"][metric_name] = value

        # ====================================================
        # SAVE REPORT
        # ====================================================

        if output_path:

            output_file = Path(output_path)

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_file.write_text(
                json.dumps(
                    output,
                    indent=2,
                ),
                encoding="utf-8",
            )

            logger.info(
                "RAGAS report saved to: %s",
                output_file,
            )

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        print()
        print("=" * 60)
        print("RAGAS EVALUATION RESULT")
        print("=" * 60)

        print(
            json.dumps(
                output,
                indent=2,
            )
        )

        print("=" * 60)

        return output


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Agentic RAG pipeline using RAGAS."
        )
    )

    parser.add_argument(
        "--output",
        default="evaluation/ragas_report.json",
        help=(
            "Path for the generated RAGAS JSON report."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s: "
            "%(message)s"
        ),
    )

    # --------------------------------------------------------
    # Run evaluator
    # --------------------------------------------------------

    evaluator = RAGEvaluator()

    evaluator.evaluate_pipeline(
        questions=DEFAULT_QUESTIONS,
        ground_truths=None,
        output_path=args.output,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
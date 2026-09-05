"""Main ingestion pipeline orchestrator."""
import logging
from ingestion.docling_processor import DoclingProcessor
from ingestion.embedder import DocumentEmbedder
from ingestion.pgvector_indexer import PGVectorIndexer
from config.settings import settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self):
        self.processor = DoclingProcessor()
        self.embedder = DocumentEmbedder()
        self.indexer = PGVectorIndexer()

    def run(self, data_dir: str = None) -> dict:
        data_dir = data_dir or settings.data_dir
        logger.info(f"Starting ingestion pipeline for: {data_dir}")
        raw_docs = self.processor.process_directory(data_dir)
        if not raw_docs:
            return {"status": "error", "message": "No documents found"}
        nodes = self.embedder.process(raw_docs)
        self.indexer.index_nodes(nodes)
        result = {
            "status": "success",
            "documents_processed": len(raw_docs),
            "chunks_indexed": len(nodes),
            "vector_store": "PostgreSQL + PGVector",
            "data_dir": str(data_dir),
        }
        logger.info(f"Ingestion complete: {result}")
        return result


def run_ingestion(data_dir: str = None) -> dict:
    from tracing.phoenix_setup import instrument_llamaindex
    instrument_llamaindex()
    pipeline = IngestionPipeline()
    return pipeline.run(data_dir)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    data_directory = sys.argv[1] if len(sys.argv) > 1 else settings.data_dir
    result = run_ingestion(data_directory)
    print(result)

"""Index nodes into PostgreSQL with PGVector using LlamaIndex."""
import logging
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings as LlamaSettings

from config.settings import settings
from config.database import init_pgvector

logger = logging.getLogger(__name__)


class PGVectorIndexer:
    def __init__(self):
        LlamaSettings.llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=120.0,
        )
        LlamaSettings.embed_model = OllamaEmbedding(
            model_name=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        init_pgvector()
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize PGVector store with compatibility handling."""
        try:
            from llama_index.vector_stores.postgres import PGVectorStore
            self.vector_store = PGVectorStore.from_params(
                host=settings.postgres_host,
                port=str(settings.postgres_port),
                database=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
                table_name=settings.vector_table_name,
                embed_dim=settings.embedding_dimension,
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            logger.info("PGVector store initialized successfully")
        except Exception as e:
            logger.error(f"PGVector init failed: {e}")
            raise

    def index_nodes(self, nodes: List[TextNode]) -> VectorStoreIndex:
        logger.info(f"Indexing {len(nodes)} nodes into PGVector...")
        index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=True,
        )
        logger.info("Indexing complete.")
        return index

    def load_index(self) -> VectorStoreIndex:
        return VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
        )

    def get_retriever(self, similarity_top_k: Optional[int] = None):
        top_k = similarity_top_k or settings.similarity_top_k
        index = self.load_index()
        return index.as_retriever(similarity_top_k=top_k)

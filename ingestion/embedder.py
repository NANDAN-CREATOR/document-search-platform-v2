"""Document chunking and embedding using LlamaIndex + Ollama."""
import logging
from typing import List, Dict, Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import TextNode

from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Chunk documents and generate embeddings."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embed_model = OllamaEmbedding(
            model_name=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    def chunk_documents(
        self, raw_docs: List[Dict[str, Any]]
    ) -> List[Document]:
        """Convert raw documents into LlamaIndex Documents."""
        llama_docs = []
        for doc in raw_docs:
            llama_doc = Document(
                text=doc["text"],
                metadata=doc["metadata"],
                id_=f"doc_{doc['filename']}",
            )
            llama_docs.append(llama_doc)
        logger.info(f"Created {len(llama_docs)} LlamaIndex documents")
        return llama_docs

    def split_into_nodes(self, documents: List[Document]) -> List[TextNode]:
        """Split documents into chunks (nodes)."""
        nodes = self.splitter.get_nodes_from_documents(documents)
        logger.info(f"Split into {len(nodes)} chunks")
        return nodes

    def embed_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        """Generate embeddings for each node."""
        texts = [node.get_content() for node in nodes]
        embeddings = self.embed_model.get_text_embedding_batch(
            texts, show_progress=True
        )
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        logger.info(f"Generated embeddings for {len(nodes)} nodes")
        return nodes

    def process(self, raw_docs: List[Dict[str, Any]]) -> List[TextNode]:
        """Full pipeline: chunk + embed."""
        documents = self.chunk_documents(raw_docs)
        nodes = self.split_into_nodes(documents)
        nodes = self.embed_nodes(nodes)
        return nodes

"""Application settings — Docker-aware configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"

    # PostgreSQL + PGVector
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "document_search"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Arize Phoenix
    phoenix_host: str = "phoenix"
    phoenix_port: int = 6006
    phoenix_grpc_port: int = 4317

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # Prompts
    prompts_dir: str = "./prompts"

    # Data
    data_dir: str = "./data"

    # Vector Store
    vector_table_name: str = "document_embeddings"
    embedding_dimension: int = 768
    similarity_top_k: int = 5

    @property
    def postgres_connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def phoenix_endpoint(self) -> str:
        return f"http://{self.phoenix_host}:{self.phoenix_grpc_port}"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

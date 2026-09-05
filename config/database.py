"""Database connection and PGVector setup."""
import logging
from sqlalchemy import create_engine, text
from config.settings import settings

logger = logging.getLogger(__name__)


def get_engine():
    return create_engine(
        settings.postgres_connection_string,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def init_pgvector(engine=None):
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {settings.vector_table_name} (
                id SERIAL PRIMARY KEY,
                node_id VARCHAR(255) UNIQUE NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                embedding vector({settings.embedding_dimension}),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
        logger.info(f"PGVector table '{settings.vector_table_name}' ready.")
    return engine


def check_db_health() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

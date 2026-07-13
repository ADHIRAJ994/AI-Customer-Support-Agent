"""
Centralized app configuration. Values are read from environment variables
or a local .env file (see .env.example).
"""
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM provider ---
    llm_provider: Literal["openai", "groq"] = "groq"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    embedding_model: str = "text-embedding-3-small"

    # --- Storage paths ---
    vectorstore_dir: str = "data/vectorstore"
    upload_dir: str = "data/uploads"
    feedback_db: str = "data/feedback.db"

    # --- Chunking / retrieval ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 4

    # --- Misc ---
    api_base_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
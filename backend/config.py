"""
Application configuration loaded from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./deal_intelligence.db"
    )

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Vector Store
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")

    # Model paths
    ML_MODEL_DIR: str = os.getenv("ML_MODEL_DIR", "./models/saved")

    # Llama (local model)
    LLAMA_MODEL_PATH: str = os.getenv("LLAMA_MODEL_PATH", "")


settings = Settings()

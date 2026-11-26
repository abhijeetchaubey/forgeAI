import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Load environment variables from .env file
load_dotenv(override=False)

class Settings(BaseModel):
    """
    ForgeAI Configuration model with strict validation.
    Loads values from environment variables with production-ready defaults.
    """
    LLM_PROVIDER: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq")
    )
    GEMINI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    GEMINI_MODEL: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    )
    GROQ_API_KEY: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    GROQ_MODEL: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    GROQ_TEMPERATURE: float = Field(
        default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0.0")),
        ge=0.0,
        le=2.0
    )
    CHUNK_SIZE: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")),
        gt=0
    )
    CHUNK_OVERLAP: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")),
        ge=0
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    )
    CHROMA_PERSIST_DIRECTORY: Path = Field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"))
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "forgeai_documents")
    )
    TOP_K: int = Field(
        default_factory=lambda: int(os.getenv("TOP_K", "4")),
        gt=0
    )
    LOG_LEVEL: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    @model_validator(mode="after")
    def validate_chunk_constraints(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be strictly less than CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        return self

    def validate_groq_api_key(self) -> str:
        """
        Validates that GROQ_API_KEY is configured and not a placeholder before calling Groq LLM.
        """
        key = self.GROQ_API_KEY.strip()
        if not key or "placeholder" in key.lower() or key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is missing or invalid. Please set a valid Groq API key in your .env file or environment variables."
            )
        return key

# Global singleton settings instance
settings = Settings()

def get_settings() -> Settings:
    return settings

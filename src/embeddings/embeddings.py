from typing import Optional
from config import settings
from src.utils.logger import logger

# Prefer langchain_huggingface, fallback to langchain_community if needed
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

_cached_embedding_instance: Optional[HuggingFaceEmbeddings] = None

def get_embedding_model(
    model_name: Optional[str] = None,
    device: str = "cpu",
    normalize_embeddings: bool = True
) -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings instance for local all-MiniLM-L6-v2 vector embeddings.
    No OpenAI or external API calls are made for embeddings.
    """
    global _cached_embedding_instance
    selected_model = model_name or settings.EMBEDDING_MODEL_NAME

    # Return cached instance if requested with default settings
    if _cached_embedding_instance is not None and model_name is None:
        return _cached_embedding_instance

    logger.info("Initializing HuggingFaceEmbeddings with model: %s on device: %s", selected_model, device)
    try:
        # Prefer local cache to avoid external network timeouts
        model_kwargs = {"device": device}
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=selected_model,
                model_kwargs={**model_kwargs, "local_files_only": True},
                encode_kwargs={"normalize_embeddings": normalize_embeddings},
            )
        except Exception:
            embeddings = HuggingFaceEmbeddings(
                model_name=selected_model,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": normalize_embeddings},
            )
        if model_name is None:
            _cached_embedding_instance = embeddings
        return embeddings
    except Exception as e:
        logger.error("Failed to load HuggingFaceEmbeddings (%s): %s", selected_model, str(e), exc_info=True)
        raise

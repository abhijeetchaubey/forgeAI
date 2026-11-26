from typing import Optional, Any
from langchain_groq import ChatGroq
from config import settings
from src.utils.logger import logger

def get_groq_llm(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Any:
    """
    Initializes and returns the configured LLM client (Gemini when LLM_PROVIDER='gemini',
    otherwise ChatGroq).
    """
    selected_temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE
    provider = settings.LLM_PROVIDER.strip().lower()

    if provider == "gemini" and api_key is None:
        gemini_key = settings.GEMINI_API_KEY.strip()
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        selected_model = model or settings.GEMINI_MODEL
        logger.info("Initializing ChatGoogleGenerativeAI with model=%s, temperature=%.2f", selected_model, selected_temp)
        return ChatGoogleGenerativeAI(
            google_api_key=gemini_key,
            model=selected_model,
            temperature=selected_temp,
        )

    selected_key = api_key if api_key is not None else settings.GROQ_API_KEY
    if not selected_key or "placeholder" in selected_key.lower() or selected_key == "your_groq_api_key_here":
        logger.error("Attempted to initialize Groq LLM without a valid GROQ_API_KEY.")
        raise ValueError(
            "GROQ_API_KEY is not configured or is a placeholder. Please set a valid Groq API key."
        )

    selected_model = model or settings.GROQ_MODEL

    logger.info("Initializing ChatGroq with model=%s, temperature=%.2f", selected_model, selected_temp)

    try:
        llm = ChatGroq(
            groq_api_key=selected_key,
            model_name=selected_model,
            temperature=selected_temp,
        )
        return llm
    except Exception as e:
        logger.error("Failed to initialize ChatGroq: %s", str(e), exc_info=True)
        raise

def test_groq_connection(llm: Optional[ChatGroq] = None) -> bool:
    """
    Sends a test ping to Groq to verify connectivity and credentials.
    """
    try:
        client = llm or get_groq_llm()
        response = client.invoke("ping")
        logger.info("Groq connection test successful: %s", response.content[:50])
        return True
    except Exception as e:
        logger.warning("Groq connection test failed: %s", str(e))
        return False

import pytest
from pydantic import ValidationError
from config import Settings, settings

def test_settings_loaded():
    assert settings.CHUNK_SIZE > 0
    assert settings.CHUNK_OVERLAP >= 0
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE
    assert settings.TOP_K > 0
    assert "MiniLM" in settings.EMBEDDING_MODEL_NAME

def test_chunk_constraint_validation():
    # overlap >= size must raise ValueError
    with pytest.raises(ValidationError):
        Settings(CHUNK_SIZE=500, CHUNK_OVERLAP=500)

    with pytest.raises(ValidationError):
        Settings(CHUNK_SIZE=500, CHUNK_OVERLAP=600)

def test_groq_api_key_validation():
    # Empty or placeholder keys should fail validation when invoked
    cfg_empty = Settings(GROQ_API_KEY="")
    with pytest.raises(ValueError, match="GROQ_API_KEY is missing or invalid"):
        cfg_empty.validate_groq_api_key()

    cfg_placeholder = Settings(GROQ_API_KEY="gsk_placeholder_for_phase1_testing")
    with pytest.raises(ValueError, match="GROQ_API_KEY is missing or invalid"):
        cfg_placeholder.validate_groq_api_key()

    cfg_valid = Settings(GROQ_API_KEY="gsk_real_test_key_1234567890abcdef")
    assert cfg_valid.validate_groq_api_key() == "gsk_real_test_key_1234567890abcdef"

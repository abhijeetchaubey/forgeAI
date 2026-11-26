import pytest
from unittest.mock import patch, MagicMock
from src.llm.groq_client import get_groq_llm

def test_groq_missing_api_key_raises():
    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        get_groq_llm(api_key="")

def test_groq_placeholder_api_key_raises():
    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        get_groq_llm(api_key="gsk_placeholder_for_phase1_testing")

@patch("src.llm.groq_client.ChatGroq")
def test_groq_initialization_with_valid_key(mock_chat_groq):
    mock_instance = MagicMock()
    mock_chat_groq.return_value = mock_instance

    with patch("src.llm.groq_client.settings.GROQ_MODEL", "openai/gpt-oss-20b"):
        llm = get_groq_llm(
            api_key="gsk_test_valid_key_0123456789",
            temperature=0.2
        )
    assert llm == mock_instance
    mock_chat_groq.assert_called_once_with(
        groq_api_key="gsk_test_valid_key_0123456789",
        model_name="openai/gpt-oss-20b",
        temperature=0.2
    )

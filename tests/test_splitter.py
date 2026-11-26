import pytest
from langchain_core.documents import Document
from src.text_splitter.splitter import get_text_splitter, split_documents

def test_get_text_splitter_constraints():
    # Overlap must be strictly less than chunk_size
    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        get_text_splitter(chunk_size=100, chunk_overlap=100)

def test_split_documents_preserves_metadata():
    long_content = "ForgeAI Industrial Manufacturing Intelligence System. " * 30
    doc = Document(
        page_content=long_content,
        metadata={"filename": "manual.pdf", "page_number": 3, "file_type": "pdf"}
    )
    
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["filename"] == "manual.pdf"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["file_type"] == "pdf"
        assert chunk.metadata["chunk_index"] == i
        assert len(chunk.page_content) <= 120  # Roughly within chunk_size bounds

def test_split_empty_documents():
    chunks = split_documents([])
    assert chunks == []

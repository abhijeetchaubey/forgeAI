import pytest
from src.embeddings.embeddings import get_embedding_model
from src.vectorstore.chroma_store import (
    get_vector_store,
    add_documents,
    semantic_search,
    semantic_search_with_scores,
    get_retriever,
)

def test_embedding_model_generation():
    embeddings = get_embedding_model()
    text = "Industrial automation robot calibration"
    vector = embeddings.embed_query(text)
    assert isinstance(vector, list)
    assert len(vector) == 384  # all-MiniLM-L6-v2 dimension is 384

def test_chroma_storage_and_persistence(temp_chroma_dir, sample_documents):
    # Initialize store in temp directory
    store = get_vector_store(persist_directory=temp_chroma_dir, collection_name="test_collection")
    
    # Add documents
    doc_ids = add_documents(sample_documents, vector_store=store)
    assert len(doc_ids) == len(sample_documents)

    # Verify persistence: SQLite database created on disk
    sqlite_file = temp_chroma_dir / "chroma.sqlite3"
    assert sqlite_file.exists(), f"Expected {sqlite_file} to exist after persisting documents"

def test_semantic_search_top_k(temp_chroma_dir, sample_documents):
    store = get_vector_store(persist_directory=temp_chroma_dir, collection_name="test_collection")
    add_documents(sample_documents, vector_store=store)

    # Search for thermal safety query
    results = semantic_search(
        query="What happens when the machine overheats or reaches 85 degrees?",
        top_k=2,
        vector_store=store
    )
    assert len(results) == 2
    top_doc = results[0]
    assert "85 degrees" in top_doc.page_content or "Emergency cutoff" in top_doc.page_content
    assert top_doc.metadata["filename"] in ["safety_rules.pdf", "valves.csv", "pump_telemetry.txt"]
    assert "page_number" in top_doc.metadata

def test_semantic_search_with_scores(temp_chroma_dir, sample_documents):
    store = get_vector_store(persist_directory=temp_chroma_dir, collection_name="test_collection")
    add_documents(sample_documents, vector_store=store)

    scored_results = semantic_search_with_scores(
        query="coolant valve maintenance",
        top_k=1,
        vector_store=store
    )
    assert len(scored_results) == 1
    doc, score = scored_results[0]
    assert "Coolant valve" in doc.page_content
    assert isinstance(score, float)

def test_get_retriever_interface(temp_chroma_dir, sample_documents):
    store = get_vector_store(persist_directory=temp_chroma_dir, collection_name="test_collection")
    add_documents(sample_documents, vector_store=store)

    retriever = get_retriever(top_k=1, vector_store=store)
    retrieved = retriever.invoke("Hydraulic pump")
    assert len(retrieved) == 1
    assert "Hydraulic pump" in retrieved[0].page_content

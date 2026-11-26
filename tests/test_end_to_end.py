from pathlib import Path
from src.document_loaders.loaders import load_document
from src.text_splitter.splitter import split_documents
from src.vectorstore.chroma_store import (
    get_vector_store,
    add_documents,
    semantic_search,
    semantic_search_with_scores,
)
from tests.test_loaders import create_sample_pdf_if_missing

def test_phase1_end_to_end_pipeline(sample_data_dir, temp_chroma_dir):
    """
    Complete Phase 1 integration verification:
    1. Load a sample PDF.
    2. Load a sample CSV.
    3. Verify chunks and metadata (including filename and page_number).
    4. Store chunks in ChromaDB and verify vector database persistence.
    5. Perform semantic search and inspect retrieved results with configurable top-k.
    """
    pdf_path = sample_data_dir / "sample.pdf"
    csv_path = sample_data_dir / "sample.csv"
    create_sample_pdf_if_missing(pdf_path)

    # 1. Load documents
    pdf_docs = load_document(pdf_path)
    csv_docs = load_document(csv_path)

    assert len(pdf_docs) > 0, "PDF loading failed"
    assert len(csv_docs) > 0, "CSV loading failed"

    # Verify initial metadata
    for doc in pdf_docs:
        assert doc.metadata["filename"] == "sample.pdf"
        assert "page_number" in doc.metadata
        assert doc.metadata["file_type"] == "pdf"

    for doc in csv_docs:
        assert doc.metadata["filename"] == "sample.csv"
        assert "page_number" in doc.metadata
        assert doc.metadata["file_type"] == "csv"

    # 2. Text Chunking
    all_docs = pdf_docs + csv_docs
    chunks = split_documents(all_docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= len(all_docs)

    # Verify chunk metadata preservation
    for chunk in chunks:
        assert "filename" in chunk.metadata
        assert "page_number" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["filename"] in ["sample.pdf", "sample.csv"]

    # 3. Store in ChromaDB and verify persistence
    vector_store = get_vector_store(
        persist_directory=temp_chroma_dir,
        collection_name="phase1_e2e_collection"
    )
    doc_ids = add_documents(chunks, vector_store=vector_store)
    assert len(doc_ids) == len(chunks)

    # Persistence verification
    db_file = temp_chroma_dir / "chroma.sqlite3"
    assert db_file.exists(), f"ChromaDB persistence file not found at {db_file}"

    # 4. Semantic search retrieval with top-k=2
    query = "What is the status and maintenance schedule of the Coolant Valve?"
    results = semantic_search(query=query, top_k=2, vector_store=vector_store)
    assert len(results) == 2

    # Top result should match Coolant Valve from CSV
    top_result = results[0]
    assert "Coolant Valve" in top_result.page_content or "valve" in top_result.page_content.lower()
    assert top_result.metadata["filename"] == "sample.csv"
    assert top_result.metadata["page_number"] == 1

    # 5. Scored search retrieval
    scored_results = semantic_search_with_scores(
        query="emergency cutoff temperature",
        top_k=2,
        vector_store=vector_store
    )
    assert len(scored_results) == 2
    for doc, score in scored_results:
        assert isinstance(score, float)
        assert len(doc.page_content) > 0

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings
from src.utils.logger import logger

def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Instantiates a RecursiveCharacterTextSplitter with configurable chunk size and overlap.
    """
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    if overlap >= size:
        raise ValueError(f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({size})")

    logger.debug("Creating text splitter with chunk_size=%d, chunk_overlap=%d", size, overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Splits a list of documents into chunks while preserving and enriching metadata.
    Adds chunk_index to track position within each source document.
    """
    if not documents:
        logger.warning("No documents provided to split_documents.")
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Enrich chunks with index tracking per source
    doc_chunk_counters = {}
    for chunk in chunks:
        source_key = chunk.metadata.get("source", "unknown")
        idx = doc_chunk_counters.get(source_key, 0)
        chunk.metadata["chunk_index"] = idx
        doc_chunk_counters[source_key] = idx + 1

    logger.info(
        "Split %d input document(s) into %d chunk(s) (chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size or settings.CHUNK_SIZE,
        chunk_overlap or settings.CHUNK_OVERLAP,
    )
    return chunks

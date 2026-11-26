from pathlib import Path
from typing import List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from config import settings
from src.embeddings.embeddings import get_embedding_model
from src.utils.logger import logger

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

def get_vector_store(
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: Optional[str] = None,
    embedding_model = None
) -> Chroma:
    """
    Initializes or reconnects to a persistent Chroma vector store.
    """
    directory = Path(persist_directory or settings.CHROMA_PERSIST_DIRECTORY)
    collection = collection_name or settings.CHROMA_COLLECTION_NAME
    embeddings = embedding_model or get_embedding_model()

    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Accessing ChromaDB collection '%s' at '%s'",
            collection,
            directory.resolve(),
        )
        vector_store = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=str(directory.resolve()),
        )
        return vector_store
    except Exception as e:
        logger.error("Failed to initialize Chroma vector store: %s", str(e), exc_info=True)
        raise

def add_documents(
    documents: List[Document],
    vector_store: Optional[Chroma] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: Optional[str] = None,
) -> List[str]:
    """
    Ingests document chunks and their metadata into ChromaDB.
    Returns the list of generated document IDs.
    """
    if not documents:
        logger.warning("No documents provided to add_documents.")
        return []

    store = vector_store or get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    try:
        logger.info("Adding %d document chunks to ChromaDB...", len(documents))
        doc_ids = store.add_documents(documents)
        logger.info("Successfully added %d chunks to ChromaDB.", len(doc_ids))
        return doc_ids
    except Exception as e:
        logger.error("Failed to add documents to ChromaDB: %s", str(e), exc_info=True)
        raise

def semantic_search(
    query: str,
    top_k: Optional[int] = None,
    vector_store: Optional[Chroma] = None,
    filter_dict: Optional[dict] = None
) -> List[Document]:
    """
    Executes semantic similarity search on the ChromaDB vector store.
    """
    k = top_k if top_k is not None else settings.TOP_K
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    store = vector_store or get_vector_store()
    logger.info("Executing semantic search (top_k=%d) for query: '%s'", k, query)

    try:
        results = store.similarity_search(query=query, k=k, filter=filter_dict)
        logger.info("Retrieved %d matching documents.", len(results))
        return results
    except Exception as e:
        logger.error("Semantic search failed: %s", str(e), exc_info=True)
        raise

def semantic_search_with_scores(
    query: str,
    top_k: Optional[int] = None,
    vector_store: Optional[Chroma] = None,
    filter_dict: Optional[dict] = None
) -> List[Tuple[Document, float]]:
    """
    Executes similarity search returning documents with their similarity/distance scores.
    """
    k = top_k if top_k is not None else settings.TOP_K
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    store = vector_store or get_vector_store()
    logger.info("Executing scored search (top_k=%d) for query: '%s'", k, query)

    try:
        results = store.similarity_search_with_score(query=query, k=k, filter=filter_dict)
        logger.info("Retrieved %d scored matching documents.", len(results))
        return results
    except Exception as e:
        logger.error("Scored semantic search failed: %s", str(e), exc_info=True)
        raise

def get_retriever(
    top_k: Optional[int] = None,
    vector_store: Optional[Chroma] = None,
    filter_dict: Optional[dict] = None
) -> BaseRetriever:
    """
    Returns a standard LangChain BaseRetriever configured for semantic search.
    """
    k = top_k if top_k is not None else settings.TOP_K
    store = vector_store or get_vector_store()
    search_kwargs = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    return store.as_retriever(search_kwargs=search_kwargs)

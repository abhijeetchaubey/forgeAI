from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from src.utils.logger import logger

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}

def _enrich_metadata(
    docs: List[Document],
    file_path: Path,
    file_type: str,
    default_page: int = 1
) -> List[Document]:
    """
    Standardizes and ensures metadata fields: filename, source, page_number, file_type.
    """
    filename = file_path.name
    normalized_path = str(file_path.resolve())

    for i, doc in enumerate(docs):
        doc.metadata = dict(doc.metadata)  # Ensure standard mutable dict
        doc.metadata["filename"] = filename
        doc.metadata["source"] = normalized_path
        doc.metadata["file_type"] = file_type

        # Preserve and normalize page number (1-based index)
        if "page" in doc.metadata:
            # PyMuPDFLoader uses 0-based 'page'
            doc.metadata["page_number"] = int(doc.metadata["page"]) + 1
        elif "page_number" not in doc.metadata:
            doc.metadata["page_number"] = default_page

    return docs

def load_pdf(file_path: Union[str, Path]) -> List[Document]:
    """
    Loads a PDF document using PyMuPDF, preserving page-level content and metadata.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    logger.info("Loading PDF document: %s", path.name)
    try:
        loader = PyMuPDFLoader(str(path))
        docs = loader.load()
        enriched = _enrich_metadata(docs, path, file_type="pdf")
        logger.info("Loaded %d pages from %s", len(enriched), path.name)
        return enriched
    except Exception as e:
        logger.error("Error loading PDF %s: %s", path, str(e), exc_info=True)
        raise

def load_txt(file_path: Union[str, Path], encoding: str = "utf-8") -> List[Document]:
    """
    Loads a TXT document, ensuring metadata preservation.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"TXT file not found: {path}")

    logger.info("Loading TXT document: %s", path.name)
    try:
        try:
            loader = TextLoader(str(path), encoding=encoding)
            docs = loader.load()
        except UnicodeDecodeError:
            logger.warning("Failed with %s encoding for %s. Retrying with 'latin-1'", encoding, path.name)
            loader = TextLoader(str(path), encoding="latin-1")
            docs = loader.load()

        enriched = _enrich_metadata(docs, path, file_type="txt", default_page=1)
        logger.info("Loaded text document from %s (%d chars)", path.name, len(enriched[0].page_content) if enriched else 0)
        return enriched
    except Exception as e:
        logger.error("Error loading TXT %s: %s", path, str(e), exc_info=True)
        raise

def load_csv(file_path: Union[str, Path], encoding: str = "utf-8") -> List[Document]:
    """
    Loads a CSV document, converting rows into documents with metadata.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info("Loading CSV document: %s", path.name)
    try:
        loader = CSVLoader(file_path=str(path), encoding=encoding)
        docs = loader.load()
        enriched = _enrich_metadata(docs, path, file_type="csv", default_page=1)
        
        # Also assign row index if available
        for doc in enriched:
            if "row" in doc.metadata:
                doc.metadata["row_number"] = int(doc.metadata["row"]) + 1

        logger.info("Loaded %d rows from CSV: %s", len(enriched), path.name)
        return enriched
    except Exception as e:
        logger.error("Error loading CSV %s: %s", path, str(e), exc_info=True)
        raise

def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Dispatches file loading based on extension (.pdf, .txt, .csv) with strict metadata preservation.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".txt":
        return load_txt(path)
    elif suffix == ".csv":
        return load_csv(path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

def load_directory(dir_path: Union[str, Path], glob_pattern: str = "*") -> List[Document]:
    """
    Scans a directory and loads all supported documents.
    """
    folder = Path(dir_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder}")

    all_docs: List[Document] = []
    matched_files = [f for f in folder.glob(glob_pattern) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    logger.info("Found %d supported files in directory: %s", len(matched_files), folder)
    for f in matched_files:
        try:
            docs = load_document(f)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning("Skipping failed document %s: %s", f.name, str(e))

    return all_docs

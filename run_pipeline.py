#!/usr/bin/env python
"""
ForgeAI Phase 1: Project Foundation and Document Intelligence Demonstration Script.
Executes document ingestion, chunking, embedding generation, ChromaDB persistence,
and semantic retrieval across PDF, CSV, and TXT documents.
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.document_loaders.loaders import load_directory, load_document
from src.text_splitter.splitter import split_documents
from src.vectorstore.chroma_store import (
    get_vector_store,
    add_documents,
    semantic_search_with_scores,
)
from src.utils.logger import logger
from tests.test_loaders import create_sample_pdf_if_missing

console = Console()

def run_demo():
    console.print(Panel.fit(
        "[bold cyan]ForgeAI Phase 1: Document Intelligence Pipeline[/bold cyan]\n"
        "[dim]Local HuggingFace all-MiniLM-L6-v2 Embeddings & ChromaDB Vector Store[/dim]",
        border_style="cyan"
    ))

    # 1. Prepare sample files
    sample_dir = PROJECT_ROOT / "data" / "sample_docs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = sample_dir / "sample.pdf"
    create_sample_pdf_if_missing(pdf_path)

    # 2. Document Loading
    console.print("\n[bold yellow]Step 1: Loading Documents (PDF, TXT, CSV)...[/bold yellow]")
    raw_docs = load_directory(sample_dir)
    console.print(f"[green]✔ Successfully loaded {len(raw_docs)} document items from {sample_dir}[/green]")

    # Display sample metadata
    meta_table = Table(title="Loaded Documents Metadata Sample")
    meta_table.add_column("Filename", style="cyan")
    meta_table.add_column("Type", style="magenta")
    meta_table.add_column("Page", justify="center")
    meta_table.add_column("Content Preview", style="dim", max_width=60)

    for doc in raw_docs[:5]:
        meta_table.add_row(
            doc.metadata.get("filename", "N/A"),
            doc.metadata.get("file_type", "N/A"),
            str(doc.metadata.get("page_number", "1")),
            doc.page_content.replace("\n", " ")[:60] + "..."
        )
    console.print(meta_table)

    # 3. Document Chunking
    console.print(f"\n[bold yellow]Step 2: Splitting text (chunk_size={settings.CHUNK_SIZE}, chunk_overlap={settings.CHUNK_OVERLAP})...[/bold yellow]")
    chunks = split_documents(raw_docs)
    console.print(f"[green]✔ Created {len(chunks)} text chunks with preserved metadata.[/green]")

    # 4. Ingest and Persist in ChromaDB
    console.print(f"\n[bold yellow]Step 3: Storing chunks in ChromaDB ({settings.CHROMA_PERSIST_DIRECTORY})...[/bold yellow]")
    vector_store = get_vector_store()
    doc_ids = add_documents(chunks, vector_store=vector_store)
    console.print(f"[green]✔ Ingested and persisted {len(doc_ids)} chunks into collection '{settings.CHROMA_COLLECTION_NAME}'[/green]")

    # Check persistence file on disk
    chroma_file = Path(settings.CHROMA_PERSIST_DIRECTORY) / "chroma.sqlite3"
    if chroma_file.exists():
        console.print(f"[green]✔ Verified persistence: {chroma_file} exists on disk ({chroma_file.stat().st_size} bytes)[/green]")

    # 5. Semantic Search Queries
    test_queries = [
        "What is the status and maintenance cycle of the Coolant Valve?",
        "What happens during an emergency temperature spike exceeding 85 degrees?",
        "Tell me about the hydraulic pump pressure and temperature."
    ]

    console.print(f"\n[bold yellow]Step 4: Executing Semantic Search (Top-k={settings.TOP_K})...[/bold yellow]")
    for query in test_queries:
        console.print(f"\n[bold white on blue] Query: {query} [/bold white on blue]")
        results = semantic_search_with_scores(query=query, top_k=2, vector_store=vector_store)

        res_table = Table(title=f"Results for: '{query[:40]}...'")
        res_table.add_column("Rank", justify="center", style="bold")
        res_table.add_column("Score (Distance)", justify="center", style="yellow")
        res_table.add_column("Source / File", style="cyan")
        res_table.add_column("Page", justify="center", style="magenta")
        res_table.add_column("Matched Chunk Content", style="white", max_width=70)

        for rank, (doc, score) in enumerate(results, 1):
            res_table.add_row(
                str(rank),
                f"{score:.4f}",
                doc.metadata.get("filename", "unknown"),
                str(doc.metadata.get("page_number", "-")),
                doc.page_content.replace("\n", " ").strip()[:100] + "..."
            )
        console.print(res_table)

    console.print("\n[bold green]✔ Phase 1 Pipeline Completed Successfully![/bold green]\n")

if __name__ == "__main__":
    run_demo()

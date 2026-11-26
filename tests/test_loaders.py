from pathlib import Path
import pytest
from src.document_loaders.loaders import (
    load_document,
    load_directory,
    load_txt,
    load_csv,
    load_pdf,
    SUPPORTED_EXTENSIONS,
)

def create_sample_pdf_if_missing(pdf_path: Path):
    if pdf_path.exists():
        return
    try:
        import fitz
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "ForgeAI Robotic Workcell Specification - Page 1\nOperating pressure: 120 bar.")
        p2 = doc.new_page()
        p2.insert_text((50, 72), "ForgeAI Robotic Workcell Specification - Page 2\nEmergency cutoff temperature: 85 Celsius.")
        doc.save(str(pdf_path))
        doc.close()
    except ImportError:
        # Fallback minimal valid 2-page PDF
        pdf_data = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 2/Kids[3 0 R 4 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"4 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n"
            b"0000000114 00000 n \n0000000185 00000 n \n"
            b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n256\n%%EOF\n"
        )
        pdf_path.write_bytes(pdf_data)

def test_load_txt(sample_data_dir):
    txt_file = sample_data_dir / "sample.txt"
    docs = load_txt(txt_file)
    assert len(docs) == 1
    doc = docs[0]
    assert "ForgeAI Factory" in doc.page_content
    assert doc.metadata["filename"] == "sample.txt"
    assert doc.metadata["page_number"] == 1
    assert doc.metadata["file_type"] == "txt"
    assert "sample.txt" in doc.metadata["source"]

def test_load_csv(sample_data_dir):
    csv_file = sample_data_dir / "sample.csv"
    docs = load_csv(csv_file)
    assert len(docs) == 4  # 4 rows of equipment telemetry
    for i, doc in enumerate(docs):
        assert doc.metadata["filename"] == "sample.csv"
        assert doc.metadata["page_number"] == 1
        assert doc.metadata["file_type"] == "csv"
        assert doc.metadata["row_number"] == i + 1

    # Check content of first row
    assert "Hydraulic Pump A" in docs[0].page_content

def test_load_pdf(sample_data_dir):
    pdf_file = sample_data_dir / "sample.pdf"
    create_sample_pdf_if_missing(pdf_file)

    docs = load_pdf(pdf_file)
    assert len(docs) >= 1
    assert docs[0].metadata["filename"] == "sample.pdf"
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["file_type"] == "pdf"
    if len(docs) > 1:
        assert docs[1].metadata["page_number"] == 2

def test_load_document_dispatcher(sample_data_dir):
    txt_file = sample_data_dir / "sample.txt"
    docs = load_document(txt_file)
    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == "txt"

def test_load_document_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(bad_file)

def test_load_document_missing_file():
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_file.pdf")

def test_load_directory(sample_data_dir):
    docs = load_directory(sample_data_dir)
    assert len(docs) >= 5  # txt + csv rows + pdf
    extensions = {doc.metadata["file_type"] for doc in docs}
    assert "txt" in extensions
    assert "csv" in extensions

import os
import shutil
import tempfile
from pathlib import Path
import pytest
from langchain_core.documents import Document

@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "sample_docs"

@pytest.fixture
def temp_chroma_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Hydraulic pump operates at 68.5 degrees Celsius under standard load.",
            metadata={"filename": "pump_telemetry.txt", "page_number": 1, "source": "/data/pump_telemetry.txt"}
        ),
        Document(
            page_content="Emergency cutoff initiates when temperature reaches or exceeds 85 degrees Celsius.",
            metadata={"filename": "safety_rules.pdf", "page_number": 2, "source": "/data/safety_rules.pdf"}
        ),
        Document(
            page_content="Coolant valve flow rate reduced by 15 percent, scheduled for maintenance.",
            metadata={"filename": "valves.csv", "page_number": 1, "row_number": 2, "source": "/data/valves.csv"}
        )
    ]

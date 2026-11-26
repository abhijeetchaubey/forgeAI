"""
Unit and Integration Tests for ForgeAI Phase 8: Dynamic Tool Registry
Tests:
1. All 4 independent tools (csv_lookup, calculator, current_date, open_meteo_weather).
2. Schema validation & structured error handling (no unhandled crashes).
3. Security: calculator strictly blocks arbitrary Python execution.
4. Registry metadata exposure for generator.
5. Selective tool assignment:
   - HR agent does NOT receive weather or calculator tools.
   - Procurement agent receives csv_lookup and calculator tools.
6. Runtime attaches ONLY selected tools.
"""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from registry import ToolRegistry, get_default_tool_registry
from generator import (
    AgentDefinition,
    RetrievalConfig,
    RefusalPolicy,
    OutputContract,
    ToolSpec,
)
from agent_loader import load_agent


# ===========================================================================
# 1. Registry & Metadata Exposure
# ===========================================================================

def test_registry_has_all_four_independent_tools():
    reg = get_default_tool_registry()
    names = set(reg.list_tool_names())
    assert {"csv_lookup", "calculator", "current_date", "open_meteo_weather"}.issubset(names)

    metadata = reg.get_tool_metadata_for_generator()
    assert len(metadata) == 4
    for item in metadata:
        assert "name" in item
        assert "description" in item
        assert "input_schema" in item
        assert "output_schema" in item
        assert "properties" in item["input_schema"]


# ===========================================================================
# 2. Individual Tool Execution, Validation & Error Handling
# ===========================================================================

def test_csv_lookup_tool_valid_and_errors(tmp_path):
    reg = get_default_tool_registry()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "sample.csv"

    # Valid filtered lookup
    res = reg.execute_tool(
        "csv_lookup",
        file_path=str(sample_csv),
        filter_column="status",
        filter_value="Warning",
    )
    assert res.success is True
    assert res.row_count == 1
    assert res.rows[0]["component"] == "Coolant Valve B"

    # Missing file -> graceful structured error
    res_missing = reg.execute_tool("csv_lookup", file_path=str(tmp_path / "missing.csv"))
    assert res_missing.success is False
    assert "not found" in res_missing.error.lower()

    # Invalid column -> graceful structured error
    res_col = reg.execute_tool(
        "csv_lookup",
        file_path=str(sample_csv),
        filter_column="nonexistent_col",
        filter_value="foo",
    )
    assert res_col.success is False
    assert "not found" in res_col.error.lower()

    # Input validation failure (empty file_path)
    res_invalid = reg.execute_tool("csv_lookup", file_path="")
    assert res_invalid.success is False
    assert "validation error" in res_invalid.error.lower()


def test_calculator_tool_safe_and_blocks_arbitrary_python():
    reg = get_default_tool_registry()

    # Valid arithmetic
    res = reg.execute_tool("calculator", expression="(120 * 1.5) + sqrt(144)")
    assert res.success is True
    assert res.result == 192.0

    # Division by zero -> structured error
    res_div0 = reg.execute_tool("calculator", expression="100 / 0")
    assert res_div0.success is False
    assert "zero" in res_div0.error.lower()

    # Arbitrary Python execution attempts MUST be blocked
    malicious_payloads = [
        "__import__('os').system('echo hacked')",
        "open('secret.txt').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "eval('1 + 1')",
        "lambda x: x + 1",
    ]
    for payload in malicious_payloads:
        blocked = reg.execute_tool("calculator", expression=payload)
        assert blocked.success is False, f"Security failure: payload was not blocked: {payload}"
        assert blocked.result is None


def test_current_date_tool():
    reg = get_default_tool_registry()

    res = reg.execute_tool("current_date", timezone_name="UTC")
    assert res.success is True
    assert res.current_date is not None
    assert len(res.current_date) == 10  # YYYY-MM-DD
    assert res.day_of_week is not None

    # Invalid timezone -> structured error
    res_bad = reg.execute_tool("current_date", timezone_name="Invalid/Timezone")
    assert res_bad.success is False
    assert res_bad.error is not None


def test_open_meteo_tool_validation_and_http_handling():
    reg = get_default_tool_registry()

    # Input validation: out-of-range latitude
    res_bad_lat = reg.execute_tool("open_meteo_weather", latitude=150.0, longitude=10.0)
    assert res_bad_lat.success is False
    assert "validation error" in res_bad_lat.error.lower()

    # Mocked keyless HTTP response from Open-Meteo
    fake_payload = json.dumps({
        "latitude": 52.52,
        "longitude": 13.41,
        "current_weather": {
            "temperature": 18.4,
            "windspeed": 11.2,
            "weathercode": 2,
            "time": "2026-11-26T14:00",
        },
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = fake_payload
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res_ok = reg.execute_tool(
            "open_meteo_weather",
            latitude=52.52,
            longitude=13.41,
            location_name="Berlin",
        )

    assert res_ok.success is True
    assert res_ok.location == "Berlin"
    assert res_ok.temperature_celsius == 18.4
    assert res_ok.windspeed_kmh == 11.2


# ===========================================================================
# 3. Selective Tool Assignment & Runtime Attachment
# ===========================================================================

def test_hr_agent_does_not_receive_weather_or_calculator_tools():
    reg = get_default_tool_registry()
    hr_tools = reg.select_tools_for_requirements(
        scope="Answer employee HR questions about parental leave, dress code, and company benefits.",
        must_answer_topics=["Parental leave policy", "Health benefits enrollment"],
        user_expectations="Provide accurate policy summaries from the HR handbook PDF.",
        document_metadata=[{"filename": "hr_handbook.pdf", "file_type": "pdf"}],
    )

    assert "open_meteo_weather" not in hr_tools
    assert "calculator" not in hr_tools
    assert "csv_lookup" not in hr_tools


def test_procurement_agent_receives_csv_lookup_and_calculator_tools():
    reg = get_default_tool_registry()
    procurement_tools = reg.select_tools_for_requirements(
        scope="Analyze procurement vendor quotes, compare unit prices in CSV tables, and calculate total order cost with tax.",
        must_answer_topics=["Vendor pricing lookup", "Order total and budget calculation"],
        user_expectations="Look up supplier rows and compute total procurement cost.",
        document_metadata=[{"filename": "vendor_quotes.csv", "file_type": "csv"}],
    )

    assert "csv_lookup" in procurement_tools
    assert "calculator" in procurement_tools
    assert "open_meteo_weather" not in procurement_tools


def test_runtime_attaches_only_selected_tools(tmp_path):
    reg = get_default_tool_registry()
    # Procurement agent with ONLY csv_lookup and calculator selected
    selected_specs = reg.to_tool_specs(["csv_lookup", "calculator"])

    agent_def = AgentDefinition(
        agent_name="Procurement Analyst",
        description="Analyzes vendor CSVs and calculates order totals.",
        system_prompt="Use csv_lookup and calculator.",
        audience="Procurement team",
        tone="Analytical",
        supported_topics=["Vendor pricing"],
        unsupported_topics=["Weather"],
        retrieval_config=RetrievalConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200,
            top_k=4,
            similarity_threshold=1.35,
            collection_name="test_proc",
            persist_directory=str(tmp_path / "chroma"),
        ),
        selected_tools=selected_specs,
        refusal_policy=RefusalPolicy(
            refusal_rules=["Refuse weather"],
            unsupported_topics_handling="Refuse",
            must_refuse_topics_handling="Refuse",
            default_refusal_response="Cannot help.",
        ),
        output_contract=OutputContract(
            tone_and_style="Concise",
            constraints=["Cite CSV"],
        ),
    )

    mock_vs = MagicMock()
    mock_llm = MagicMock()
    bundle = load_agent(agent_def, custom_vector_store=mock_vs, llm=mock_llm)

    assert set(bundle.tool_map.keys()) == {"csv_lookup", "calculator"}
    assert "open_meteo_weather" not in bundle.tool_map
    assert "current_date" not in bundle.tool_map

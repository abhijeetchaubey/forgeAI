import pytest
from pathlib import Path
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from generator import (
    AgentDefinition,
    RetrievalConfig,
    RefusalPolicy,
    OutputContract,
    ToolSpec,
    generate_agent,
    DEFAULT_TOOL_REGISTRY,
    _LLMGeneratedAgentComponents
)
from intake import RequirementSpec
from groundability import GroundabilityReport


# ---------------------------------------------------------------------------
# Schema and Configuration Validation Tests
# ---------------------------------------------------------------------------

def test_retrieval_config_validation():
    # Valid configuration
    cfg = RetrievalConfig(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        top_k=4,
        similarity_threshold=1.35,
        collection_name="test_col",
        persist_directory="./data/chroma",
    )
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 200

    # Overlap >= size must raise ValidationError
    with pytest.raises(ValidationError):
        RetrievalConfig(
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=500,
            top_k=4,
            similarity_threshold=1.35,
            collection_name="test_col",
            persist_directory="./data/chroma",
        )


def test_agent_definition_json_roundtrip(tmp_path):
    agent_def = AgentDefinition(
        agent_name="Compliance Auditor AI",
        description="Audits regulatory compliance filings.",
        system_prompt="You are a strict compliance auditor.",
        audience="Compliance officers",
        tone="Formal, objective",
        supported_topics=["Article 12 disclosure rules"],
        unsupported_topics=["Tax advice", "Speculative forecasting"],
        retrieval_config=RetrievalConfig(
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=800,
            chunk_overlap=100,
            top_k=5,
            similarity_threshold=1.20,
            collection_name="compliance_col",
            persist_directory="./data/compliance_chroma",
        ),
        selected_tools=[
            ToolSpec(
                name="document_retriever",
                description="Fetches compliance docs.",
                parameters={"query": {"type": "string"}},
                enabled=True
            )
        ],
        refusal_policy=RefusalPolicy(
            refusal_rules=["Never provide tax or investment advice."],
            unsupported_topics_handling="State that filing documentation does not contain this information.",
            must_refuse_topics_handling="Refuse immediately.",
            default_refusal_response="I cannot assist with tax or speculative matters.",
        ),
        output_contract=OutputContract(
            format="markdown",
            require_citations=True,
            citation_format="[source: {filename}, page: {page_number}]",
            tone_and_style="Formal and objective",
            constraints=["Must cite legal section"]
        )
    )

    out_file = tmp_path / "test_agent.json"
    saved = agent_def.save_to_json(out_file)
    assert saved.exists()

    loaded = AgentDefinition.load_from_json(out_file)
    assert loaded.agent_name == "Compliance Auditor AI"
    assert loaded.retrieval_config.chunk_size == 800
    assert len(loaded.selected_tools) == 1
    assert loaded.selected_tools[0].name == "document_retriever"
    assert loaded.supported_topics == ["Article 12 disclosure rules"]


# ---------------------------------------------------------------------------
# Tool Selection Tests
# ---------------------------------------------------------------------------

def test_tool_selection_from_custom_registry(tmp_path):
    custom_registry = [
        ToolSpec(
            name="document_retriever",
            description="Searches compliance docs.",
            parameters={"query": {"type": "string"}},
            enabled=True
        ),
        ToolSpec(
            name="sec_filing_validator",
            description="Validates SEC filing formats and deadlines.",
            parameters={"cik": {"type": "string"}},
            enabled=True
        ),
        ToolSpec(
            name="unrelated_game_engine",
            description="Simulates game physics.",
            parameters={},
            enabled=False
        )
    ]

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Mock structured response selecting specific tools
    mock_structured.invoke.return_value = _LLMGeneratedAgentComponents(
        agent_name="SEC Filing Compliance Agent",
        description="Assists with SEC regulatory deadlines and filing checks.",
        system_prompt="You verify SEC compliance using provided documents and filings validator.",
        selected_tool_names=["sec_filing_validator", "document_retriever", "non_existent_tool"],
        refusal_rules=["Decline investment recommendations."],
        unsupported_topics_handling="State that evidence is absent.",
        must_refuse_topics_handling="Refuse investment questions.",
        default_refusal_response="I cannot assist with investment matters.",
        tone_and_style="Rigorous and clear",
        output_constraints=["Cite SEC rules"]
    )

    req = RequirementSpec(
        project_name="SEC Compliance",
        intended_audience="Compliance team",
        scope="Verify SEC filing deadlines and requirements",
        tone="Precise",
        must_answer_topics=["Filing deadlines"],
        must_refuse_topics=["Stock trading tips"],
        raw_brief="Help our team verify SEC filing rules."
    )

    ground = GroundabilityReport(
        total_topics_analyzed=1,
        supported_count=1,
        partially_supported_count=0,
        unsupported_count=0,
        topic_evaluations=[],
        guaranteed_capabilities=["Filing deadlines"],
        partially_supported_topics=[],
        blocked_unsupported_topics=[],
        recommendations=[]
    )

    agent_def = generate_agent(
        requirements=req,
        groundability=ground,
        tool_registry=custom_registry,
        output_path=tmp_path / "custom_agent.json",
        llm=mock_llm
    )

    selected_names = [t.name for t in agent_def.selected_tools]
    assert "sec_filing_validator" in selected_names
    assert "document_retriever" in selected_names
    assert "unrelated_game_engine" not in selected_names
    assert "non_existent_tool" not in selected_names  # Filtered out safely


def test_generator_retry_loop_recovery(tmp_path):
    """Verifies that the generator retries and recovers from initial failures."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # First attempt raises exception, second attempt succeeds
    mock_structured.invoke.side_effect = [
        RuntimeError("Transient API connection error"),
        _LLMGeneratedAgentComponents(
            agent_name="Resilient Agent",
            description="Agent that recovers from transient errors.",
            system_prompt="You are a helpful and grounded assistant.",
            selected_tool_names=["document_retriever"],
            refusal_rules=["Refuse unverified claims."],
            unsupported_topics_handling="State information is missing.",
            must_refuse_topics_handling="Refuse.",
            default_refusal_response="Cannot answer.",
            tone_and_style="Professional",
            output_constraints=["Strict grounding"]
        )
    ]

    req = RequirementSpec(
        project_name="Test Resilient",
        intended_audience="Engineers",
        scope="Diagnostics",
        tone="Clear",
        must_answer_topics=["Sensor telemetry"],
        must_refuse_topics=["Legal claims"],
        raw_brief="Test brief"
    )

    ground = GroundabilityReport(
        total_topics_analyzed=1,
        supported_count=1,
        partially_supported_count=0,
        unsupported_count=0,
        topic_evaluations=[],
        guaranteed_capabilities=["Sensor telemetry"],
        partially_supported_topics=[],
        blocked_unsupported_topics=[],
        recommendations=[]
    )

    agent_def = generate_agent(
        requirements=req,
        groundability=ground,
        output_path=tmp_path / "resilient_agent.json",
        llm=mock_llm,
        max_retries=3
    )

    assert agent_def.agent_name == "Resilient Agent"
    assert mock_structured.invoke.call_count == 2


# ---------------------------------------------------------------------------
# End-to-End Live Generation Test
# ---------------------------------------------------------------------------

def test_live_agent_generation_from_files(tmp_path):
    """End-to-end integration test reading existing requirements.json & groundability_report.json."""
    req_file = Path("requirements.json")
    ground_file = Path("groundability_report.json")
    out_file = tmp_path / "e2e_agent.json"

    if req_file.exists() and ground_file.exists():
        agent = generate_agent(
            requirements=req_file,
            groundability=ground_file,
            output_path=out_file
        )
        assert out_file.exists()
        assert len(agent.supported_topics) > 0
        assert len(agent.unsupported_topics) > 0
        assert len(agent.selected_tools) > 0
        assert agent.retrieval_config.chunk_size > 0
        assert agent.refusal_policy.default_refusal_response != ""

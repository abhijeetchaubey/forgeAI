import pytest
from pathlib import Path
from groundability import (
    GroundingStatus,
    GroundabilityReport,
    TopicGroundingEvaluation,
    EvidenceItem,
    evaluate_topic_groundability,
    analyze_groundability,
)
from intake import RequirementSpec
from src.vectorstore.chroma_store import get_vector_store

def test_evaluate_supported_topic(temp_chroma_dir):
    # Simulated concrete evidence for thermal cutoff
    evidence = [
        EvidenceItem(
            source_file="sample.pdf",
            page_number=2,
            chunk_index=1,
            similarity_score=0.85,
            excerpt="ForgeAI Robotic Cell Specification - Page 2\nEmergency cutoff temperature: 85 Celsius.\nAutomated thermal shutdown protocols active."
        ),
        EvidenceItem(
            source_file="sample.csv",
            page_number=1,
            row_number=4,
            chunk_index=3,
            similarity_score=0.92,
            excerpt="component: Emergency Brake D, status: Critical, temperature_celsius: 91.2"
        )
    ]

    evaluation = evaluate_topic_groundability(
        topic="Emergency thermal cutoff trigger temperature",
        evidence_items=evidence
    )

    assert evaluation.status == GroundingStatus.SUPPORTED
    assert evaluation.confidence >= 0.8
    assert len(evaluation.evidence) > 0
    assert evaluation.missing_information is None or evaluation.missing_information == ""

def test_evaluate_partially_supported_topic():
    # Simulated partial evidence: mentions flow reduction and warning, but no calibration or repair steps
    evidence = [
        EvidenceItem(
            source_file="sample.csv",
            page_number=1,
            row_number=2,
            chunk_index=1,
            similarity_score=1.10,
            excerpt="id: 102, component: Coolant Valve B, status: Warning, Flow rate reduced by 15 percent."
        )
    ]

    evaluation = evaluate_topic_groundability(
        topic="Step-by-step disassembly and overhaul procedure for Coolant Valve B",
        evidence_items=evidence
    )

    assert evaluation.status in [GroundingStatus.PARTIALLY_SUPPORTED, GroundingStatus.UNSUPPORTED]
    assert evaluation.missing_information is not None
    assert len(evaluation.missing_information) > 0

def test_evaluate_unsupported_topic():
    # Completely empty or irrelevant evidence
    evaluation = evaluate_topic_groundability(
        topic="Quantum cryptographic key exchange protocols for factory robots",
        evidence_items=[],
        distance_threshold=1.20
    )

    assert evaluation.status == GroundingStatus.UNSUPPORTED
    assert evaluation.confidence >= 0.8
    assert evaluation.evidence == []
    assert "absent" in evaluation.missing_information.lower() or "no relevant" in evaluation.reasoning.lower()

def test_groundability_guard_blocks_unsupported(tmp_path):
    report_file = tmp_path / "test_groundability_report.json"
    topics = [
        "Emergency cutoff temperature protocols",       # Supported in sample knowledge base
        "Lunar mission trajectory guidance algorithms",  # Completely unsupported
    ]

    store = get_vector_store()
    report = analyze_groundability(
        requirements=topics,
        vector_store=store,
        output_path=report_file
    )

    assert isinstance(report, GroundabilityReport)
    assert report_file.exists()

    # Verify that unsupported topic is STRICTLY prevented from guaranteed capabilities
    assert "Lunar mission trajectory guidance algorithms" not in report.guaranteed_capabilities
    assert "Lunar mission trajectory guidance algorithms" in report.blocked_unsupported_topics
    assert report.unsupported_count >= 1

    # Verify JSON persistence
    loaded = GroundabilityReport.load_from_json(report_file)
    assert loaded.total_topics_analyzed == 2
    assert loaded.unsupported_count == report.unsupported_count
    assert loaded.guaranteed_capabilities == report.guaranteed_capabilities

def test_groundability_from_requirement_spec(tmp_path):
    spec = RequirementSpec(
        project_name="Diagnostic Assistant",
        intended_audience="Operators",
        scope="Factory Floor",
        tone="Technical",
        must_answer_topics=[
            "Operating pressure of robotic cell",
            "Deep sea submarine navigation system"
        ],
        raw_brief="Test brief",
    )

    report = analyze_groundability(spec, output_path=tmp_path / "rep.json")
    assert report.total_topics_analyzed == 2
    assert "Deep sea submarine navigation system" in report.blocked_unsupported_topics
    assert "Deep sea submarine navigation system" not in report.guaranteed_capabilities

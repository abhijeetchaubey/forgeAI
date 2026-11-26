import pytest
from pathlib import Path
from pydantic import ValidationError
from intake import RequirementSpec, conduct_intake

def test_requirement_spec_validation():
    # Test valid model creation and constraints
    spec = RequirementSpec(
        project_name="Factory Assistant",
        intended_audience="Operators",
        scope="Machine troubleshooting",
        tone="Concise",
        user_expectations=["Quick answers"],
        must_answer_topics=["Hydraulic pressure"],
        must_refuse_topics=["Finance", "Salaries"],
        clarification_questions=[f"Question {i}" for i in range(10)],  # Over 5
        raw_brief="Help operators troubleshoot machines.",
    )
    # Must enforce at most 5 clarification questions
    assert len(spec.clarification_questions) == 5

def test_requirement_spec_json_io(tmp_path):
    spec = RequirementSpec(
        project_name="Test Spec",
        intended_audience="Technicians",
        scope="Diagnostics",
        tone="Technical",
        user_expectations=["Accuracy"],
        must_answer_topics=["Motor temperature"],
        must_refuse_topics=["Stock prices"],
        clarification_questions=["What is the operating voltage?"],
        raw_brief="Diagnose motors.",
    )
    json_path = tmp_path / "test_requirements.json"
    saved_path = spec.save_to_json(json_path)
    assert saved_path.exists()

    loaded = RequirementSpec.load_from_json(json_path)
    assert loaded.project_name == spec.project_name
    assert loaded.must_answer_topics == spec.must_answer_topics
    assert loaded.must_refuse_topics == spec.must_refuse_topics
    assert len(loaded.clarification_questions) == 1

def test_conduct_intake_empty_brief_raises():
    with pytest.raises(ValueError, match="Brief cannot be empty"):
        conduct_intake("")

def test_conduct_intake_execution(tmp_path):
    brief = (
        "Build an AI assistant for robotic cell technicians. "
        "It must explain pneumatic pressure alarms and thermal shutdown thresholds. "
        "It must strictly refuse to answer company salary or personal questions."
    )
    doc_meta = [{"filename": "cell_specs.pdf"}]
    output_file = tmp_path / "req.json"

    spec = conduct_intake(
        brief=brief,
        document_metadata=doc_meta,
        output_path=output_file
    )

    assert isinstance(spec, RequirementSpec)
    assert output_file.exists()
    assert spec.intended_audience != ""
    assert len(spec.must_answer_topics) >= 1
    assert len(spec.must_refuse_topics) >= 1
    assert len(spec.clarification_questions) <= 5
    assert "cell_specs.pdf" in spec.documents_referenced

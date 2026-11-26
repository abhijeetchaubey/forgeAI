import pytest
from pathlib import Path
from unittest.mock import MagicMock, call
from pydantic import ValidationError
from langchain_core.documents import Document

from eval_generator import (
    TestCase,
    EvaluationSet,
    EvalCategory,
    _normalize_question,
    _build_grounded_cases_prompt,
    _invoke_case_generation,
    verify_test_case_evidence,
    _LLMGeneratedTestCase,
    _LLMEvalSetWrapper,
    PROMPTS_DIR,
    ensure_prompts_saved,
    load_prompt_templates,
)


def test_grounded_prompt_does_not_request_multi_document_cases_from_one_source():
    document = Document(
        page_content="Full-time employees receive 22 annual PTO days.",
        metadata={"filename": "leave_policy.txt"},
    )

    prompt = _build_grounded_cases_prompt([document])

    assert "Do not generate 'multi_document' cases." in prompt
    assert "category: 'answerable'" in prompt


def test_grounded_prompt_requests_cross_document_cases_for_multiple_sources():
    documents = [
        Document(page_content="PTO policy.", metadata={"filename": "leave_policy.txt"}),
        Document(page_content="Benefits policy.", metadata={"filename": "benefits.txt"}),
    ]

    prompt = _build_grounded_cases_prompt(documents)

    assert "'multi_document'" in prompt


def test_case_generation_uses_strict_json_schema():
    test_case = _LLMGeneratedTestCase(
        question="How many annual leave days are provided?",
        category=EvalCategory.ANSWERABLE,
        expected_facts=["22 days"],
        expected_source_document=["leave_policy.txt"],
        expected_page_number="1",
        should_refuse=False,
        evaluation_criteria="Must state 22 days.",
    )
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _LLMEvalSetWrapper(test_cases=[test_case])
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm

    cases = _invoke_case_generation(llm, "system prompt", "human prompt")

    assert cases == [test_case]
    llm.with_structured_output.assert_called_once_with(
        _LLMEvalSetWrapper,
        method="json_schema",
        strict=True,
    )


def test_case_generation_falls_back_to_json_mode():
    test_case = _LLMGeneratedTestCase(
        question="How many annual leave days are provided?",
        category=EvalCategory.ANSWERABLE,
        expected_facts=["22 days"],
        expected_source_document=["leave_policy.txt"],
        expected_page_number="1",
        should_refuse=False,
        evaluation_criteria="Must state 22 days.",
    )
    strict_llm = MagicMock()
    strict_llm.invoke.side_effect = ValueError("strict schema unavailable")
    json_mode_llm = MagicMock()
    json_mode_llm.invoke.return_value = _LLMEvalSetWrapper(test_cases=[test_case])
    llm = MagicMock()
    llm.with_structured_output.side_effect = [strict_llm, json_mode_llm]

    cases = _invoke_case_generation(llm, "system prompt", "human prompt")

    assert cases == [test_case]
    assert llm.with_structured_output.call_args_list == [
        call(_LLMEvalSetWrapper, method="json_schema", strict=True),
        call(_LLMEvalSetWrapper, method="json_mode"),
    ]


def test_test_case_schema_validation():
    # Valid test case
    tc = TestCase(
        test_id="EVAL-001",
        question="What is the operating pressure?",
        category=EvalCategory.ANSWERABLE,
        expected_facts=["120 bar"],
        expected_source_document=["sample.pdf"],
        expected_page_number="page 1",
        should_refuse=False,
        evaluation_criteria="Must state 120 bar."
    )
    assert tc.test_id == "EVAL-001"
    assert not tc.should_refuse

    # Out of scope questions must enforce should_refuse = True
    tc_oos = TestCase(
        test_id="EVAL-002",
        question="What are company stock options?",
        category=EvalCategory.OUT_OF_SCOPE,
        expected_facts=["Refuse out-of-scope"],
        expected_source_document=[],
        expected_page_number=None,
        should_refuse=False,  # Validator should enforce True
        evaluation_criteria="Must decline."
    )
    assert tc_oos.should_refuse is True


def test_eval_set_json_roundtrip(tmp_path):
    tc = TestCase(
        test_id="EVAL-001",
        question="Test question?",
        category=EvalCategory.ANSWERABLE,
        expected_facts=["Fact 1"],
        expected_source_document=["sample.txt"],
        expected_page_number="page 1",
        should_refuse=False,
        evaluation_criteria="Criteria 1"
    )
    eval_set = EvaluationSet(
        total_test_cases=1,
        category_counts={"answerable": 1},
        test_cases=[tc]
    )

    out_file = tmp_path / "test_eval_set.json"
    saved = eval_set.save_to_json(out_file)
    assert saved.exists()

    loaded = EvaluationSet.load_from_json(out_file)
    assert loaded.total_test_cases == 1
    assert loaded.test_cases[0].question == "Test question?"


def test_question_normalization_and_deduplication():
    q1 = "What is the emergency cutoff temperature?"
    q2 = "WHAT IS THE EMERGENCY CUTOFF TEMPERATURE???"
    q3 = "   what is the emergency cutoff temperature?   "
    assert _normalize_question(q1) == _normalize_question(q2) == _normalize_question(q3)


def test_prompts_saved_in_prompts_directory():
    ensure_prompts_saved(PROMPTS_DIR)
    assert (PROMPTS_DIR / "eval_system_prompt.txt").exists()
    assert (PROMPTS_DIR / "eval_generation_prompt.txt").exists()

    templates = load_prompt_templates(PROMPTS_DIR)
    assert "ForgeAI Automatic Evaluation Generator" in templates["system"]
    assert "answerable" in templates["generation"]


def test_saved_eval_set_compliance():
    """Verifies that the generated eval_set.json satisfies all Phase 5 requirements."""
    eval_file = Path("eval_set.json")
    assert eval_file.exists(), "eval_set.json must exist"

    eval_set = EvaluationSet.load_from_json(eval_file)
    # 1. At least 20 test cases
    assert eval_set.total_test_cases >= 20, f"Expected >= 20 cases, got {eval_set.total_test_cases}"

    # 2. Contains all four categories
    categories = {tc.category for tc in eval_set.test_cases}
    assert EvalCategory.ANSWERABLE in categories
    assert EvalCategory.MULTI_DOCUMENT in categories
    assert EvalCategory.AMBIGUOUS in categories
    assert EvalCategory.OUT_OF_SCOPE in categories

    # 3. Check uniqueness of questions
    questions = [tc.question.strip().lower() for tc in eval_set.test_cases]
    assert len(questions) == len(set(questions)), "Duplicate questions found in evaluation set"

    # 4. Check out of scope refusal consistency
    for tc in eval_set.test_cases:
        if tc.category == EvalCategory.OUT_OF_SCOPE:
            assert tc.should_refuse is True
        elif tc.category in [EvalCategory.ANSWERABLE, EvalCategory.MULTI_DOCUMENT]:
            assert tc.should_refuse is False
            assert len(tc.expected_facts) > 0
            assert len(tc.expected_source_document) > 0

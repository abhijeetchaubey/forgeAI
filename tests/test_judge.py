"""
Unit and Integration Tests for ForgeAI Phase 6: Automated Judge & Evaluation System
Verifies:
1. Deliberately correct answers.
2. Deliberately incorrect answers.
3. Deliberately unsupported questions with failure to refuse (auto-fail).
4. Deliberately correctly refused unsupported questions.
5. Deliberately partially correct answers.
6. Confident, lengthy fluff answers (anti-bloat & confidence penalty enforcement).
7. Deterministic citation and fact coverage checks.
8. Scorecard generation and JSON artifact serialization.
"""

import pytest
from pathlib import Path
from pydantic import BaseModel

from eval_generator import TestCase, EvalCategory, EvaluationSet
from chat_runner import AgentResponse, Citation
from deterministic_checks import (
    run_deterministic_checks,
    detect_refusal,
    check_citation_correctness,
    check_facts_coverage,
    calculate_grounding_and_bloat_penalties,
    DeterministicResult
)
from scoring import (
    ScoringConfig,
    ScoringWeights,
    calculate_test_case_scores,
    generate_scorecards,
    save_evaluation_artifacts,
    LLMEvalResult
)
from judge import EvaluationJudge, LLMJudge


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_answerable_case() -> TestCase:
    return TestCase(
        test_id="EVAL-001",
        question="What is the maintenance cycle for Hydraulic Pump A?",
        category=EvalCategory.ANSWERABLE,
        expected_facts=["30 days"],
        expected_source_document=["sample.csv"],
        expected_page_number=None,
        should_refuse=False,
        evaluation_criteria="The answer must state the maintenance cycle as '30 days' and cite sample.csv."
    )


@pytest.fixture
def sample_multi_doc_case() -> TestCase:
    return TestCase(
        test_id="EVAL-008",
        question="Which components are due for maintenance within the next 15 days?",
        category=EvalCategory.MULTI_DOCUMENT,
        expected_facts=[
            "Coolant Valve B – maintenance_cycle: 14 days",
            "Emergency Brake D – maintenance_cycle: 7 days"
        ],
        expected_source_document=["sample.csv", "sample.txt"],
        expected_page_number="page 1",
        should_refuse=False,
        evaluation_criteria="Must list both Coolant Valve B (14 days) and Emergency Brake D (7 days) citing sample.csv and sample.txt."
    )


@pytest.fixture
def sample_out_of_scope_case() -> TestCase:
    return TestCase(
        test_id="EVAL-018",
        question="What is the warranty period for the hydraulic pump?",
        category=EvalCategory.OUT_OF_SCOPE,
        expected_facts=["Refusal reason: warranty information is not covered."],
        expected_source_document=[],
        expected_page_number=None,
        should_refuse=True,
        evaluation_criteria="The answer must refuse, stating warranty details are not in the corpus."
    )


# ===========================================================================
# 1. Deliberately Correct Answer Tests
# ===========================================================================

def test_deliberately_correct_answer(sample_answerable_case):
    """A correct answer with expected facts and citations must achieve high scores and PASS."""
    correct_response = AgentResponse(
        answer="The maintenance cycle for Hydraulic Pump A is exactly 30 days as specified in the telemetry records. [sample.csv]",
        is_refusal=False,
        citations=[
            Citation(source="sample.csv", page_number=1, excerpt="Hydraulic Pump A,Operational,30 days,68.5")
        ],
        agent_name="TestAgent"
    )

    judge = EvaluationJudge(mock_judge=True)
    result = judge.evaluate_test_case(sample_answerable_case, precomputed_response=correct_response)

    assert result.passed is True
    assert result.scores.status == "PASS"
    assert result.scores.composite_score >= 0.85
    assert result.scores.factual_correctness >= 0.90
    assert result.scores.expected_facts_coverage == 1.0
    assert result.scores.citation_correctness >= 0.90
    assert "30 days" in result.deterministic_checks.facts_covered


# ===========================================================================
# 2. Deliberately Incorrect Answer Tests
# ===========================================================================

def test_deliberately_incorrect_answer(sample_answerable_case):
    """An incorrect answer with wrong facts must fail fact matching and be graded FAIL."""
    incorrect_response = AgentResponse(
        answer="The maintenance cycle for Hydraulic Pump A is 180 days and operates at high temperature. [sample.csv]",
        is_refusal=False,
        citations=[
            Citation(source="sample.csv", page_number=1, excerpt="Hydraulic Pump A,Operational,30 days,68.5")
        ],
        agent_name="TestAgent"
    )

    judge = EvaluationJudge(mock_judge=True)
    result = judge.evaluate_test_case(sample_answerable_case, precomputed_response=incorrect_response)

    assert result.passed is False
    assert result.scores.status == "FAIL"
    assert result.scores.expected_facts_coverage == 0.0
    assert "30 days" in result.deterministic_checks.facts_missing
    assert result.scores.composite_score < 0.70


# ===========================================================================
# 3. Deliberately Unsupported Answer (Fails to Refuse)
# ===========================================================================

def test_deliberately_unsupported_unrefused_answer(sample_out_of_scope_case):
    """
    When an out-of-scope question is answered instead of refused,
    strict refusal enforcement must trigger an automatic FAIL.
    """
    hallucinated_response = AgentResponse(
        answer="The warranty period for the hydraulic pump is 5 years with full replacement coverage under standard factory terms.",
        is_refusal=False,  # Failed to refuse!
        citations=[],
        agent_name="TestAgent"
    )

    judge = EvaluationJudge(mock_judge=True)
    result = judge.evaluate_test_case(sample_out_of_scope_case, precomputed_response=hallucinated_response)

    assert result.passed is False
    assert result.scores.status == "FAIL"
    assert result.deterministic_checks.refusal_passed is False
    assert result.scores.refusal_accuracy == 0.0
    assert "STRICT REFUSAL VIOLATION" in result.scores.verdict_reason


# ===========================================================================
# 4. Deliberately Correctly Refused Unsupported Answer
# ===========================================================================

def test_deliberately_correctly_refused_answer(sample_out_of_scope_case):
    """
    When an out-of-scope question is correctly declined per policy,
    it must receive full refusal credit and PASS.
    """
    refusal_response = AgentResponse(
        answer="I cannot provide warranty details because warranty information is not mentioned in the provided documents.",
        is_refusal=True,
        citations=[],
        agent_name="TestAgent"
    )

    judge = EvaluationJudge(mock_judge=True)
    result = judge.evaluate_test_case(sample_out_of_scope_case, precomputed_response=refusal_response)

    assert result.passed is True
    assert result.scores.status == "PASS"
    assert result.scores.refusal_accuracy == 1.0
    assert result.deterministic_checks.refusal_passed is True
    assert result.scores.composite_score >= 0.80


# ===========================================================================
# 5. Deliberately Partially Correct Answer
# ===========================================================================

def test_deliberately_partially_correct_answer(sample_multi_doc_case):
    """
    An answer covering only 1 of 2 expected facts and missing a citation
    must receive partial fact coverage and lower score.
    """
    partially_correct_response = AgentResponse(
        answer="Coolant Valve B has a maintenance cycle of 14 days and is due soon. [sample.csv]",
        is_refusal=False,
        citations=[
            Citation(source="sample.csv", page_number=1, excerpt="Coolant Valve B,Warning,14 days")
        ],
        agent_name="TestAgent"
    )

    judge = EvaluationJudge(mock_judge=True)
    result = judge.evaluate_test_case(sample_multi_doc_case, precomputed_response=partially_correct_response)

    # 1 out of 2 expected facts covered
    assert result.deterministic_checks.facts_coverage_score == 0.50
    # sample.txt missing from citations
    assert "sample.txt" in result.deterministic_checks.missing_sources
    # Score reflects partial credit
    assert result.scores.composite_score < 0.70
    assert result.scores.status == "FAIL"
    assert "sample.txt" in result.scores.verdict_reason or "Missing" in result.scores.verdict_reason


# ===========================================================================
# 6. Lengthy & Confident Hallucination (Anti-Bloat & Confidence Penalty)
# ===========================================================================

def test_confident_lengthy_fluff_answer_penalized(sample_answerable_case):
    """
    Rule: Do not award full credit merely because an answer is lengthy or confident.
    A 150+ word answer full of empty confidence and zero actual facts must be heavily penalized.
    """
    fluff_answer = (
        "Without a doubt, it is a proven fact without question that the comprehensive engineering "
        "infrastructure governing the robotic automation cells is operating under peak efficiency. "
        "I am 100% certain and completely guaranteed that our preventive maintenance framework "
        "follows world-class industrial methodologies. The mechanical actuators, hydraulic conduits, "
        "and pneumatic subsystems undergo rigorous, flawless inspection routines continuously. "
        "It is indisputable that every single component is handled with utmost technical rigor, "
        "exceeding all international compliance benchmarks across all manufacturing workcells."
    )
    bloat_response = AgentResponse(
        answer=fluff_answer,
        is_refusal=False,
        citations=[],
        agent_name="TestAgent"
    )

    judge = EvaluationJudge(mock_judge=True)
    result = judge.evaluate_test_case(sample_answerable_case, precomputed_response=bloat_response)

    # Must fail because expected fact '30 days' is completely missing
    assert result.passed is False
    assert result.scores.status == "FAIL"

    # Anti-bloat / verbosity / confidence penalties must have triggered
    det = result.deterministic_checks
    assert det.verbosity_penalty > 0.0 or det.confidence_penalty > 0.0
    assert det.facts_coverage_score == 0.0


# ===========================================================================
# 7. Deterministic Checks Unit Tests
# ===========================================================================

def test_detect_refusal_patterns():
    # True positives
    assert detect_refusal("I cannot answer this question based on the context.") is True
    assert detect_refusal("This information is not mentioned in the provided documents.") is True
    assert detect_refusal("The query falls outside the scope of the manual.") is True
    assert detect_refusal("Topic is an unsupported topic.", is_refusal_flag=True) is True

    # True negatives
    assert detect_refusal("The operating pressure is 120 bar as described in sample.pdf.") is False
    assert detect_refusal("Hydraulic Pump A maintenance cycle is 30 days.") is False


def test_citation_correctness_checker():
    # Exact source match
    score, passed, cited, missing, page_match = check_citation_correctness(
        expected_sources=["sample.pdf"],
        cited_citations=[{"source": "sample.pdf", "page_number": 2}],
        answer_text="Source is sample.pdf page 2.",
        expected_page_number="page 2"
    )
    assert passed is True
    assert score == 1.0
    assert missing == []
    assert page_match is True

    # Missing expected source
    score, passed, cited, missing, page_match = check_citation_correctness(
        expected_sources=["sample.pdf", "sample.csv"],
        cited_citations=[{"source": "sample.pdf", "page_number": 1}],
        answer_text="Refer to sample.pdf.",
        expected_page_number=None
    )
    assert passed is False
    assert "sample.csv" in missing
    assert score == 0.5


def test_facts_coverage_checker():
    # Number + unit matching
    score, covered, missing = check_facts_coverage(
        expected_facts=["30 days", "68.5 Celsius"],
        answer_text="The cycle is 30 days and temp is 68.5°C."
    )
    assert score == 1.0
    assert len(covered) == 2
    assert len(missing) == 0


# ===========================================================================
# 8. Scorecard Generation and JSON Serialization
# ===========================================================================

def test_scorecard_aggregation_and_artifacts(tmp_path, sample_answerable_case, sample_out_of_scope_case):
    judge = EvaluationJudge(mock_judge=True)

    # 1. Answerable case (PASS)
    resp1 = AgentResponse(
        answer="The maintenance cycle for Hydraulic Pump A is 30 days. [sample.csv]",
        is_refusal=False,
        citations=[Citation(source="sample.csv", page_number=1, excerpt="30 days")],
        agent_name="Agent"
    )
    r1 = judge.evaluate_test_case(sample_answerable_case, precomputed_response=resp1)

    # 2. Out of scope case (PASS via refusal)
    resp2 = AgentResponse(
        answer="I cannot help with this. Warranty information is not covered in the documents.",
        is_refusal=True,
        citations=[],
        agent_name="Agent"
    )
    r2 = judge.evaluate_test_case(sample_out_of_scope_case, precomputed_response=resp2)

    cat_scorecards, overall = generate_scorecards([r1, r2])

    assert overall.total_test_cases == 2
    assert overall.total_passed == 2
    assert overall.total_failed == 0
    assert overall.overall_pass_rate == 1.0
    assert "answerable" in cat_scorecards
    assert "out_of_scope" in cat_scorecards

    # Test saving artifacts
    res_path = tmp_path / "evaluation_results.json"
    sc_path = tmp_path / "scorecard.json"

    saved_res, saved_sc = save_evaluation_artifacts(
        test_results=[r1, r2],
        overall_scorecard=overall,
        category_scorecards=cat_scorecards,
        evaluation_results_path=res_path,
        scorecard_path=sc_path
    )

    assert saved_res.exists()
    assert saved_sc.exists()

"""
Unit and Integration Tests for ForgeAI Phase 7: Critic-Optimizer Loop
Tests:
1. Failure classification into all 4 taxonomy categories:
   - Retrieval miss
   - Prompt defect
   - Hallucination
   - Wrong refusal
2. Targeted configuration patching (only relevant sections modified).
3. Original agent preservation & versioned output file generation.
4. LangGraph conditional routing:
   - Stop on pass threshold.
   - Continue to critic/optimizer when below threshold.
   - Stop at max 3 iterations with honest report.
5. iteration_logs.json schema and field completeness.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from generator import (
    AgentDefinition,
    RetrievalConfig,
    RefusalPolicy,
    OutputContract,
    ToolSpec,
)
from critic import (
    CriticAgent,
    CriticDiagnosis,
    FailureCategory,
    classify_failure_deterministically,
)
from optimizer import (
    AgentOptimizer,
    CriticOptimizerWorkflow,
    increment_version,
    build_versioned_agent_path,
)
from scoring import OverallScorecard, TestCaseResult, TestCaseMetricScores, LLMEvalResult
from deterministic_checks import DeterministicResult


@pytest.fixture
def sample_agent_def() -> AgentDefinition:
    return AgentDefinition(
        agent_name="Test Diagnostic Agent",
        description="Diagnostic assistant for testing.",
        system_prompt="You are a diagnostic assistant. Retrieve passages with `document_retriever`.",
        audience="Engineers",
        tone="Technical",
        supported_topics=["Pump cycles"],
        unsupported_topics=["HR policies"],
        retrieval_config=RetrievalConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200,
            top_k=4,
            similarity_threshold=1.35,
            collection_name="forgeai_documents",
            persist_directory="data/chroma_db",
        ),
        selected_tools=[
            ToolSpec(name="document_retriever", description="Retrieves docs", parameters={}, enabled=True)
        ],
        refusal_policy=RefusalPolicy(
            refusal_rules=["Refuse HR questions."],
            unsupported_topics_handling="Refuse if unsupported.",
            must_refuse_topics_handling="Decline immediately.",
            default_refusal_response="I'm sorry, but I cannot help with that request.",
        ),
        output_contract=OutputContract(
            format="markdown",
            require_citations=True,
            citation_format="[source: {filename}, page: {page_number}]",
            tone_and_style="Concise",
            constraints=["Be concise."],
        ),
        version="1.0.0",
    )


def test_classify_all_four_failure_categories():
    # 1. Wrong refusal (failed to refuse out-of-scope)
    f_wrong_refusal = classify_failure_deterministically({
        "test_id": "EVAL-018",
        "question": "What is the warranty period?",
        "category": "out_of_scope",
        "generated_answer": "The warranty is 2 years.",
        "should_refuse": True,
        "is_refusal": False,
        "citations": [],
        "deterministic_checks": {"refusal_passed": False},
        "scores": {"verdict_reason": "STRICT REFUSAL VIOLATION"},
    })
    assert f_wrong_refusal.failure_category == FailureCategory.WRONG_REFUSAL

    # 2. Retrieval miss (missing expected sources)
    f_retrieval_miss = classify_failure_deterministically({
        "test_id": "EVAL-008",
        "question": "Which components are due in 15 days?",
        "category": "multi_document",
        "generated_answer": "Coolant Valve B is due in 14 days.",
        "should_refuse": False,
        "is_refusal": False,
        "citations": [{"source": "sample.csv", "page_number": 1}],
        "deterministic_checks": {
            "refusal_passed": True,
            "missing_sources": ["sample.txt"],
            "citation_score": 0.5,
        },
        "scores": {"grounding_accuracy": 0.8},
    })
    assert f_retrieval_miss.failure_category == FailureCategory.RETRIEVAL_MISS

    # 3. Hallucination
    f_hallucination = classify_failure_deterministically({
        "test_id": "EVAL-001",
        "question": "What is the maintenance cycle?",
        "category": "answerable",
        "generated_answer": "Without a doubt, 100% certain it is 999 days.",
        "should_refuse": False,
        "is_refusal": False,
        "citations": [{"source": "sample.csv", "page_number": 1}],
        "deterministic_checks": {
            "refusal_passed": True,
            "missing_sources": [],
            "confidence_penalty": 0.2,
            "facts_coverage_score": 0.0,
        },
        "llm_eval": {"hallucination_detected": True},
        "scores": {"grounding_accuracy": 0.2},
    })
    assert f_hallucination.failure_category == FailureCategory.HALLUCINATION

    # 4. Prompt defect (tool_use_failed runtime error)
    f_prompt_defect = classify_failure_deterministically({
        "test_id": "EVAL-002",
        "question": "What is the temperature reading for Emergency Brake D?",
        "category": "answerable",
        "generated_answer": "Error generating answer: Error code: 400 - {'error': {'message': 'Tool choice is none, but model called a tool', 'code': 'tool_use_failed'}}",
        "should_refuse": False,
        "is_refusal": True,
        "citations": [],
        "deterministic_checks": {"refusal_passed": False},
        "scores": {},
    })
    assert f_prompt_defect.failure_category == FailureCategory.PROMPT_DEFECT


def test_targeted_fixes_only_modify_relevant_config(sample_agent_def):
    # Diagnosis with ONLY Retrieval miss -> refusal_policy and output_contract must NOT be modified
    critic = CriticAgent(mock_mode=True)
    failed_cases = [{
        "test_id": "EVAL-008",
        "question": "Which components are due in 15 days?",
        "category": "multi_document",
        "generated_answer": "Coolant Valve B.",
        "should_refuse": False,
        "is_refusal": False,
        "citations": [{"source": "sample.csv", "page_number": 1}],
        "deterministic_checks": {"refusal_passed": True, "missing_sources": ["sample.txt"], "citation_score": 0.5},
        "scores": {"grounding_accuracy": 0.8},
    }]
    diagnosis = critic.diagnose(failed_cases, sample_agent_def, iteration=1)

    updated, cfg_before, cfg_after, fixes = AgentOptimizer.apply_targeted_fixes(sample_agent_def, diagnosis)

    assert updated.version == "1.0.1"
    assert "retrieval_config" in cfg_before
    assert "retrieval_config" in cfg_after
    assert updated.retrieval_config.top_k == 6
    # Unrelated sections remain untouched
    assert "refusal_policy" not in cfg_after
    assert "output_contract" not in cfg_after
    assert updated.agent_name == sample_agent_def.agent_name
    assert updated.system_prompt == sample_agent_def.system_prompt
    assert updated.refusal_policy == sample_agent_def.refusal_policy


def test_versioning_and_original_preservation(tmp_path, sample_agent_def):
    orig_path = tmp_path / "agent.json"
    sample_agent_def.save_to_json(orig_path)
    orig_content = orig_path.read_text(encoding="utf-8")

    assert increment_version("1.0.0") == "1.0.1"
    v_path = build_versioned_agent_path(orig_path, "1.0.1")
    assert v_path.name == "agent_v1.0.1.json"

    # Simulate saving updated version
    updated = sample_agent_def.model_copy(deep=True)
    updated.version = "1.0.1"
    updated.save_to_json(v_path)

    assert v_path.exists()
    # Original must be completely unchanged
    assert orig_path.read_text(encoding="utf-8") == orig_content


def _make_dummy_eval_output(score: float, failed_ids: list[str]):
    results = []
    for tid in failed_ids:
        results.append(
            TestCaseResult(
                test_id=tid,
                question="Mock question?",
                category="out_of_scope",
                generated_answer="Mock answer",
                is_refusal=False,
                should_refuse=True,
                deterministic_checks=DeterministicResult(refusal_passed=False, refusal_score=0.0),
                llm_eval=LLMEvalResult(),
                scores=TestCaseMetricScores(
                    factual_correctness=0.0,
                    grounding_accuracy=0.5,
                    expected_facts_coverage=1.0,
                    refusal_accuracy=0.0,
                    citation_correctness=1.0,
                    composite_score=score,
                    status="FAIL",
                    verdict_reason="STRICT REFUSAL VIOLATION",
                ),
                passed=False,
                failure_reason="STRICT REFUSAL VIOLATION",
            )
        )
    overall = OverallScorecard(
        total_test_cases=max(len(failed_ids), 1),
        total_passed=0 if failed_ids else 1,
        total_failed=len(failed_ids),
        overall_pass_rate=0.0 if failed_ids else 1.0,
        overall_composite_score=score,
        avg_metrics={},
        category_scorecards={},
        category_scores={},
        failed_test_ids=failed_ids,
        failed_test_details=[],
    )
    return results, overall, {}


def test_langgraph_routing_stops_when_passed(tmp_path, sample_agent_def):
    agent_path = tmp_path / "agent.json"
    sample_agent_def.save_to_json(agent_path)
    logs_path = tmp_path / "iteration_logs.json"

    with patch("optimizer.EvaluationJudge.from_agent_file") as mock_from_file:
        mock_judge = MagicMock()
        mock_judge.evaluate_all.return_value = _make_dummy_eval_output(0.85, [])
        mock_from_file.return_value = mock_judge

        wf = CriticOptimizerWorkflow(
            original_agent_path=agent_path,
            eval_set_path="eval_set.json",
            pass_threshold=0.70,
            max_iterations=3,
            mock_mode=True,
            iteration_logs_path=logs_path,
        )
        report = wf.run()

    assert report["status"] == "passed"
    assert report["iterations_executed"] == 0
    assert report["final_score"] == 0.85
    assert logs_path.exists()


def test_langgraph_routing_max_three_iterations_and_logs(tmp_path, sample_agent_def):
    agent_path = tmp_path / "agent.json"
    sample_agent_def.save_to_json(agent_path)
    logs_path = tmp_path / "iteration_logs.json"

    # Score improves slightly each run (0.50 -> 0.58 -> 0.64 -> 0.68) but stays below 0.70
    eval_outputs = [
        _make_dummy_eval_output(0.50, ["EVAL-018", "EVAL-020"]),
        _make_dummy_eval_output(0.58, ["EVAL-018"]),
        _make_dummy_eval_output(0.64, ["EVAL-018"]),
        _make_dummy_eval_output(0.68, ["EVAL-018"]),
    ]

    with patch("optimizer.EvaluationJudge.from_agent_file") as mock_from_file:
        mock_judge = MagicMock()
        mock_judge.evaluate_all.side_effect = eval_outputs
        mock_from_file.return_value = mock_judge

        wf = CriticOptimizerWorkflow(
            original_agent_path=agent_path,
            eval_set_path="eval_set.json",
            pass_threshold=0.70,
            max_iterations=3,
            mock_mode=True,
            iteration_logs_path=logs_path,
        )
        report = wf.run()

    assert report["status"] == "max_iterations_reached"
    assert report["iterations_executed"] == 3
    assert report["final_score"] == 0.68
    assert len(report["iterations"]) == 3

    # Verify required log fields on every iteration
    for idx, entry in enumerate(report["iterations"], 1):
        assert entry["iteration_number"] == idx
        assert "failed_test_ids" in entry
        assert "failure_categories" in entry
        assert "root_cause" in entry
        assert "config_before" in entry
        assert "config_after" in entry
        assert "fix_applied" in entry
        assert entry["score_before"] is not None
        assert entry["score_after"] is not None
        assert "remaining_failures" in entry

    # Original agent.json untouched, v1.0.1, v1.0.2, v1.0.3 created
    assert (tmp_path / "agent_v1.0.1.json").exists()
    assert (tmp_path / "agent_v1.0.2.json").exists()
    assert (tmp_path / "agent_v1.0.3.json").exists()
    assert json.loads(agent_path.read_text(encoding="utf-8"))["version"] == "1.0.0"

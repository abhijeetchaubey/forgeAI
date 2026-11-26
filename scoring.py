"""
ForgeAI Phase 6: Configurable Weighted Scoring System & Scorecard Generator
Combines deterministic verification results with LLM judge evaluations:
- Configurable weights across 5 core metrics:
  1. Factual correctness
  2. Grounding accuracy
  3. Expected facts coverage
  4. Refusal accuracy
  5. Citation correctness
- Strict refusal enforcement (automatic fail if unsupported query is answered).
- Anti-bloat / anti-confidence penalties: no credit awarded purely for length or confidence.
- PASS / FAIL determination with detailed diagnostics.
- Category-wise and overall scorecard computation.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, model_validator
import json
from pathlib import Path

from deterministic_checks import DeterministicResult
from src.utils.logger import logger


class ScoringWeights(BaseModel):
    """
    Configurable relative weights for the 5 evaluation dimensions.
    Standard answers distribute across all 5 dimensions.
    """
    factual_correctness: float = Field(default=0.25, ge=0.0, le=1.0)
    grounding_accuracy: float = Field(default=0.20, ge=0.0, le=1.0)
    expected_facts_coverage: float = Field(default=0.20, ge=0.0, le=1.0)
    refusal_accuracy: float = Field(default=0.20, ge=0.0, le=1.0)
    citation_correctness: float = Field(default=0.15, ge=0.0, le=1.0)

    def normalized(self, for_refusal: bool = False) -> Dict[str, float]:
        """
        Returns normalized weights summing to 1.0.
        For refusal/out-of-scope cases, weights are shifted to refusal and anti-hallucination.
        """
        if for_refusal:
            # When query should be refused: refusal accuracy is primary (70%),
            # grounding (anti-hallucination) is 15%, factual correctness is 15%
            return {
                "factual_correctness": 0.15,
                "grounding_accuracy": 0.15,
                "expected_facts_coverage": 0.0,
                "refusal_accuracy": 0.60,
                "citation_correctness": 0.10
            }
        
        total = (
            self.factual_correctness
            + self.grounding_accuracy
            + self.expected_facts_coverage
            + self.refusal_accuracy
            + self.citation_correctness
        )
        if total <= 0:
            return {
                "factual_correctness": 0.25,
                "grounding_accuracy": 0.20,
                "expected_facts_coverage": 0.20,
                "refusal_accuracy": 0.20,
                "citation_correctness": 0.15
            }
        return {
            "factual_correctness": self.factual_correctness / total,
            "grounding_accuracy": self.grounding_accuracy / total,
            "expected_facts_coverage": self.expected_facts_coverage / total,
            "refusal_accuracy": self.refusal_accuracy / total,
            "citation_correctness": self.citation_correctness / total,
        }


class ScoringConfig(BaseModel):
    """Configuration for evaluation scoring and pass criteria."""
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    pass_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Minimum composite score to PASS")
    strict_refusal_enforcement: bool = Field(default=True, description="Enforce automatic FAIL if refusal behavior is wrong")
    anti_bloat_enforcement: bool = Field(default=True, description="Apply penalties for length/confidence fluff")


class TestCaseMetricScores(BaseModel):
    """Evaluation score breakdown for a single test case."""
    __test__ = False
    factual_correctness: float = Field(description="Score for factual accuracy (0.0 - 1.0)")
    grounding_accuracy: float = Field(description="Score for document grounding (0.0 - 1.0)")
    expected_facts_coverage: float = Field(description="Score for covering expected facts (0.0 - 1.0)")
    refusal_accuracy: float = Field(description="Score for refusal compliance (0.0 - 1.0)")
    citation_correctness: float = Field(description="Score for citing correct sources and pages (0.0 - 1.0)")
    composite_score: float = Field(description="Final weighted composite score (0.0 - 1.0)")
    status: str = Field(description="'PASS' or 'FAIL'")
    verdict_reason: str = Field(description="Human-readable justification for the verdict")


class LLMEvalResult(BaseModel):
    """Result of independent Groq LLM judge evaluation."""
    factual_score: float = Field(default=1.0, ge=0.0, le=1.0)
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    refusal_score: float = Field(default=1.0, ge=0.0, le=1.0)
    hallucination_detected: bool = Field(default=False)
    critique: str = Field(default="No critique provided.")
    raw_response: Optional[str] = Field(default=None)


class TestCaseResult(BaseModel):
    """Comprehensive evaluation record for a single test case."""
    __test__ = False  # Prevent pytest from identifying this class as a test suite
    test_id: str
    question: str
    category: str
    generated_answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    is_refusal: bool = False
    expected_facts: List[str] = Field(default_factory=list)
    expected_sources: List[str] = Field(default_factory=list)
    expected_page_number: Optional[str] = None
    should_refuse: bool = False
    evaluation_criteria: str = ""
    deterministic_checks: DeterministicResult
    llm_eval: LLMEvalResult
    scores: TestCaseMetricScores
    passed: bool
    failure_reason: Optional[str] = None


class CategoryScorecard(BaseModel):
    """Aggregated evaluation metrics for a single question category."""
    category: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    avg_composite_score: float
    avg_factual_correctness: float
    avg_grounding_accuracy: float
    avg_expected_facts_coverage: float
    avg_refusal_accuracy: float
    avg_citation_correctness: float


class OverallScorecard(BaseModel):
    """Aggregated evaluation scorecard across the entire test suite."""
    total_test_cases: int
    total_passed: int
    total_failed: int
    overall_pass_rate: float
    overall_composite_score: float
    avg_metrics: Dict[str, float]
    category_scorecards: Dict[str, CategoryScorecard]
    category_scores: Dict[str, float]
    failed_test_ids: List[str]
    failed_test_details: List[Dict[str, Any]]


def calculate_test_case_scores(
    deterministic: DeterministicResult,
    llm_eval: LLMEvalResult,
    category: str,
    should_refuse: bool,
    config: Optional[ScoringConfig] = None
) -> TestCaseMetricScores:
    """
    Blends deterministic checks and LLM judge assessments using configurable weights.
    Applies strict refusal enforcement and anti-bloat/confidence penalties.
    """
    config = config or ScoringConfig()
    w = config.weights.normalized(for_refusal=should_refuse)

    # 1. Refusal Accuracy:
    # 70% deterministic check + 30% LLM judgment
    refusal_acc = round(
        0.70 * deterministic.refusal_score + 0.30 * llm_eval.refusal_score,
        4
    )

    # 2. Factual Correctness:
    # Blend LLM evaluation with deterministic fact presence
    if should_refuse:
        # Refusal is factually correct if properly declined
        factual_corr = 1.0 if deterministic.refusal_passed else 0.0
    else:
        # Combine LLM factual score (50%) and deterministic fact coverage (50%)
        factual_corr = round(
            0.50 * llm_eval.factual_score + 0.50 * deterministic.facts_coverage_score,
            4
        )
        if llm_eval.hallucination_detected:
            factual_corr = max(0.0, factual_corr - 0.25)

    # 3. Grounding Accuracy:
    # Combine deterministic lexical grounding (50%) and LLM grounding score (50%)
    if should_refuse:
        grounding_acc = 1.0 if not deterministic.cited_sources else 0.5
    else:
        grounding_acc = round(
            0.50 * deterministic.grounding_score + 0.50 * llm_eval.grounding_score,
            4
        )

    # 4. Expected Facts Coverage:
    facts_cov = round(deterministic.facts_coverage_score, 4)

    # 5. Citation Correctness:
    cit_corr = round(deterministic.citation_score, 4)

    # 6. Base Composite Score
    raw_composite = (
        w["factual_correctness"] * factual_corr
        + w["grounding_accuracy"] * grounding_acc
        + w["expected_facts_coverage"] * facts_cov
        + w["refusal_accuracy"] * refusal_acc
        + w["citation_correctness"] * cit_corr
    )

    # 7. Deduct Anti-Bloat and Confidence Penalties
    if config.anti_bloat_enforcement:
        total_penalty = deterministic.verbosity_penalty + deterministic.confidence_penalty
        final_composite = max(0.0, round(raw_composite - total_penalty, 4))
    else:
        final_composite = round(raw_composite, 4)

    # 8. Determine PASS / FAIL and Failure Reasons
    reasons: List[str] = []
    auto_fail = False

    # Strict Refusal Enforcement Rule
    if config.strict_refusal_enforcement:
        if should_refuse and not deterministic.refusal_passed:
            auto_fail = True
            reasons.append("STRICT REFUSAL VIOLATION: Failed to refuse an out-of-scope query.")
        elif not should_refuse and category != "ambiguous" and deterministic.is_actual_refusal:
            auto_fail = True
            reasons.append("UNJUSTIFIED REFUSAL: Refused an answerable query.")

    if auto_fail:
        status = "FAIL"
        final_composite = min(final_composite, 0.30)
    elif final_composite >= config.pass_threshold:
        status = "PASS"
        reasons.append(f"Score {final_composite:.2f} meets pass threshold {config.pass_threshold:.2f}.")
    else:
        status = "FAIL"
        reasons.append(f"Score {final_composite:.2f} is below pass threshold {config.pass_threshold:.2f}.")

    if deterministic.facts_missing and not should_refuse:
        reasons.append(f"Missing facts: {', '.join(deterministic.facts_missing[:2])}")
    if deterministic.missing_sources and not should_refuse:
        reasons.append(f"Missing citations: {', '.join(deterministic.missing_sources)}")
    if deterministic.verbosity_penalty > 0:
        reasons.append(f"Verbosity penalty applied: -{deterministic.verbosity_penalty:.2f}")
    if deterministic.confidence_penalty > 0:
        reasons.append(f"Confidence fluff penalty applied: -{deterministic.confidence_penalty:.2f}")

    verdict_reason = " | ".join(reasons)

    return TestCaseMetricScores(
        factual_correctness=factual_corr,
        grounding_accuracy=grounding_acc,
        expected_facts_coverage=facts_cov,
        refusal_accuracy=refusal_acc,
        citation_correctness=cit_corr,
        composite_score=final_composite,
        status=status,
        verdict_reason=verdict_reason
    )


def generate_scorecards(
    test_results: List[TestCaseResult]
) -> Tuple[Dict[str, CategoryScorecard], OverallScorecard]:
    """
    Aggregates test results into category-wise scorecards and overall scorecard.
    """
    if not test_results:
        empty_overall = OverallScorecard(
            total_test_cases=0,
            total_passed=0,
            total_failed=0,
            overall_pass_rate=0.0,
            overall_composite_score=0.0,
            avg_metrics={},
            category_scorecards={},
            category_scores={},
            failed_test_ids=[],
            failed_test_details=[]
        )
        return {}, empty_overall

    # Group by category
    by_category: Dict[str, List[TestCaseResult]] = {}
    for r in test_results:
        by_category.setdefault(r.category, []).append(r)

    category_scorecards: Dict[str, CategoryScorecard] = {}
    category_scores: Dict[str, float] = {}

    for cat_name, cases in by_category.items():
        total = len(cases)
        passed = sum(1 for c in cases if c.passed)
        failed = total - passed
        pass_rate = round(passed / total, 4) if total > 0 else 0.0

        avg_comp = round(sum(c.scores.composite_score for c in cases) / total, 4)
        avg_fact = round(sum(c.scores.factual_correctness for c in cases) / total, 4)
        avg_ground = round(sum(c.scores.grounding_accuracy for c in cases) / total, 4)
        avg_facts_cov = round(sum(c.scores.expected_facts_coverage for c in cases) / total, 4)
        avg_refusal = round(sum(c.scores.refusal_accuracy for c in cases) / total, 4)
        avg_cit = round(sum(c.scores.citation_correctness for c in cases) / total, 4)

        cat_sc = CategoryScorecard(
            category=cat_name,
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            pass_rate=pass_rate,
            avg_composite_score=avg_comp,
            avg_factual_correctness=avg_fact,
            avg_grounding_accuracy=avg_ground,
            avg_expected_facts_coverage=avg_facts_cov,
            avg_refusal_accuracy=avg_refusal,
            avg_citation_correctness=avg_cit
        )
        category_scorecards[cat_name] = cat_sc
        category_scores[cat_name] = avg_comp

    # Overall Metrics
    total_cases = len(test_results)
    total_passed = sum(1 for c in test_results if c.passed)
    total_failed = total_cases - total_passed
    overall_pass_rate = round(total_passed / total_cases, 4) if total_cases > 0 else 0.0

    avg_overall_comp = round(sum(c.scores.composite_score for c in test_results) / total_cases, 4)
    avg_metrics = {
        "factual_correctness": round(sum(c.scores.factual_correctness for c in test_results) / total_cases, 4),
        "grounding_accuracy": round(sum(c.scores.grounding_accuracy for c in test_results) / total_cases, 4),
        "expected_facts_coverage": round(sum(c.scores.expected_facts_coverage for c in test_results) / total_cases, 4),
        "refusal_accuracy": round(sum(c.scores.refusal_accuracy for c in test_results) / total_cases, 4),
        "citation_correctness": round(sum(c.scores.citation_correctness for c in test_results) / total_cases, 4),
    }

    failed_cases = [c for c in test_results if not c.passed]
    failed_test_ids = [c.test_id for c in failed_cases]
    failed_test_details = [
        {
            "test_id": c.test_id,
            "category": c.category,
            "question": c.question,
            "generated_answer": c.generated_answer,
            "composite_score": c.scores.composite_score,
            "failure_reason": c.scores.verdict_reason,
            "missing_facts": c.deterministic_checks.facts_missing,
            "missing_sources": c.deterministic_checks.missing_sources,
        }
        for c in failed_cases
    ]

    overall_scorecard = OverallScorecard(
        total_test_cases=total_cases,
        total_passed=total_passed,
        total_failed=total_failed,
        overall_pass_rate=overall_pass_rate,
        overall_composite_score=avg_overall_comp,
        avg_metrics=avg_metrics,
        category_scorecards=category_scorecards,
        category_scores=category_scores,
        failed_test_ids=failed_test_ids,
        failed_test_details=failed_test_details
    )

    return category_scorecards, overall_scorecard


def save_evaluation_artifacts(
    test_results: List[TestCaseResult],
    overall_scorecard: OverallScorecard,
    category_scorecards: Dict[str, CategoryScorecard],
    evaluation_results_path: Union[str, Path] = "evaluation_results.json",
    scorecard_path: Union[str, Path] = "scorecard.json"
) -> Tuple[Path, Path]:
    """
    Saves evaluation_results.json and scorecard.json to disk.
    """
    results_file = Path(evaluation_results_path)
    scorecard_file = Path(scorecard_path)

    # 1. evaluation_results.json
    results_payload = {
        "summary": {
            "total_test_cases": overall_scorecard.total_test_cases,
            "total_passed": overall_scorecard.total_passed,
            "total_failed": overall_scorecard.total_failed,
            "overall_pass_rate": overall_scorecard.overall_pass_rate,
            "overall_composite_score": overall_scorecard.overall_composite_score,
        },
        "individual_results": [r.model_dump() for r in test_results],
        "failed_test_details": overall_scorecard.failed_test_details
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info("Saved detailed evaluation results to %s", results_file)

    # 2. scorecard.json
    scorecard_payload = {
        "overall_evaluation": {
            "overall_composite_score": overall_scorecard.overall_composite_score,
            "overall_pass_rate": overall_scorecard.overall_pass_rate,
            "total_test_cases": overall_scorecard.total_test_cases,
            "total_passed": overall_scorecard.total_passed,
            "total_failed": overall_scorecard.total_failed,
            "metric_averages": overall_scorecard.avg_metrics
        },
        "category_scorecards": {
            k: v.model_dump() for k, v in category_scorecards.items()
        },
        "category_scores": overall_scorecard.category_scores,
        "failed_test_ids": overall_scorecard.failed_test_ids,
        "failed_test_count": len(overall_scorecard.failed_test_ids)
    }
    with open(scorecard_file, "w", encoding="utf-8") as f:
        json.dump(scorecard_payload, f, indent=2)
    logger.info("Saved scorecard to %s", scorecard_file)

    return results_file, scorecard_file

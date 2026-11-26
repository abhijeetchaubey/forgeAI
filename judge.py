"""
ForgeAI Phase 6: Automated Judge & Evaluation System
Orchestrates end-to-end evaluation of generated agents against eval_set.json:
1. Loads evaluation test cases.
2. Executes every test case through the generic AgentRuntime.
3. Captures generated answer, refusal flags, and source citations.
4. Executes deterministic checks (factual coverage, source citation, refusal verification, anti-bloat).
5. Uses Groq LLM as an independent, objective evaluation judge.
6. Combines deterministic checks and LLM evaluations via configurable weighted scoring.
7. Assigns PASS or FAIL for every test case.
8. Calculates category-wise and overall scorecard metrics.
9. Exports evaluation_results.json and scorecard.json.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
import json
from pathlib import Path
import re
import argparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from config import settings
from src.llm.groq_client import get_groq_llm
from src.utils.logger import logger
from eval_generator import EvaluationSet, TestCase, EvalCategory
from chat_runner import AgentRuntime, AgentResponse
from deterministic_checks import run_deterministic_checks, DeterministicResult
from scoring import (
    ScoringConfig,
    ScoringWeights,
    TestCaseMetricScores,
    LLMEvalResult,
    TestCaseResult,
    CategoryScorecard,
    OverallScorecard,
    calculate_test_case_scores,
    generate_scorecards,
    save_evaluation_artifacts
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
console = Console()


def load_prompt_templates() -> Tuple[str, str]:
    """Loads system and evaluation prompt templates for the LLM judge."""
    sys_file = PROMPTS_DIR / "judge_system_prompt.txt"
    eval_file = PROMPTS_DIR / "judge_evaluation_prompt.txt"

    if sys_file.exists():
        sys_prompt = sys_file.read_text(encoding="utf-8")
    else:
        sys_prompt = (
            "You are the ForgeAI Independent Evaluation Judge.\n"
            "Evaluate whether the response is factually accurate, grounded, and properly adheres to refusal policies."
        )

    if eval_file.exists():
        eval_prompt = eval_file.read_text(encoding="utf-8")
    else:
        eval_prompt = (
            "TEST CASE: {question}\nEXPECTED: {expected_facts}\n"
            "ANSWER: {generated_answer}\nCITATIONS: {citations}\n"
            "Evaluate factual correctness, grounding, and refusal compliance in JSON."
        )

    return sys_prompt, eval_prompt


class LLMJudge:
    """Independent evaluation judge powered by ChatGroq."""

    def __init__(self, llm: Optional[ChatGroq] = None, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.sys_template, self.eval_template = load_prompt_templates()
        if not mock_mode:
            try:
                # Use zero temperature for deterministic, consistent evaluations
                self.llm = llm or get_groq_llm(temperature=0.0)
            except Exception as e:
                logger.warning("Could not initialize Groq client for LLMJudge (%s). Falling back to mock/deterministic mode.", str(e))
                self.llm = None
                self.mock_mode = True
        else:
            self.llm = None

    def evaluate(
        self,
        test_case: TestCase,
        answer: str,
        citations: List[Any],
        is_refusal: bool,
        deterministic: Optional[DeterministicResult] = None
    ) -> LLMEvalResult:
        """
        Submits agent response and test criteria to Groq LLM for independent grading.
        """
        # If in mock mode or LLM is unavailable, infer judgment from deterministic checks
        if self.mock_mode or not self.llm:
            return self._fallback_evaluation(test_case, deterministic)

        # Format prompt variables
        citations_summary = [
            f"{getattr(c, 'source', None) or c.get('source', 'unknown')} (p. {getattr(c, 'page_number', None) or c.get('page_number', '1')})"
            for c in citations
        ] if citations else ["None"]

        user_content = self.eval_template.format(
            test_id=test_case.test_id,
            category=test_case.category.value if hasattr(test_case.category, "value") else str(test_case.category),
            question=test_case.question,
            should_refuse="TRUE (Must refuse)" if test_case.should_refuse else "FALSE (Must answer)",
            expected_facts="\n".join(f"- {f}" for f in test_case.expected_facts),
            expected_sources=", ".join(test_case.expected_source_document) or "None",
            expected_page=test_case.expected_page_number or "N/A",
            evaluation_criteria=test_case.evaluation_criteria,
            generated_answer=answer,
            citations=", ".join(citations_summary),
            is_refusal="Yes" if is_refusal else "No"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.sys_template),
                HumanMessage(content=user_content)
            ])
            raw_text = response.content.strip()
            return self._parse_llm_response(raw_text, deterministic)
        except Exception as e:
            logger.error("LLMJudge API error for %s: %s. Using fallback evaluation.", test_case.test_id, str(e))
            return self._fallback_evaluation(test_case, deterministic)

    def _parse_llm_response(self, text: str, deterministic: Optional[DeterministicResult]) -> LLMEvalResult:
        """Parses and validates the JSON output returned by the LLM judge."""
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return LLMEvalResult(
                    factual_score=float(data.get("factual_score", 1.0)),
                    grounding_score=float(data.get("grounding_score", 1.0)),
                    refusal_score=float(data.get("refusal_score", 1.0)),
                    hallucination_detected=bool(data.get("hallucination_detected", False)),
                    critique=str(data.get("critique", "LLM evaluation parsed successfully.")),
                    raw_response=text
                )
        except Exception as parse_err:
            logger.warning("Failed to parse LLMJudge JSON response: %s", str(parse_err))

        return self._fallback_evaluation(None, deterministic)

    def _fallback_evaluation(
        self,
        test_case: Optional[TestCase],
        deterministic: Optional[DeterministicResult]
    ) -> LLMEvalResult:
        """Generates evaluation metrics from deterministic checks when LLM is unavailable."""
        if not deterministic:
            return LLMEvalResult(
                factual_score=1.0,
                grounding_score=1.0,
                refusal_score=1.0,
                hallucination_detected=False,
                critique="Default mock evaluation without deterministic data."
            )

        factual = deterministic.facts_coverage_score
        grounding = deterministic.grounding_score
        refusal = deterministic.refusal_score
        hallucination = (deterministic.confidence_penalty > 0 or (grounding < 0.3 and factual < 0.5))

        critique = (
            f"Deterministic evaluation: Facts coverage={factual:.2f}, "
            f"Grounding={grounding:.2f}, Refusal compliance={refusal:.2f}."
        )

        return LLMEvalResult(
            factual_score=factual,
            grounding_score=grounding,
            refusal_score=refusal,
            hallucination_detected=hallucination,
            critique=critique
        )


class EvaluationJudge:
    """
    Main evaluation coordinator that loads test cases, queries the agent runtime,
    executes deterministic checks and LLM judging, and produces scorecards.
    """

    def __init__(
        self,
        runtime: Optional[AgentRuntime] = None,
        llm_judge: Optional[LLMJudge] = None,
        scoring_config: Optional[ScoringConfig] = None,
        mock_judge: bool = False
    ):
        self.runtime = runtime
        if mock_judge and llm_judge is None:
            self.llm_judge = LLMJudge(mock_mode=True)
        else:
            self.llm_judge = llm_judge or LLMJudge()
        self.scoring_config = scoring_config or ScoringConfig()

    @classmethod
    def from_agent_file(
        cls,
        agent_path: Union[str, Path] = "agent.json",
        scoring_config: Optional[ScoringConfig] = None,
        mock_judge: bool = False
    ) -> "EvaluationJudge":
        """Factory initializing the runtime and judge from an agent configuration file."""
        runtime = AgentRuntime.from_agent_file(agent_path)
        judge = LLMJudge(mock_mode=mock_judge)
        return cls(runtime=runtime, llm_judge=judge, scoring_config=scoring_config)

    def evaluate_test_case(
        self,
        test_case: TestCase,
        precomputed_response: Optional[AgentResponse] = None
    ) -> TestCaseResult:
        """
        Evaluates a single test case either by querying the live agent runtime
        or by inspecting a supplied precomputed AgentResponse.
        """
        # 1. Obtain agent response
        if precomputed_response is not None:
            agent_resp = precomputed_response
        elif self.runtime is not None:
            agent_resp = self.runtime.ask(test_case.question)
        else:
            raise ValueError("Neither a live AgentRuntime nor a precomputed response was provided.")

        answer = agent_resp.answer
        is_refusal = agent_resp.is_refusal
        citations = agent_resp.citations

        # 2. Run deterministic checks
        cat_str = test_case.category.value if hasattr(test_case.category, "value") else str(test_case.category)
        deterministic = run_deterministic_checks(
            answer=answer,
            category=cat_str,
            expected_facts=test_case.expected_facts,
            expected_sources=test_case.expected_source_document,
            expected_page_number=test_case.expected_page_number,
            should_refuse=test_case.should_refuse,
            is_refusal_flag=is_refusal,
            citations=citations
        )

        # 3. Run independent LLM judge
        llm_eval = self.llm_judge.evaluate(
            test_case=test_case,
            answer=answer,
            citations=citations,
            is_refusal=is_refusal,
            deterministic=deterministic
        )

        # 4. Calculate combined weighted scores
        scores = calculate_test_case_scores(
            deterministic=deterministic,
            llm_eval=llm_eval,
            category=cat_str,
            should_refuse=test_case.should_refuse,
            config=self.scoring_config
        )

        # Convert citations to dictionaries for clean JSON serialization
        serialized_citations = []
        for c in citations:
            if hasattr(c, "model_dump"):
                serialized_citations.append(c.model_dump())
            elif isinstance(c, dict):
                serialized_citations.append(c)
            else:
                serialized_citations.append({"source": str(c)})

        failure_reason = scores.verdict_reason if scores.status == "FAIL" else None

        return TestCaseResult(
            test_id=test_case.test_id,
            question=test_case.question,
            category=cat_str,
            generated_answer=answer,
            citations=serialized_citations,
            is_refusal=is_refusal,
            expected_facts=test_case.expected_facts,
            expected_sources=test_case.expected_source_document,
            expected_page_number=test_case.expected_page_number,
            should_refuse=test_case.should_refuse,
            evaluation_criteria=test_case.evaluation_criteria,
            deterministic_checks=deterministic,
            llm_eval=llm_eval,
            scores=scores,
            passed=(scores.status == "PASS"),
            failure_reason=failure_reason
        )

    def evaluate_all(
        self,
        eval_set: Union[EvaluationSet, str, Path] = "eval_set.json",
        output_results_path: Union[str, Path] = "evaluation_results.json",
        output_scorecard_path: Union[str, Path] = "scorecard.json"
    ) -> Tuple[List[TestCaseResult], OverallScorecard, Dict[str, CategoryScorecard]]:
        """
        Executes and grades every test case in the evaluation set.
        Saves evaluation_results.json and scorecard.json.
        """
        if isinstance(eval_set, (str, Path)):
            eval_set_obj = EvaluationSet.load_from_json(eval_set)
        else:
            eval_set_obj = eval_set

        total_cases = len(eval_set_obj.test_cases)
        results: List[TestCaseResult] = []

        console.print(Panel.fit(
            f"[bold cyan]ForgeAI Phase 6: Automated Judge & Evaluation Runner[/bold cyan]\n"
            f"[dim]Total Test Cases: {total_cases} across {len(eval_set_obj.category_counts)} categories[/dim]",
            border_style="cyan"
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            eval_task = progress.add_task("[yellow]Evaluating test cases...", total=total_cases)

            for tc in eval_set_obj.test_cases:
                progress.update(eval_task, description=f"[yellow]Evaluating {tc.test_id} ({tc.category.value if hasattr(tc.category, 'value') else tc.category})...")
                res = self.evaluate_test_case(tc)
                results.append(res)
                progress.advance(eval_task)

        # Generate scorecards
        category_scorecards, overall_scorecard = generate_scorecards(results)

        # Save artifacts to JSON
        save_evaluation_artifacts(
            test_results=results,
            overall_scorecard=overall_scorecard,
            category_scorecards=category_scorecards,
            evaluation_results_path=output_results_path,
            scorecard_path=output_scorecard_path
        )

        # Print visual evaluation report
        self._print_visual_report(overall_scorecard, category_scorecards, results)

        return results, overall_scorecard, category_scorecards

    def _print_visual_report(
        self,
        overall: OverallScorecard,
        categories: Dict[str, CategoryScorecard],
        results: List[TestCaseResult]
    ):
        """Displays rich console summary tables."""
        console.print("\n")

        # Category Breakdown Table
        cat_table = Table(title="[bold green]Evaluation Results by Category[/bold green]")
        cat_table.add_column("Category", style="cyan", justify="left")
        cat_table.add_column("Cases", justify="center")
        cat_table.add_column("Passed", justify="center", style="green")
        cat_table.add_column("Failed", justify="center", style="red")
        cat_table.add_column("Pass Rate", justify="center", style="bold yellow")
        cat_table.add_column("Avg Composite", justify="center")
        cat_table.add_column("Factual", justify="center")
        cat_table.add_column("Grounding", justify="center")
        cat_table.add_column("Facts Cov", justify="center")
        cat_table.add_column("Refusal", justify="center")
        cat_table.add_column("Citation", justify="center")

        for cat_name, sc in categories.items():
            cat_table.add_row(
                cat_name,
                str(sc.total_cases),
                str(sc.passed_cases),
                str(sc.failed_cases),
                f"{sc.pass_rate * 100:.1f}%",
                f"{sc.avg_composite_score:.2f}",
                f"{sc.avg_factual_correctness:.2f}",
                f"{sc.avg_grounding_accuracy:.2f}",
                f"{sc.avg_expected_facts_coverage:.2f}",
                f"{sc.avg_refusal_accuracy:.2f}",
                f"{sc.avg_citation_correctness:.2f}"
            )
        console.print(cat_table)

        # Overall Summary Panel
        pass_color = "bold green" if overall.overall_pass_rate >= 0.70 else "bold red"
        summary_panel = Panel(
            f"[bold white]Overall Evaluation Summary[/bold white]\n"
            f"Total Tests: [cyan]{overall.total_test_cases}[/cyan] | "
            f"Passed: [green]{overall.total_passed}[/green] | "
            f"Failed: [red]{overall.total_failed}[/red]\n"
            f"Overall Pass Rate: [{pass_color}]{overall.overall_pass_rate * 100:.1f}%[/{pass_color}]\n"
            f"Overall Composite Score: [bold yellow]{overall.overall_composite_score:.2f} / 1.00[/bold yellow]\n"
            f"Metrics: Factual={overall.avg_metrics.get('factual_correctness', 0):.2f} | "
            f"Grounding={overall.avg_metrics.get('grounding_accuracy', 0):.2f} | "
            f"Facts Coverage={overall.avg_metrics.get('expected_facts_coverage', 0):.2f} | "
            f"Refusal={overall.avg_metrics.get('refusal_accuracy', 0):.2f} | "
            f"Citation={overall.avg_metrics.get('citation_correctness', 0):.2f}",
            title="[bold yellow]ForgeAI Scorecard[/bold yellow]",
            border_style="yellow"
        )
        console.print(summary_panel)

        # Failed Test Cases Table (if any)
        if overall.failed_test_details:
            fail_table = Table(title="[bold red]Failed Test Cases Detail[/bold red]")
            fail_table.add_column("ID", style="bold red", justify="center")
            fail_table.add_column("Category", style="cyan")
            fail_table.add_column("Question", max_width=45)
            fail_table.add_column("Score", justify="center", style="yellow")
            fail_table.add_column("Failure Diagnosis", style="dim", max_width=50)

            for f in overall.failed_test_details:
                fail_table.add_row(
                    f["test_id"],
                    f["category"],
                    f["question"],
                    f"{f['composite_score']:.2f}",
                    f["failure_reason"]
                )
            console.print(fail_table)


def main():
    """CLI entrypoint for running Phase 6 evaluations."""
    parser = argparse.ArgumentParser(description="ForgeAI Phase 6: Automated Judge and Evaluation System")
    parser.add_argument("--eval-set", default="eval_set.json", help="Path to eval_set.json")
    parser.add_argument("--agent", default="agent.json", help="Path to agent.json specification")
    parser.add_argument("--results-out", default="evaluation_results.json", help="Output path for evaluation_results.json")
    parser.add_argument("--scorecard-out", default="scorecard.json", help="Output path for scorecard.json")
    parser.add_argument("--mock-llm-judge", action="store_true", help="Use deterministic mock judge instead of live Groq API")
    parser.add_argument("--pass-threshold", type=float, default=0.70, help="Pass threshold composite score (0.0 - 1.0)")
    args = parser.parse_args()

    config = ScoringConfig(pass_threshold=args.pass_threshold)
    judge = EvaluationJudge.from_agent_file(
        agent_path=args.agent,
        scoring_config=config,
        mock_judge=args.mock_llm_judge
    )

    judge.evaluate_all(
        eval_set=args.eval_set,
        output_results_path=args.results_out,
        output_scorecard_path=args.scorecard_out
    )


if __name__ == "__main__":
    main()

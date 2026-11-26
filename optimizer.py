"""
ForgeAI Phase 7: Critic-Optimizer Loop using LangGraph
Implements a stateful, bounded (max 3 iterations) self-improvement loop:
1. Evaluates the current agent against eval_set.json using Phase 6 EvaluationJudge.
2. Routes conditionally via LangGraph:
   - If composite score >= pass_threshold (or 0 failures): STOP (passed).
   - If max_iterations reached: STOP (honest report).
   - Otherwise: route to Critic -> Optimizer -> Re-evaluate.
3. Critic classifies failures (Retrieval miss, Prompt defect, Hallucination, Wrong refusal)
   and diagnoses root causes.
4. Optimizer applies ONLY targeted corrections to relevant configuration sections,
   preserves the original agent.json, and saves a new versioned agent file.
5. Logs every iteration (IDs, categories, root causes, config before/after, fixes,
   score before/after, remaining failures) to iteration_logs.json.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
from typing import TypedDict, List, Dict, Any, Optional, Tuple, Union

from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from generator import AgentDefinition
from judge import EvaluationJudge
from scoring import ScoringConfig
from critic import CriticAgent, CriticDiagnosis, FailureCategory
from src.utils.logger import logger

console = Console()


# ---------------------------------------------------------------------------
# LangGraph State Definition
# ---------------------------------------------------------------------------

class OptimizationState(TypedDict):
    """State passed across LangGraph nodes in the Critic-Optimizer loop."""
    iteration: int
    max_iterations: int
    pass_threshold: float
    original_agent_path: str
    current_agent_path: str
    eval_set_path: str
    mock_mode: bool

    # Evaluation metrics
    current_score: float
    previous_score: float
    current_pass_rate: float
    failed_test_ids: List[str]
    eval_results_dicts: List[Dict[str, Any]]

    # Critic & Optimizer tracking
    diagnosis: Optional[Dict[str, Any]]
    iteration_logs: List[Dict[str, Any]]

    # Termination state
    status: str  # "running", "passed", "max_iterations_reached"
    stop_reason: str


# ---------------------------------------------------------------------------
# Version & Targeted Patching Helpers
# ---------------------------------------------------------------------------

def increment_version(version_str: str) -> str:
    """Increments patch version (e.g., '1.0.0' -> '1.0.1')."""
    parts = version_str.strip().split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{major}.{minor}.{patch + 1}"
    return f"{version_str}.1"


def build_versioned_agent_path(original_path: Union[str, Path], new_version: str) -> Path:
    """Generates a non-destructive versioned filename (preserving original agent.json)."""
    orig = Path(original_path)
    stem = re.sub(r"_v\d+\.\d+\.\d+$", "", orig.stem)
    return orig.parent / f"{stem}_v{new_version}{orig.suffix}"


class AgentOptimizer:
    """
    Applies surgical, failure-driven configuration patches to an AgentDefinition
    without regenerating the entire agent or modifying unrelated settings.
    """

    @staticmethod
    def apply_targeted_fixes(
        agent_def: AgentDefinition,
        diagnosis: CriticDiagnosis
    ) -> Tuple[AgentDefinition, Dict[str, Any], Dict[str, Any], List[str]]:
        """
        Applies only the fixes warranted by diagnosed failure categories.
        Returns: (updated_agent_def, config_before, config_after, fixes_applied)
        """
        updated = deepcopy(agent_def)
        config_before: Dict[str, Any] = {}
        config_after: Dict[str, Any] = {}
        fixes_applied: List[str] = []

        diagnosed_cats = set(diagnosis.failure_category_counts.keys())
        evidences_text = " ".join(c.evidence for c in diagnosis.classifications)

        # 1. Fix RETRIEVAL_MISS -> modify ONLY retrieval_config (and citation constraint if needed)
        if FailureCategory.RETRIEVAL_MISS.value in diagnosed_cats:
            before_ret = updated.retrieval_config.model_dump()
            new_top_k = min(updated.retrieval_config.top_k + 2, 10)
            new_thresh = round(min(updated.retrieval_config.similarity_threshold + 0.10, 1.65), 2)

            if new_top_k != updated.retrieval_config.top_k or new_thresh != updated.retrieval_config.similarity_threshold:
                config_before["retrieval_config"] = before_ret
                updated.retrieval_config.top_k = new_top_k
                updated.retrieval_config.similarity_threshold = new_thresh
                config_after["retrieval_config"] = updated.retrieval_config.model_dump()
                fixes_applied.append(
                    f"retrieval_config: increased top_k {before_ret['top_k']}->{new_top_k} and "
                    f"similarity_threshold {before_ret['similarity_threshold']}->{new_thresh} to capture cross-document chunks."
                )

        # 2. Fix WRONG_REFUSAL -> modify ONLY refusal_policy and refusal section of system_prompt
        if FailureCategory.WRONG_REFUSAL.value in diagnosed_cats:
            config_before["refusal_policy"] = updated.refusal_policy.model_dump()
            if "system_prompt" not in config_before:
                config_before["system_prompt"] = updated.system_prompt

            strict_rule = (
                "STRICT OUT-OF-SCOPE REFUSAL: Before answering, verify that the specific subject, metric, "
                "or attribute requested by the user is explicitly present in DOCUMENT EVIDENCE. "
                "If the exact requested information (e.g., warranty, cost, employee count, network diagram, "
                "software version control) is NOT in DOCUMENT EVIDENCE, you MUST refuse immediately using: "
                f"\"{updated.refusal_policy.default_refusal_response}\". Do NOT extrapolate from unrelated chunks."
            )
            if strict_rule not in updated.refusal_policy.unsupported_topics_handling:
                updated.refusal_policy.unsupported_topics_handling = strict_rule
            if "Refuse immediately if the specific requested attribute or topic is absent from DOCUMENT EVIDENCE." not in updated.refusal_policy.refusal_rules:
                updated.refusal_policy.refusal_rules.append(
                    "Refuse immediately if the specific requested attribute or topic is absent from DOCUMENT EVIDENCE."
                )

            if "STRICT OUT-OF-SCOPE REFUSAL" not in updated.system_prompt:
                updated.system_prompt = updated.system_prompt.rstrip() + f"\n\n{strict_rule}"

            config_after["refusal_policy"] = updated.refusal_policy.model_dump()
            config_after["system_prompt"] = updated.system_prompt
            fixes_applied.append(
                "refusal_policy & system_prompt: added strict evidence-presence verification before answering out-of-scope queries."
            )

        # 3. Fix PROMPT_DEFECT -> modify ONLY system_prompt / output_contract
        if FailureCategory.PROMPT_DEFECT.value in diagnosed_cats:
            if "system_prompt" not in config_before:
                config_before["system_prompt"] = updated.system_prompt
            if "output_contract" not in config_before:
                config_before["output_contract"] = updated.output_contract.model_dump()

            # Fix 3a: Remove raw tool-calling directives if tool_use_failed occurred
            if "tool_use_failed" in evidences_text or "Tool choice is none" in evidences_text:
                # Neutralize instructions that cause the model to emit JSON tool calls
                cleaned_prompt = re.sub(
                    r"- When a critical condition is detected.*?incident_ticket_creator`?\.",
                    "- Note any critical conditions in plain text based on DOCUMENT EVIDENCE (do NOT emit JSON tool calls).",
                    updated.system_prompt,
                    flags=re.DOTALL
                )
                cleaned_prompt = cleaned_prompt.replace(
                    "Retrieve relevant passages with `document_retriever` and exact telemetry rows with `tabular_data_inspector`.",
                    "Use the provided DOCUMENT EVIDENCE directly."
                )
                cleaned_prompt = cleaned_prompt.replace(
                    "Perform any threshold checks or unit conversions with `engineering_calculator`.",
                    "Show any threshold comparisons or unit calculations explicitly in text."
                )
                if "Do NOT output JSON tool calls" not in cleaned_prompt:
                    cleaned_prompt += "\n- IMPORTANT: Answer strictly in natural language text. Do NOT output JSON tool calls or function call syntax."
                updated.system_prompt = cleaned_prompt
                fixes_applied.append(
                    "system_prompt: removed raw tool-invocation directives that caused Groq tool_use_failed 400 errors."
                )

            # Fix 3b: Add ambiguous query & exact fact/citation guidance if missing
            prompt_addendum = (
                "\n\n**Precision & Ambiguity Handling**\n"
                "- State exact numbers, units (e.g., Celsius, bar, days), and component names verbatim from DOCUMENT EVIDENCE.\n"
                "- Cite every source document used as `[source: <filename>, page: <page_number>]`.\n"
                "- If a query is ambiguous or lacks required context (e.g., unspecified component ID, missing current/last maintenance date, "
                "missing time range, or absent logs), explicitly state what is missing and ask the user for clarification while citing the relevant document(s)."
            )
            if "Precision & Ambiguity Handling" not in updated.system_prompt:
                updated.system_prompt = updated.system_prompt.rstrip() + prompt_addendum
                fixes_applied.append(
                    "system_prompt: added explicit precision, verbatim unit/number preservation, and ambiguous-query clarification guidelines."
                )

            citation_constraint = "Every answer using documents must explicitly include [source: <filename>, page: <page_number>] for each referenced file."
            if citation_constraint not in updated.output_contract.constraints:
                updated.output_contract.constraints.append(citation_constraint)
                fixes_applied.append(
                    "output_contract: added explicit source+page citation constraint."
                )

            config_after["system_prompt"] = updated.system_prompt
            config_after["output_contract"] = updated.output_contract.model_dump()

        # 4. Fix HALLUCINATION -> modify ONLY output_contract & grounding constraint in system_prompt
        if FailureCategory.HALLUCINATION.value in diagnosed_cats:
            if "system_prompt" not in config_before:
                config_before["system_prompt"] = updated.system_prompt
            if "output_contract" not in config_before:
                config_before["output_contract"] = updated.output_contract.model_dump()

            anti_hallucination = (
                "\n- ZERO HALLUCINATION RULE: Never invent numbers, dates, logs, or technical claims not directly quoted or supported in DOCUMENT EVIDENCE."
            )
            if "ZERO HALLUCINATION RULE" not in updated.system_prompt:
                updated.system_prompt = updated.system_prompt.rstrip() + anti_hallucination
            no_speculation = "Do not speculate or extrapolate beyond verbatim facts in DOCUMENT EVIDENCE."
            if no_speculation not in updated.output_contract.constraints:
                updated.output_contract.constraints.append(no_speculation)

            config_after["system_prompt"] = updated.system_prompt
            config_after["output_contract"] = updated.output_contract.model_dump()
            fixes_applied.append(
                "system_prompt & output_contract: reinforced strict zero-hallucination and verbatim grounding constraints."
            )

        # Increment version & update timestamp
        updated.version = increment_version(agent_def.version)
        updated.created_at = datetime.utcnow().isoformat()

        return updated, config_before, config_after, fixes_applied


# ---------------------------------------------------------------------------
# LangGraph Workflow Definition
# ---------------------------------------------------------------------------

class CriticOptimizerWorkflow:
    """
    LangGraph-orchestrated Critic-Optimizer Loop.
    Enforces a maximum of 3 optimization iterations with real evaluation runs.
    """

    def __init__(
        self,
        original_agent_path: Union[str, Path] = "agent.json",
        eval_set_path: Union[str, Path] = "eval_set.json",
        pass_threshold: float = 0.70,
        max_iterations: int = 3,
        mock_mode: bool = False,
        iteration_logs_path: Union[str, Path] = "iteration_logs.json",
        scoring_config: Optional[ScoringConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.original_agent_path = Path(original_agent_path)
        self.eval_set_path = Path(eval_set_path)
        self.pass_threshold = pass_threshold
        self.max_iterations = min(max_iterations, 3)  # Enforce max 3 iterations
        self.mock_mode = mock_mode
        self.output_dir = Path(output_dir) if output_dir else self.original_agent_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_logs_path = Path(iteration_logs_path)
        self.scoring_config = scoring_config or ScoringConfig(pass_threshold=pass_threshold)
        self.critic = CriticAgent(mock_mode=mock_mode)
        self.graph = self._build_graph()

    def _evaluate_node(self, state: OptimizationState) -> Dict[str, Any]:
        """Executes the complete Phase 6 evaluation suite on current_agent_path."""
        agent_path = state["current_agent_path"]
        iteration = state["iteration"]

        console.print(f"\n[bold cyan]>>> [LangGraph: evaluate_node] Running evaluation on {agent_path} (completed iterations: {iteration})[/bold cyan]")

        judge = EvaluationJudge.from_agent_file(
            agent_path=agent_path,
            scoring_config=self.scoring_config,
            mock_judge=state["mock_mode"]
        )
        res_name = f"evaluation_results_iter{iteration}.json" if iteration > 0 else "evaluation_results.json"
        sc_name = f"scorecard_iter{iteration}.json" if iteration > 0 else "scorecard.json"
        results, overall_sc, _ = judge.evaluate_all(
            eval_set=state["eval_set_path"],
            output_results_path=self.output_dir / res_name,
            output_scorecard_path=self.output_dir / sc_name,
        )

        new_score = overall_sc.overall_composite_score
        new_pass_rate = overall_sc.overall_pass_rate
        failed_ids = overall_sc.failed_test_ids
        results_dicts = [r.model_dump() for r in results]

        # If we just finished an optimization step (iteration > 0), close out the latest iteration log
        logs = deepcopy(state["iteration_logs"])
        if logs and logs[-1].get("score_after") is None:
            logs[-1]["score_after"] = new_score
            logs[-1]["pass_rate_after"] = new_pass_rate
            logs[-1]["score_delta"] = round(new_score - logs[-1]["score_before"], 4)
            logs[-1]["remaining_failures"] = failed_ids
            logs[-1]["remaining_failure_count"] = len(failed_ids)

        return {
            "previous_score": state["current_score"],
            "current_score": new_score,
            "current_pass_rate": new_pass_rate,
            "failed_test_ids": failed_ids,
            "eval_results_dicts": results_dicts,
            "iteration_logs": logs,
        }

    def _route_after_evaluate(self, state: OptimizationState) -> str:
        """LangGraph conditional routing after evaluation."""
        score = state["current_score"]
        threshold = state["pass_threshold"]
        iteration = state["iteration"]
        max_iter = state["max_iterations"]
        failed_ids = state["failed_test_ids"]

        if score >= threshold or len(failed_ids) == 0:
            logger.info("Evaluation PASSED threshold (%.4f >= %.4f). Stopping.", score, threshold)
            return "stop_passed"
        if iteration >= max_iter:
            logger.warning("Maximum iterations (%d) reached with score %.4f < %.4f. Stopping.", max_iter, score, threshold)
            return "stop_max_iter"
        logger.info("Evaluation below threshold (%.4f < %.4f). Routing to critic_node.", score, threshold)
        return "continue_critic"

    def _finalize_passed_node(self, state: OptimizationState) -> Dict[str, Any]:
        return {
            "status": "passed",
            "stop_reason": f"Evaluation passed configured threshold ({state['current_score']:.4f} >= {state['pass_threshold']:.4f})."
        }

    def _finalize_max_iter_node(self, state: OptimizationState) -> Dict[str, Any]:
        return {
            "status": "max_iterations_reached",
            "stop_reason": (
                f"Maximum optimization iterations ({state['max_iterations']}) reached. "
                f"Final score: {state['current_score']:.4f} (threshold: {state['pass_threshold']:.4f}), "
                f"Remaining failures: {len(state['failed_test_ids'])}."
            )
        }

    def _critic_node(self, state: OptimizationState) -> Dict[str, Any]:
        """Classifies every failure and diagnoses root causes."""
        next_iter = state["iteration"] + 1
        console.print(f"\n[bold yellow]>>> [LangGraph: critic_node] Diagnosing {len(state['failed_test_ids'])} failures (Iteration {next_iter})[/bold yellow]")

        agent_def = AgentDefinition.load_from_json(state["current_agent_path"])
        failed_results = [r for r in state["eval_results_dicts"] if not r.get("passed", False)]

        diagnosis = self.critic.diagnose(
            failed_results=failed_results,
            agent_def=agent_def,
            iteration=next_iter
        )
        return {"diagnosis": diagnosis.model_dump()}

    def _optimize_node(self, state: OptimizationState) -> Dict[str, Any]:
        """Applies targeted fixes, preserves original agent.json, and saves new version."""
        next_iter = state["iteration"] + 1
        console.print(f"\n[bold magenta]>>> [LangGraph: optimize_node] Applying targeted fixes (Iteration {next_iter})[/bold magenta]")

        current_path = Path(state["current_agent_path"])
        agent_def = AgentDefinition.load_from_json(current_path)
        diagnosis = CriticDiagnosis.model_validate(state["diagnosis"])

        updated_agent, cfg_before, cfg_after, fixes_applied = AgentOptimizer.apply_targeted_fixes(
            agent_def=agent_def,
            diagnosis=diagnosis
        )

        # Preserve original agent.json; write new versioned file
        new_path = build_versioned_agent_path(state["original_agent_path"], updated_agent.version)
        updated_agent.save_to_json(new_path)

        # Build iteration log entry (score_after & remaining_failures populated after re-evaluation)
        failure_categories_map = {
            c.test_id: c.failure_category.value
            for c in diagnosis.classifications
        }
        root_causes_map = {
            c.test_id: c.root_cause
            for c in diagnosis.classifications
        }

        log_entry = {
            "iteration_number": next_iter,
            "agent_version_before": agent_def.version,
            "agent_version_after": updated_agent.version,
            "agent_path_before": str(current_path),
            "agent_path_after": str(new_path),
            "failed_test_ids": diagnosis.failed_test_ids,
            "failure_categories": failure_categories_map,
            "failure_category_counts": diagnosis.failure_category_counts,
            "root_cause": diagnosis.overall_root_cause_summary,
            "root_causes_by_test": root_causes_map,
            "config_before": cfg_before,
            "config_after": cfg_after,
            "fix_applied": fixes_applied,
            "score_before": state["current_score"],
            "score_after": None,
            "pass_rate_before": state["current_pass_rate"],
            "pass_rate_after": None,
            "score_delta": None,
            "remaining_failures": None,
            "remaining_failure_count": None,
        }

        logs = deepcopy(state["iteration_logs"])
        logs.append(log_entry)

        return {
            "iteration": next_iter,
            "current_agent_path": str(new_path),
            "iteration_logs": logs,
        }

    def _build_graph(self):
        """Constructs and compiles the LangGraph StateGraph."""
        builder = StateGraph(OptimizationState)

        builder.add_node("evaluate", self._evaluate_node)
        builder.add_node("critic", self._critic_node)
        builder.add_node("optimize", self._optimize_node)
        builder.add_node("finalize_passed", self._finalize_passed_node)
        builder.add_node("finalize_max_iter", self._finalize_max_iter_node)

        builder.add_edge(START, "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            self._route_after_evaluate,
            {
                "stop_passed": "finalize_passed",
                "stop_max_iter": "finalize_max_iter",
                "continue_critic": "critic",
            }
        )
        builder.add_edge("critic", "optimize")
        builder.add_edge("optimize", "evaluate")
        builder.add_edge("finalize_passed", END)
        builder.add_edge("finalize_max_iter", END)

        return builder.compile()

    def run(self) -> Dict[str, Any]:
        """Executes the full LangGraph Critic-Optimizer loop and saves iteration_logs.json."""
        initial_state: OptimizationState = {
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "pass_threshold": self.pass_threshold,
            "original_agent_path": str(self.original_agent_path),
            "current_agent_path": str(self.original_agent_path),
            "eval_set_path": str(self.eval_set_path),
            "mock_mode": self.mock_mode,
            "current_score": 0.0,
            "previous_score": 0.0,
            "current_pass_rate": 0.0,
            "failed_test_ids": [],
            "eval_results_dicts": [],
            "diagnosis": None,
            "iteration_logs": [],
            "status": "running",
            "stop_reason": "",
        }

        final_state = self.graph.invoke(initial_state)

        report = {
            "status": final_state["status"],
            "stop_reason": final_state["stop_reason"],
            "original_agent_path": final_state["original_agent_path"],
            "final_agent_path": final_state["current_agent_path"],
            "pass_threshold": self.pass_threshold,
            "max_iterations": self.max_iterations,
            "iterations_executed": final_state["iteration"],
            "final_score": final_state["current_score"],
            "final_pass_rate": final_state["current_pass_rate"],
            "remaining_failures": final_state["failed_test_ids"],
            "iterations": final_state["iteration_logs"],
        }

        with open(self.iteration_logs_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved optimization iteration logs to %s", self.iteration_logs_path)

        self._print_honest_report(report)
        return report

    def _print_honest_report(self, report: Dict[str, Any]):
        """Prints a transparent, honest final optimization report."""
        console.print("\n")
        status_color = "green" if report["status"] == "passed" else "yellow"
        console.print(Panel(
            f"[bold {status_color}]ForgeAI Phase 7: Critic-Optimizer Loop Complete[/bold {status_color}]\n"
            f"Status: [bold]{report['status']}[/bold]\n"
            f"Reason: {report['stop_reason']}\n"
            f"Original Agent (Preserved): [cyan]{report['original_agent_path']}[/cyan]\n"
            f"Final Agent Version: [cyan]{report['final_agent_path']}[/cyan]\n"
            f"Iterations Executed: {report['iterations_executed']} / {report['max_iterations']}\n"
            f"Final Composite Score: [bold yellow]{report['final_score']:.4f}[/bold yellow] (Threshold: {report['pass_threshold']:.2f})\n"
            f"Final Pass Rate: [bold]{report['final_pass_rate'] * 100:.1f}%[/bold]\n"
            f"Remaining Failures ({len(report['remaining_failures'])}): {', '.join(report['remaining_failures']) or 'None'}",
            border_style=status_color
        ))

        if report["iterations"]:
            tbl = Table(title="Optimization Iteration History")
            tbl.add_column("Iter", justify="center", style="cyan")
            tbl.add_column("Version", justify="center")
            tbl.add_column("Score Before", justify="center")
            tbl.add_column("Score After", justify="center", style="bold yellow")
            tbl.add_column("Delta", justify="center")
            tbl.add_column("Failures Fixed", justify="center")
            tbl.add_column("Fixes Applied", max_width=55)

            for it in report["iterations"]:
                before_cnt = len(it["failed_test_ids"])
                after_cnt = it.get("remaining_failure_count", before_cnt)
                delta = it.get("score_delta", 0.0) or 0.0
                delta_str = f"{delta:+.4f}"
                tbl.add_row(
                    str(it["iteration_number"]),
                    f"{it['agent_version_before']} -> {it['agent_version_after']}",
                    f"{it['score_before']:.4f}",
                    f"{it['score_after']:.4f}" if it["score_after"] is not None else "N/A",
                    delta_str,
                    f"{before_cnt} -> {after_cnt}",
                    "; ".join(it["fix_applied"])
                )
            console.print(tbl)


def main():
    parser = argparse.ArgumentParser(description="ForgeAI Phase 7: Critic-Optimizer Loop")
    parser.add_argument("--agent", default="agent.json", help="Path to original agent.json")
    parser.add_argument("--eval-set", default="eval_set.json", help="Path to eval_set.json")
    parser.add_argument("--pass-threshold", type=float, default=0.70, help="Target composite score threshold")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum optimization iterations (max 3)")
    parser.add_argument("--logs-out", default="iteration_logs.json", help="Output path for iteration_logs.json")
    parser.add_argument("--mock-mode", action="store_true", help="Use deterministic critic/judge without live Groq API")
    args = parser.parse_args()

    workflow = CriticOptimizerWorkflow(
        original_agent_path=args.agent,
        eval_set_path=args.eval_set,
        pass_threshold=args.pass_threshold,
        max_iterations=args.max_iterations,
        mock_mode=args.mock_mode,
        iteration_logs_path=args.logs_out
    )
    workflow.run()


if __name__ == "__main__":
    main()

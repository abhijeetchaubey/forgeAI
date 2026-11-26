"""
ForgeAI Phase 9: Complete End-to-End Demonstrations Factory Runner
Executes two completely different domain demonstrations using the exact same
domain-independent factory pipeline and generic runtime:
  - Demo 1: HR Policy Assistant
  - Demo 2: Ship-Spare Procurement Assistant
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel

from create_demo_datasets import create_hr_policy_dataset, create_ship_spare_dataset
from src.document_loaders.loaders import load_directory
from src.text_splitter.splitter import split_documents
from src.embeddings.embeddings import get_embedding_model
from src.vectorstore.chroma_store import get_vector_store, add_documents
from intake import conduct_intake
from groundability import analyze_groundability
from generator import generate_agent
from eval_generator import generate_evaluation_set
from chat_runner import AgentRuntime
from judge import EvaluationJudge
from optimizer import CriticOptimizerWorkflow
from scoring import ScoringConfig

console = Console()


@dataclass
class DemoSpec:
    demo_id: str
    title: str
    brief: str
    data_dir: Path
    output_dir: Path
    collection_name: str
    unseen_questions: List[Dict[str, Any]] = field(default_factory=list)


def run_single_demo(
    spec: DemoSpec,
    pass_threshold: float = 0.70,
    max_iterations: int = 3,
    mock_mode: bool = False,
) -> Dict[str, Any]:
    """
    Executes the full 10-step ForgeAI factory pipeline for any domain demo
    without any domain-specific hardcoding.
    """
    console.print(Panel.fit(
        f"[bold cyan]ForgeAI Phase 9 Demo: {spec.title}[/bold cyan]\n"
        f"[dim]Data: {spec.data_dir} | Output: {spec.output_dir}[/dim]",
        border_style="cyan",
    ))

    # Prepare clean output directory and isolated ChromaDB
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = spec.output_dir / "chroma_db"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and ingest documents into isolated vector store
    raw_docs = load_directory(spec.data_dir)
    doc_metadata = []
    seen_files = set()
    for d in raw_docs:
        fname = d.metadata.get("filename", "unknown")
        if fname not in seen_files:
            seen_files.add(fname)
            doc_metadata.append({
                "filename": fname,
                "file_type": d.metadata.get("file_type", "unknown"),
                "source": d.metadata.get("source", ""),
            })

    chunks = split_documents(raw_docs)
    embeddings = get_embedding_model()
    vector_store = get_vector_store(
        persist_directory=chroma_dir,
        collection_name=spec.collection_name,
        embedding_model=embeddings,
    )
    add_documents(chunks, vector_store=vector_store)

    # 2. Intake Interviewer -> requirements.json
    req_path = spec.output_dir / "requirements.json"
    requirements = conduct_intake(
        brief=spec.brief,
        documents=raw_docs,
        document_metadata=doc_metadata,
        output_path=req_path,
    )

    # 3. Groundability Guard -> groundability_report.json
    ground_path = spec.output_dir / "groundability_report.json"
    groundability = analyze_groundability(
        requirements=requirements,
        vector_store=vector_store,
        output_path=ground_path,
    )

    # 4. Agent Generator -> agent.json (with isolated retrieval_config)
    agent_path = spec.output_dir / "agent.json"
    agent_def = generate_agent(
        requirements=requirements,
        groundability=groundability,
        document_metadata=doc_metadata,
        output_path=agent_path,
        collection_name=spec.collection_name,
        persist_directory=str(chroma_dir),
    )

    # 5. Evaluation Generator -> eval_set.json (>= 20 cases)
    eval_set_path = spec.output_dir / "eval_set.json"
    eval_set = generate_evaluation_set(
        documents=raw_docs,
        document_dir=spec.data_dir,
        vector_store=vector_store,
        output_path=eval_set_path,
        min_cases=20,
    )

    # 6 & 7 & 8. Judge + LangGraph Critic-Optimizer Loop (up to 3 iterations)
    iter_logs_path = spec.output_dir / "iteration_logs.json"
    workflow = CriticOptimizerWorkflow(
        original_agent_path=agent_path,
        eval_set_path=eval_set_path,
        pass_threshold=pass_threshold,
        max_iterations=max_iterations,
        mock_mode=mock_mode,
        iteration_logs_path=iter_logs_path,
        output_dir=spec.output_dir,
    )
    opt_report = workflow.run()

    # Load final scorecard
    final_agent_path = Path(opt_report["final_agent_path"])
    final_iter = opt_report["iterations_executed"]
    final_scorecard_file = (
        spec.output_dir / f"scorecard_iter{final_iter}.json"
        if final_iter > 0
        else spec.output_dir / "scorecard.json"
    )
    final_eval_results_file = (
        spec.output_dir / f"evaluation_results_iter{final_iter}.json"
        if final_iter > 0
        else spec.output_dir / "evaluation_results.json"
    )

    # Ensure canonical scorecard.json & evaluation_results.json reflect final run as well
    if final_iter > 0 and final_scorecard_file.exists():
        shutil.copyfile(final_scorecard_file, spec.output_dir / "scorecard.json")
    if final_iter > 0 and final_eval_results_file.exists():
        shutil.copyfile(final_eval_results_file, spec.output_dir / "evaluation_results.json")

    scorecard_data = json.loads((spec.output_dir / "scorecard.json").read_text(encoding="utf-8"))
    eval_results_data = json.loads((spec.output_dir / "evaluation_results.json").read_text(encoding="utf-8"))

    # 9. Test final agent with additional UNSEEN questions via generic AgentRuntime
    runtime = AgentRuntime.from_agent_file(final_agent_path)
    unseen_results = []
    for uq in spec.unseen_questions:
        resp = runtime.ask(uq["question"])
        unseen_results.append({
            "question": uq["question"],
            "expected_behavior": uq["expected_behavior"],
            "is_refusal": resp.is_refusal,
            "answer": resp.answer,
            "citations": [c.model_dump() for c in resp.citations],
            "tools_used": resp.tools_used,
        })

    unseen_path = spec.output_dir / "unseen_questions_results.json"
    unseen_path.write_text(json.dumps(unseen_results, indent=2), encoding="utf-8")

    # 10. Generate honest final_quality_report.md
    report_md = _build_quality_report(
        spec=spec,
        requirements=requirements.model_dump(),
        groundability=groundability.model_dump(),
        agent_def=agent_def.model_dump(),
        eval_count=eval_set.total_test_cases,
        opt_report=opt_report,
        scorecard=scorecard_data,
        failed_details=eval_results_data.get("failed_test_details", []),
        unseen_results=unseen_results,
    )
    report_path = spec.output_dir / "final_quality_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "demo_id": spec.demo_id,
        "output_dir": str(spec.output_dir),
        "final_score": opt_report["final_score"],
        "final_pass_rate": opt_report["final_pass_rate"],
        "status": opt_report["status"],
        "iterations_executed": opt_report["iterations_executed"],
        "remaining_failures": opt_report["remaining_failures"],
        "tools_selected": [t.name for t in agent_def.selected_tools],
    }


def _build_quality_report(
    spec: DemoSpec,
    requirements: Dict[str, Any],
    groundability: Dict[str, Any],
    agent_def: Dict[str, Any],
    eval_count: int,
    opt_report: Dict[str, Any],
    scorecard: Dict[str, Any],
    failed_details: List[Dict[str, Any]],
    unseen_results: List[Dict[str, Any]],
) -> str:
    overall = scorecard.get("overall_evaluation", {})
    cat_scores = scorecard.get("category_scores", {})
    tools = [t["name"] for t in agent_def.get("selected_tools", [])]

    lines = [
        f"# Final Quality Report: {spec.title}",
        "",
        f"- **Generated At:** {datetime.utcnow().isoformat()}",
        f"- **Brief:** {spec.brief}",
        f"- **Agent Name:** {agent_def.get('agent_name')}",
        f"- **Initial Agent Version:** {agent_def.get('version')} (`agent.json`)",
        f"- **Final Agent Artifact:** `{Path(opt_report['final_agent_path']).name}`",
        f"- **Selected Tools:** {', '.join(tools) if tools else 'None'}",
        f"- **Evaluation Cases:** {eval_count}",
        "",
        "## Groundability Summary",
        f"- **Guaranteed Capabilities:** {len(groundability.get('guaranteed_capabilities', []))}",
        f"- **Partially Supported Topics:** {len(groundability.get('partially_supported_topics', []))}",
        f"- **Blocked Unsupported Topics:** {len(groundability.get('blocked_unsupported_topics', []))}",
        "",
        "## Evaluation & Optimization Results",
        f"- **Optimization Status:** `{opt_report['status']}` ({opt_report['stop_reason']})",
        f"- **Iterations Executed:** {opt_report['iterations_executed']} / {opt_report['max_iterations']}",
        f"- **Final Composite Score:** **{overall.get('overall_composite_score', 0):.4f}**",
        f"- **Final Pass Rate:** **{overall.get('overall_pass_rate', 0) * 100:.1f}%** ({overall.get('total_passed', 0)}/{overall.get('total_test_cases', 0)} passed)",
        "",
        "### Category Scores",
    ]
    for cat, val in cat_scores.items():
        lines.append(f"- `{cat}`: {val:.4f}")

    lines.extend([
        "",
        "## Unseen Questions Verification",
    ])
    for idx, uq in enumerate(unseen_results, 1):
        lines.append(f"### Unseen Q{idx}: {uq['question']}")
        lines.append(f"- **Expected Behavior:** {uq['expected_behavior']}")
        lines.append(f"- **Refusal Triggered:** `{uq['is_refusal']}`")
        lines.append(f"- **Answer:** {uq['answer']}")
        lines.append("")

    lines.append("## Remaining Failures (Honest Documentation)")
    if not failed_details:
        lines.append("No remaining failures — all evaluation cases passed.")
    else:
        lines.append(f"Total remaining failures: **{len(failed_details)}**")
        lines.append("")
        for fd in failed_details:
            lines.append(f"- **{fd['test_id']}** (`{fd['category']}`, score `{fd['composite_score']:.2f}`): {fd['question']}")
            lines.append(f"  - Reason: {fd['failure_reason']}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="ForgeAI Phase 9: End-to-End Demonstrations")
    parser.add_argument("--pass-threshold", type=float, default=0.70)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--only", choices=["demo1", "demo2", "both"], default="both")
    args = parser.parse_args()

    hr_dir = create_hr_policy_dataset()
    ship_dir = create_ship_spare_dataset()

    demo1 = DemoSpec(
        demo_id="demo1_hr_policy",
        title="Demo 1: HR Policy Assistant",
        brief=(
            "Build an HR Policy Assistant for employees using our HR handbook PDF, leave policy, "
            "and employee benefits guide. It must answer questions on working hours, remote work, "
            "PTO, sick leave, parental leave, 401k matching, and stipends, and strictly refuse "
            "questions about individual salaries, executive stock options, or external legal advice."
        ),
        data_dir=hr_dir,
        output_dir=Path("outputs/demo1_hr_policy"),
        collection_name="demo1_hr_policy_docs",
        unseen_questions=[
            {
                "question": "If I take 4 consecutive sick days, do I need to submit anything?",
                "expected_behavior": "Answer yes, a licensed physician's medical certificate is required after 3 consecutive days (leave_policy.txt).",
            },
            {
                "question": "How much does the company contribute toward dependent medical insurance premiums?",
                "expected_behavior": "Answer 75% for eligible dependents (employee_benefits.txt).",
            },
            {
                "question": "What is the CEO's annual bonus structure?",
                "expected_behavior": "Refuse out-of-scope inquiry.",
            },
        ],
    )

    demo2 = DemoSpec(
        demo_id="demo2_ship_spare",
        title="Demo 2: Ship-Spare Procurement Assistant",
        brief=(
            "Build a Ship-Spare Procurement Assistant for marine procurement engineers using our "
            "product catalogue CSV, supplier directory PDFs, and spare-parts technical specifications. "
            "It must look up part pricing, lead times, minimum order quantities, supplier payment terms, "
            "and technical limits, calculate total order costs when needed, and strictly refuse "
            "questions about vessel crew schedules or unrelated consumer electronics."
        ),
        data_dir=ship_dir,
        output_dir=Path("outputs/demo2_ship_spare"),
        collection_name="demo2_ship_spare_docs",
        unseen_questions=[
            {
                "question": "What is the unit price and minimum order quantity for SP-103 Seawater Cooling Pump Impeller?",
                "expected_behavior": "Answer $320.00 USD and MOQ 4 from product_catalogue.csv.",
            },
            {
                "question": "Which supplier provides SP-102 and what are their payment terms?",
                "expected_behavior": "Identify SUP-02 (Pacific Propulsion Ltd) and Net 45 days from product_catalogue.csv and supplier_directory.pdf.",
            },
            {
                "question": "What is the captain's shore leave schedule for next month?",
                "expected_behavior": "Refuse out-of-scope inquiry.",
            },
        ],
    )

    summaries = []
    if args.only in ("demo1", "both"):
        summaries.append(run_single_demo(demo1, pass_threshold=args.pass_threshold, max_iterations=args.max_iterations))
    if args.only in ("demo2", "both"):
        summaries.append(run_single_demo(demo2, pass_threshold=args.pass_threshold, max_iterations=args.max_iterations))

    print("\n=== PHASE 9 DEMONSTRATION SUMMARY ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

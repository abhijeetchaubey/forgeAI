"""
ForgeAI Phase 7: Failure Diagnostic Critic
Analyzes failed evaluation test cases from Phase 6, classifies every failure into:
  - Retrieval miss
  - Prompt defect
  - Hallucination
  - Wrong refusal
Diagnoses root causes using evaluation evidence and recommends targeted,
surgical corrections to only the relevant agent configuration sections.
"""

from enum import Enum
import json
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from config import settings
from src.llm.groq_client import get_groq_llm
from src.utils.logger import logger
from generator import AgentDefinition

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class FailureCategory(str, Enum):
    """Strict 4-class taxonomy for agent evaluation failures."""
    RETRIEVAL_MISS = "Retrieval miss"
    PROMPT_DEFECT = "Prompt defect"
    HALLUCINATION = "Hallucination"
    WRONG_REFUSAL = "Wrong refusal"


class FailureClassification(BaseModel):
    """Diagnosis for a single failed test case."""
    test_id: str
    question: str
    test_category: str
    failure_category: FailureCategory
    evidence: str
    root_cause: str
    recommended_fix: str
    target_config_section: str = Field(
        description="One of: retrieval_config, system_prompt, refusal_policy, output_contract"
    )


class CriticDiagnosis(BaseModel):
    """Complete diagnostic report across all failures in an evaluation run."""
    iteration: int = 1
    total_failures: int
    failed_test_ids: List[str]
    failure_category_counts: Dict[str, int]
    classifications: List[FailureClassification]
    overall_root_cause_summary: str
    prioritized_recommendations: List[Dict[str, Any]] = Field(default_factory=list)


def load_critic_prompts() -> Tuple[str, str]:
    """Loads critic prompt templates from prompts/ directory."""
    sys_file = PROMPTS_DIR / "critic_system_prompt.txt"
    diag_file = PROMPTS_DIR / "critic_diagnosis_prompt.txt"

    sys_prompt = (
        sys_file.read_text(encoding="utf-8")
        if sys_file.exists()
        else "You are the ForgeAI Phase 7 Diagnostic Critic. Classify failures into Retrieval miss, Prompt defect, Hallucination, or Wrong refusal."
    )
    diag_prompt = (
        diag_file.read_text(encoding="utf-8")
        if diag_file.exists()
        else "Analyze failures: {failed_cases_summary}"
    )
    return sys_prompt, diag_prompt


def classify_failure_deterministically(
    failed_case: Dict[str, Any]
) -> FailureClassification:
    """
    Applies objective, evidence-driven rules to classify a failed test case
    into one of: Retrieval miss, Prompt defect, Hallucination, or Wrong refusal.
    """
    test_id = failed_case.get("test_id", "UNKNOWN")
    question = failed_case.get("question", "")
    test_cat = failed_case.get("category", "answerable")
    answer = failed_case.get("generated_answer", "")
    should_refuse = bool(failed_case.get("should_refuse", False))
    is_refusal = bool(failed_case.get("is_refusal", False))
    citations = failed_case.get("citations", [])

    det = failed_case.get("deterministic_checks", {})
    llm_eval = failed_case.get("llm_eval", {})
    scores = failed_case.get("scores", {})

    verdict_reason = scores.get("verdict_reason", "") or failed_case.get("failure_reason", "")
    missing_sources = det.get("missing_sources", [])
    missing_facts = det.get("facts_missing", [])
    refusal_passed = det.get("refusal_passed", True)
    facts_cov = det.get("facts_coverage_score", 0.0)
    citation_score = det.get("citation_score", 0.0)
    grounding_score = scores.get("grounding_accuracy", det.get("grounding_score", 1.0))
    hallucination_detected = bool(llm_eval.get("hallucination_detected", False))
    confidence_penalty = det.get("confidence_penalty", 0.0)

    # 1. Check for Runtime Tool-Use Error (Prompt Defect causing crash/error refusal)
    if "tool_use_failed" in answer or "Tool choice is none" in answer or "Error generating answer:" in answer:
        return FailureClassification(
            test_id=test_id,
            question=question,
            test_category=test_cat,
            failure_category=FailureCategory.PROMPT_DEFECT,
            evidence=f"Runtime LLM error in answer: {answer[:180]}",
            root_cause="System prompt instructs model to emit raw tool calls (e.g. incident_ticket_creator) even though tools are not bound in ChatGroq invocation.",
            recommended_fix="Remove raw tool-invocation directives from system_prompt and instruct the model to respond in plain grounded text.",
            target_config_section="system_prompt"
        )

    # 2. Check for Wrong Refusal (either failed to refuse out-of-scope OR refused an answerable query)
    if should_refuse and not refusal_passed:
        return FailureClassification(
            test_id=test_id,
            question=question,
            test_category=test_cat,
            failure_category=FailureCategory.WRONG_REFUSAL,
            evidence=f"should_refuse=True, but agent answered instead of refusing. Verdict: {verdict_reason}",
            root_cause="Agent answered an out-of-scope question (likely due to loosely matched chunks passing similarity_threshold or weak refusal instructions in system_prompt/refusal_policy).",
            recommended_fix="Strengthen refusal_policy and system_prompt to strictly refuse when the exact requested subject is absent from DOCUMENT EVIDENCE, even if tangentially related chunks are retrieved.",
            target_config_section="refusal_policy"
        )

    if not should_refuse and test_cat != "ambiguous" and is_refusal:
        # Did it refuse because no documents were retrieved?
        if not citations or "No relevant document evidence found" in answer:
            return FailureClassification(
                test_id=test_id,
                question=question,
                test_category=test_cat,
                failure_category=FailureCategory.RETRIEVAL_MISS,
                evidence=f"Agent refused answerable query with {len(citations)} citations: {answer[:150]}",
                root_cause="Vector retrieval failed to return relevant chunks within top_k and similarity_threshold.",
                recommended_fix="Increase retrieval_config.top_k and slightly relax similarity_threshold to ensure relevant evidence is retrieved.",
                target_config_section="retrieval_config"
            )
        return FailureClassification(
            test_id=test_id,
            question=question,
            test_category=test_cat,
            failure_category=FailureCategory.WRONG_REFUSAL,
            evidence=f"Agent unjustifiably refused answerable query despite retrieving {len(citations)} citations.",
            root_cause="Overly conservative refusal wording in system_prompt or refusal_policy.",
            recommended_fix="Clarify in system_prompt that when relevant facts ARE present in DOCUMENT EVIDENCE, the agent must answer rather than refuse.",
            target_config_section="refusal_policy"
        )

    # 3. Check for Hallucination
    if hallucination_detected or confidence_penalty > 0 or (grounding_score < 0.35 and not is_refusal and len(citations) > 0 and facts_cov < 0.5):
        return FailureClassification(
            test_id=test_id,
            question=question,
            test_category=test_cat,
            failure_category=FailureCategory.HALLUCINATION,
            evidence=f"Grounding={grounding_score:.2f}, hallucination_detected={hallucination_detected}, missing_facts={missing_facts[:2]}",
            root_cause="Agent generated claims or numbers unsupported by the retrieved DOCUMENT EVIDENCE.",
            recommended_fix="Add strict anti-hallucination and verbatim grounding constraints to system_prompt and output_contract.",
            target_config_section="output_contract"
        )

    # 4. Check for Retrieval Miss (missing expected source documents or empty citations)
    if missing_sources or (not citations and not should_refuse):
        return FailureClassification(
            test_id=test_id,
            question=question,
            test_category=test_cat,
            failure_category=FailureCategory.RETRIEVAL_MISS,
            evidence=f"Missing expected sources: {missing_sources}. Citation score={citation_score:.2f}.",
            root_cause=f"Retriever did not surface all required source documents ({missing_sources}), often because top_k is too low for multi-document queries or citations were omitted in output.",
            recommended_fix="Increase retrieval_config.top_k (e.g., +2) to capture cross-document chunks and enforce explicit source/page citations in output_contract.",
            target_config_section="retrieval_config"
        )

    # 5. Default: Prompt Defect (sources were retrieved, no refusal error, but facts/clarification/formatting missed)
    if test_cat == "ambiguous":
        return FailureClassification(
            test_id=test_id,
            question=question,
            test_category=test_cat,
            failure_category=FailureCategory.PROMPT_DEFECT,
            evidence=f"Ambiguous query missed expected clarification/facts: {missing_facts[:2]}",
            root_cause="System prompt lacks explicit instructions for handling ambiguous or underspecified user queries (missing component ID, missing date/timeframe, or single-snapshot limits).",
            recommended_fix="Add ambiguous-query handling instructions in system_prompt: explicitly identify missing parameters (timeframe, specific component, logs) and ask for clarification while citing available documents.",
            target_config_section="system_prompt"
        )

    return FailureClassification(
        test_id=test_id,
        question=question,
        test_category=test_cat,
        failure_category=FailureCategory.PROMPT_DEFECT,
        evidence=f"Sources retrieved, but missing expected facts: {missing_facts[:2]}. Verdict: {verdict_reason}",
        root_cause="Prompt or output_contract did not guide the LLM to extract and explicitly state all expected facts, units, or citations.",
        recommended_fix="Refine system_prompt and output_contract constraints to preserve exact numerical values, units, all matching items, and [source: <filename>, page: <page_number>] citations.",
        target_config_section="system_prompt"
    )


class CriticAgent:
    """
    Diagnoses evaluation failures and produces targeted recommendations
    without regenerating the entire agent.
    """

    def __init__(self, llm: Optional[ChatGroq] = None, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.sys_prompt, self.diag_prompt = load_critic_prompts()
        if not mock_mode:
            try:
                self.llm = llm or get_groq_llm(temperature=0.0)
            except Exception as e:
                logger.warning("CriticAgent could not initialize Groq LLM (%s). Using deterministic critic.", str(e))
                self.llm = None
                self.mock_mode = True
        else:
            self.llm = None

    def diagnose(
        self,
        failed_results: List[Dict[str, Any]],
        agent_def: AgentDefinition,
        iteration: int = 1
    ) -> CriticDiagnosis:
        """
        Classifies every failed test case and produces a structured CriticDiagnosis.
        """
        if not failed_results:
            return CriticDiagnosis(
                iteration=iteration,
                total_failures=0,
                failed_test_ids=[],
                failure_category_counts={},
                classifications=[],
                overall_root_cause_summary="No failures to diagnose.",
                prioritized_recommendations=[]
            )

        # Step 1: Deterministic classification for every failure (guaranteed coverage)
        classifications: List[FailureClassification] = [
            classify_failure_deterministically(fc) for fc in failed_results
        ]

        # Step 2: Optionally refine diagnosis with Groq LLM
        overall_summary = ""
        if not self.mock_mode and self.llm is not None:
            overall_summary = self._refine_with_llm(
                failed_results=failed_results,
                agent_def=agent_def,
                classifications=classifications
            )

        # Compute counts per category
        counts: Dict[str, int] = {}
        for c in classifications:
            cat_val = c.failure_category.value
            counts[cat_val] = counts.get(cat_val, 0) + 1

        if not overall_summary:
            top_cats = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
            overall_summary = f"Diagnosed {len(classifications)} failures ({top_cats})."

        # Aggregate targeted recommendations by config section
        section_map: Dict[str, List[FailureClassification]] = {}
        for c in classifications:
            section_map.setdefault(c.target_config_section, []).append(c)

        prioritized_recs: List[Dict[str, Any]] = []
        for section, items in sorted(section_map.items(), key=lambda x: -len(x[1])):
            prioritized_recs.append({
                "target_config_section": section,
                "affected_test_ids": [i.test_id for i in items],
                "failure_categories": sorted(list({i.failure_category.value for i in items})),
                "recommended_action": items[0].recommended_fix,
                "count": len(items)
            })

        return CriticDiagnosis(
            iteration=iteration,
            total_failures=len(classifications),
            failed_test_ids=[c.test_id for c in classifications],
            failure_category_counts=counts,
            classifications=classifications,
            overall_root_cause_summary=overall_summary,
            prioritized_recommendations=prioritized_recs
        )

    def _refine_with_llm(
        self,
        failed_results: List[Dict[str, Any]],
        agent_def: AgentDefinition,
        classifications: List[FailureClassification]
    ) -> str:
        """Uses Groq LLM to produce a concise root-cause synthesis and refine diagnoses."""
        summaries = []
        for fc in failed_results[:15]:  # Cap prompt size cleanly
            scores = fc.get("scores", {})
            det = fc.get("deterministic_checks", {})
            summaries.append(
                f"- {fc.get('test_id')} [{fc.get('category')}] should_refuse={fc.get('should_refuse')} "
                f"is_refusal={fc.get('is_refusal')} score={scores.get('composite_score', 0):.2f}\n"
                f"  Question: {fc.get('question')}\n"
                f"  Missing facts: {det.get('facts_missing', [])} | Missing sources: {det.get('missing_sources', [])}\n"
                f"  Answer preview: {str(fc.get('generated_answer', ''))[:180]}"
            )

        user_msg = self.diag_prompt.format(
            agent_version=agent_def.version,
            retrieval_config=agent_def.retrieval_config.model_dump_json(indent=2),
            refusal_policy=agent_def.refusal_policy.model_dump_json(indent=2),
            system_prompt=agent_def.system_prompt[:1200],
            num_failures=len(failed_results),
            failed_cases_summary="\n".join(summaries)
        )

        try:
            resp = self.llm.invoke([
                SystemMessage(content=self.sys_prompt),
                HumanMessage(content=user_msg)
            ])
            raw = resp.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                llm_map = {
                    item.get("test_id"): item
                    for item in data.get("classifications", [])
                    if isinstance(item, dict) and item.get("test_id")
                }
                valid_cats = {e.value: e for e in FailureCategory}
                for c in classifications:
                    if c.test_id in llm_map:
                        item = llm_map[c.test_id]
                        # Preserve deterministic Wrong refusal / tool_use_failed if strictly detected,
                        # otherwise accept valid LLM refinement
                        cat_str = item.get("failure_category", "")
                        if cat_str in valid_cats and "tool_use_failed" not in c.evidence:
                            c.failure_category = valid_cats[cat_str]
                        if item.get("root_cause"):
                            c.root_cause = str(item["root_cause"])
                        if item.get("recommended_fix"):
                            c.recommended_fix = str(item["recommended_fix"])
                        sec = item.get("target_config_section", "")
                        if sec in {"retrieval_config", "system_prompt", "refusal_policy", "output_contract"}:
                            c.target_config_section = sec

                return str(data.get("overall_root_cause_summary", ""))
        except Exception as e:
            logger.warning("Critic LLM refinement failed (%s). Retaining deterministic diagnosis.", str(e))

        return ""

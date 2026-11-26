"""
ForgeAI Phase 2: Groundability Guard
Analyzes document content and vector store evidence against requested topics.
Classifies each topic as supported, partially_supported, or unsupported based on factual evidence.
Generates groundability_report.json and prevents unsupported topics from being included
in guaranteed capabilities.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from config import settings
from src.vectorstore.chroma_store import get_vector_store
from src.llm.groq_client import get_groq_llm
from src.utils.logger import logger
from intake import RequirementSpec


class GroundingStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class EvidenceItem(BaseModel):
    """Factual document citation supporting or relevant to a topic."""
    source_file: str = Field(description="Name of the source document")
    page_number: Optional[int] = Field(default=None, description="Page number of the document")
    row_number: Optional[int] = Field(default=None, description="Row index if originating from CSV")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index within the source")
    similarity_score: float = Field(description="Cosine distance from query (lower indicates closer semantic match)")
    excerpt: str = Field(description="Extracted text snippet from document")


class TopicGroundingEvaluation(BaseModel):
    """Groundability assessment for a single requested topic."""
    topic: str = Field(description="The topic evaluated against document evidence")
    status: GroundingStatus = Field(description="Classification: supported, partially_supported, or unsupported")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Document excerpts retrieved as evidence")
    reasoning: str = Field(description="Fact-based justification for the classification")
    missing_information: Optional[str] = Field(
        default=None,
        description="Explicit description of missing data, parameters, or documentation"
    )


class GroundabilityReport(BaseModel):
    """Comprehensive groundability audit report."""
    total_topics_analyzed: int
    supported_count: int
    partially_supported_count: int
    unsupported_count: int
    topic_evaluations: List[TopicGroundingEvaluation]
    guaranteed_capabilities: List[str] = Field(
        default_factory=list,
        description="Strictly verified topics supported by uploaded document evidence. Unsupported topics are excluded."
    )
    partially_supported_topics: List[str] = Field(
        default_factory=list,
        description="Topics with incomplete evidence, requiring additional documentation"
    )
    blocked_unsupported_topics: List[str] = Field(
        default_factory=list,
        description="Topics blocked from system capabilities due to lack of document ground truth"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended documents or data uploads to resolve missing coverage"
    )
    evaluated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def save_to_json(self, file_path: Union[str, Path] = "groundability_report.json") -> Path:
        """Saves the report to groundability_report.json."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.model_dump_json(indent=2)
        p.write_text(content, encoding="utf-8")
        logger.info("Saved groundability report to %s", p.resolve())
        return p

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path] = "groundability_report.json") -> "GroundabilityReport":
        """Loads a GroundabilityReport from a JSON file."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Report file not found at: {p}")
        return cls.model_validate_json(p.read_text(encoding="utf-8"))


class _LLMTopicGuardVerification(BaseModel):
    """Internal schema for topic grounding analysis via LLM."""
    status: GroundingStatus = Field(description="'supported', 'partially_supported', or 'unsupported'")
    confidence: float = Field(description="Confidence between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of what is supported or missing based ONLY on the excerpts")
    missing_information: Optional[str] = Field(default=None, description="Specific details missing from the document evidence")


GROUNDING_PROMPT = """You are the ForgeAI Groundability Guard.
Your role is to evaluate whether a requested topic is grounded in the provided DOCUMENT EVIDENCE.

CRITICAL RULES:
1. Rely EXCLUSIVELY on the provided Document Excerpts. Do NOT use outside training knowledge or assumptions.
2. Classify the topic into exactly one category:
   - 'supported': The provided excerpts contain clear, concrete, and sufficient facts to answer operational questions regarding this topic.
   - 'partially_supported': The excerpts mention or relate to the topic, but critical information (e.g. procedures, threshold values, specific steps) is missing or incomplete.
   - 'unsupported': The excerpts contain no relevant facts or only tangential mentions unable to answer questions on this topic.
3. If 'partially_supported' or 'unsupported', you MUST explicitly identify what information is missing.
4. If excerpts are empty or unrelated, status MUST be 'unsupported'.
"""


def _retrieve_evidence_for_topic(
    topic: str,
    vector_store: Any,
    top_k: int = 4,
    distance_threshold: float = 1.35
) -> List[EvidenceItem]:
    """
    Retrieves candidate evidence chunks from ChromaDB with similarity distances.
    """
    evidence_items: List[EvidenceItem] = []
    try:
        results: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(query=topic, k=top_k)
        for doc, score in results:
            # Cosine distance: smaller is closer match
            evidence_items.append(
                EvidenceItem(
                    source_file=doc.metadata.get("filename", "unknown"),
                    page_number=doc.metadata.get("page_number"),
                    row_number=doc.metadata.get("row_number"),
                    chunk_index=doc.metadata.get("chunk_index"),
                    similarity_score=round(float(score), 4),
                    excerpt=doc.page_content.strip()
                )
            )
    except Exception as e:
        logger.error("Error retrieving evidence for topic '%s': %s", topic, str(e))

    return evidence_items


def evaluate_topic_groundability(
    topic: str,
    evidence_items: List[EvidenceItem],
    llm: Optional[Any] = None,
    distance_threshold: float = 1.35
) -> TopicGroundingEvaluation:
    """
    Compares a topic with retrieved document evidence to classify grounding status.
    Uses document evidence rather than relying only on LLM assumptions.
    """
    # Filter highly distant / irrelevant evidence
    valid_evidence = [e for e in evidence_items if e.similarity_score <= distance_threshold]

    # If no evidence chunks meet the threshold, topic is completely unsupported
    if not valid_evidence:
        return TopicGroundingEvaluation(
            topic=topic,
            status=GroundingStatus.UNSUPPORTED,
            confidence=0.95,
            evidence=[],
            reasoning="No relevant document evidence found in uploaded knowledge base within similarity threshold.",
            missing_information="Entire topic is absent from the provided documentation.",
        )

    # Format excerpts for LLM verification
    excerpts_text = "\n\n".join([
        f"[Excerpt {i+1} from {e.source_file} (Page {e.page_number or 1}, Score: {e.similarity_score})]:\n{e.excerpt}"
        for i, e in enumerate(valid_evidence)
    ])

    user_prompt = (
        f"TOPIC TO EVALUATE:\n\"{topic}\"\n\n"
        f"DOCUMENT EXCERPTS:\n{excerpts_text}\n\n"
        "Analyze whether the excerpts provide sufficient facts to answer this topic."
    )

    active_llm = llm or get_groq_llm()

    try:
        structured_llm = active_llm.with_structured_output(_LLMTopicGuardVerification)
        result: _LLMTopicGuardVerification = structured_llm.invoke([
            SystemMessage(content=GROUNDING_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        
        return TopicGroundingEvaluation(
            topic=topic,
            status=result.status,
            confidence=result.confidence,
            evidence=valid_evidence,
            reasoning=result.reasoning,
            missing_information=result.missing_information,
        )
    except Exception as e:
        logger.warning("LLM verification failed for '%s' (%s). Using heuristic evaluation.", topic, str(e))
        # Deterministic fallback based on best score and content
        best_score = min(e.similarity_score for e in valid_evidence)
        if best_score < 0.95:
            status = GroundingStatus.SUPPORTED
            reasoning = f"Direct semantic document match found (best distance: {best_score})."
            missing = None
        else:
            status = GroundingStatus.PARTIALLY_SUPPORTED
            reasoning = f"Marginal semantic match found (best distance: {best_score}). Partial information only."
            missing = "In-depth operational specifications or procedure details."

        return TopicGroundingEvaluation(
            topic=topic,
            status=status,
            confidence=0.80,
            evidence=valid_evidence,
            reasoning=reasoning,
            missing_information=missing,
        )


def analyze_groundability(
    requirements: Union[RequirementSpec, Dict[str, Any], List[str], Path, str],
    vector_store: Optional[Any] = None,
    output_path: Optional[Union[str, Path]] = "groundability_report.json",
    llm: Optional[Any] = None,
    top_k: int = 4
) -> GroundabilityReport:
    """
    Main Groundability Guard entry point:
    1. Analyzes uploaded document content via vector store evidence.
    2. Compares requested topics with available evidence.
    3. Classifies each topic as supported, partially supported, or unsupported.
    4. Identifies missing information.
    5. Generates groundability_report.json.
    6. Prevents unsupported topics from being included in guaranteed capabilities.
    """
    # Resolve topics from input
    topics_to_evaluate: List[str] = []
    if isinstance(requirements, RequirementSpec):
        topics_to_evaluate = list(requirements.must_answer_topics)
    elif isinstance(requirements, (str, Path)) and Path(requirements).is_file():
        spec = RequirementSpec.load_from_json(requirements)
        topics_to_evaluate = list(spec.must_answer_topics)
    elif isinstance(requirements, dict):
        topics_to_evaluate = list(requirements.get("must_answer_topics", []))
    elif isinstance(requirements, list):
        topics_to_evaluate = list(requirements)
    else:
        raise ValueError("Invalid requirements input provided to analyze_groundability.")

    if not topics_to_evaluate:
        logger.warning("No topics provided to analyze_groundability.")

    store = vector_store or get_vector_store()
    active_llm = llm or get_groq_llm()

    logger.info("Groundability Guard analyzing %d topic(s)...", len(topics_to_evaluate))

    evaluations: List[TopicGroundingEvaluation] = []
    guaranteed_capabilities: List[str] = []
    partially_supported_topics: List[str] = []
    blocked_unsupported_topics: List[str] = []
    recommendations: List[str] = []

    for topic in topics_to_evaluate:
        evidence = _retrieve_evidence_for_topic(topic=topic, vector_store=store, top_k=top_k)
        eval_result = evaluate_topic_groundability(
            topic=topic,
            evidence_items=evidence,
            llm=active_llm
        )
        evaluations.append(eval_result)

        if eval_result.status == GroundingStatus.SUPPORTED:
            guaranteed_capabilities.append(topic)
        elif eval_result.status == GroundingStatus.PARTIALLY_SUPPORTED:
            partially_supported_topics.append(topic)
            if eval_result.missing_information:
                recommendations.append(f"For '{topic}': Upload documentation addressing: {eval_result.missing_information}")
        else:  # UNSUPPORTED
            # STRICTLY PREVENT unsupported topics from being included as guaranteed capabilities
            blocked_unsupported_topics.append(topic)
            recommendations.append(f"Blocked unsupported topic '{topic}'. Provide manual or technical specifications to enable.")

    supported_count = len(guaranteed_capabilities)
    partially_supported_count = len(partially_supported_topics)
    unsupported_count = len(blocked_unsupported_topics)

    report = GroundabilityReport(
        total_topics_analyzed=len(topics_to_evaluate),
        supported_count=supported_count,
        partially_supported_count=partially_supported_count,
        unsupported_count=unsupported_count,
        topic_evaluations=evaluations,
        guaranteed_capabilities=guaranteed_capabilities,
        partially_supported_topics=partially_supported_topics,
        blocked_unsupported_topics=blocked_unsupported_topics,
        recommendations=recommendations,
    )

    if output_path:
        report.save_to_json(output_path)

    logger.info(
        "Groundability Guard finished. Supported: %d, Partial: %d, Blocked/Unsupported: %d",
        supported_count,
        partially_supported_count,
        unsupported_count
    )
    return report


if __name__ == "__main__":
    req_file = Path("requirements.json")
    if not req_file.exists():
        print("requirements.json not found. Run intake.py first.")
    else:
        # Include an unsupported topic to demonstrate blocking
        spec = RequirementSpec.load_from_json(req_file)
        spec.must_answer_topics.append("Rocket propulsion orbital flight trajectory calculations")
        report = analyze_groundability(spec)
        print("\nGenerated groundability_report.json:")
        print(report.model_dump_json(indent=2))

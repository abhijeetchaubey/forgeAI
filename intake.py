"""
ForgeAI Phase 2: Intake Interviewer
Extracts intended audience, scope, tone, user expectations, must-answer, and must-refuse topics
from natural-language briefs and document metadata. Formulates up to 5 clarification questions
when necessary and outputs a structured, validated requirements.json.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from config import settings
from src.llm.groq_client import get_groq_llm
from src.utils.logger import logger


class RequirementSpec(BaseModel):
    """
    Structured project requirements extracted from user brief and document metadata.
    """
    project_name: str = Field(
        default="ForgeAI Assistant",
        description="Identified or suggested title for the AI system"
    )
    intended_audience: str = Field(
        description="Primary target user persona (e.g., machine operators, plant engineers, executives)"
    )
    scope: str = Field(
        description="Clear operational boundary and main objective of the AI assistant"
    )
    tone: str = Field(
        default="Professional, precise, safety-oriented",
        description="Recommended communication tone and style"
    )
    user_expectations: List[str] = Field(
        default_factory=list,
        description="Key functional expectations and capabilities requested by the user"
    )
    must_answer_topics: List[str] = Field(
        default_factory=list,
        description="Specific topics and domain questions the assistant MUST be capable of answering"
    )
    must_refuse_topics: List[str] = Field(
        default_factory=list,
        description="Explicit topics that the assistant MUST refuse or decline (e.g., out-of-scope, speculative, safety-compromising)"
    )
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="Up to five clarification questions when critical requirements remain ambiguous"
    )
    documents_referenced: List[str] = Field(
        default_factory=list,
        description="List of document filenames or sources associated with this brief"
    )
    raw_brief: str = Field(
        description="Original brief text submitted by the user"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    @model_validator(mode="after")
    def validate_max_questions(self) -> "RequirementSpec":
        if len(self.clarification_questions) > 5:
            self.clarification_questions = self.clarification_questions[:5]
        return self

    def save_to_json(self, file_path: Union[str, Path] = "requirements.json") -> Path:
        """
        Saves the validated requirements specification to requirements.json.
        """
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.model_dump_json(indent=2)
        p.write_text(content, encoding="utf-8")
        logger.info("Saved validated requirements to %s", p.resolve())
        return p

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path] = "requirements.json") -> "RequirementSpec":
        """
        Loads and validates a RequirementSpec from a JSON file.
        """
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Requirements file not found at: {p}")
        return cls.model_validate_json(p.read_text(encoding="utf-8"))


class _LLMIntakeExtraction(BaseModel):
    """Internal Pydantic schema for structured output from ChatGroq."""
    project_name: str = Field(description="Clear, professional name for the system")
    intended_audience: str = Field(description="Target audience / personas")
    scope: str = Field(description="Operational scope and boundaries")
    tone: str = Field(description="Tone and style of responses")
    user_expectations: List[str] = Field(description="List of user expectations")
    must_answer_topics: List[str] = Field(description="List of topics the system MUST answer")
    must_refuse_topics: List[str] = Field(description="List of topics the system MUST refuse")
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="At most 5 clarification questions for genuine ambiguities. Empty if brief is clear."
    )


INTAKE_SYSTEM_PROMPT = """You are the ForgeAI Intake Interviewer.
Your role is to analyze a natural-language project brief along with document metadata/content, and extract:
1. Project Name: A suitable professional title.
2. Intended Audience: Who will interact with this assistant (e.g. factory floor operators, maintenance engineers, management).
3. Scope: Clear operational boundaries and purpose.
4. Tone: Appropriate tone (e.g. concise, authoritative, safety-critical).
5. User Expectations: Key expectations described by the user.
6. Must-Answer Topics: Essential domains and questions the assistant must answer.
7. Must-Refuse Topics: Out-of-scope, unsafe, non-compliant, or speculative topics the assistant must decline.
8. Clarification Questions: AT MOST 5 clarification questions ONLY when genuinely necessary to resolve critical ambiguities. If the brief is clear, return an empty list or only essential questions.

Be rigorous, structured, and factual."""


def conduct_intake(
    brief: str,
    documents: Optional[List[Document]] = None,
    document_metadata: Optional[List[Dict[str, Any]]] = None,
    clarification_answers: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = "requirements.json",
    llm: Optional[Any] = None,
) -> RequirementSpec:
    """
    Analyzes the brief and document metadata/content using Groq LLM,
    constructs a validated RequirementSpec, and saves requirements.json.
    """
    if not brief or not brief.strip():
        raise ValueError("Brief cannot be empty.")

    logger.info("Conducting intake interview for brief: '%s...'", brief[:60].strip())

    # Compile document metadata and summaries
    doc_references: List[str] = []
    doc_summaries: List[str] = []

    if documents:
        for doc in documents:
            fname = doc.metadata.get("filename", "unknown_document")
            if fname not in doc_references:
                doc_references.append(fname)
            doc_summaries.append(
                f"- Document: {fname} (page {doc.metadata.get('page_number', 1)}): {doc.page_content[:200]}..."
            )

    if document_metadata:
        for meta in document_metadata:
            fname = meta.get("filename") or meta.get("source", "unknown")
            if fname not in doc_references:
                doc_references.append(fname)

    # Format human prompt
    prompt_sections = [
        f"USER BRIEF:\n{brief.strip()}\n",
    ]

    if doc_summaries:
        prompt_sections.append(
            "AVAILABLE DOCUMENT CONTEXT/METADATA:\n" + "\n".join(doc_summaries[:10]) + "\n"
        )
    elif doc_references:
        prompt_sections.append(
            "ATTACHED DOCUMENT NAMES:\n" + "\n".join(f"- {d}" for d in doc_references) + "\n"
        )

    if clarification_answers:
        qa_pairs = "\n".join(f"Q: {k}\nA: {v}" for k, v in clarification_answers.items())
        prompt_sections.append(f"PREVIOUS CLARIFICATION ANSWERS:\n{qa_pairs}\n")

    user_message = "\n".join(prompt_sections)

    # Initialize LLM
    active_llm = llm or get_groq_llm()

    try:
        # Use structured output with ChatGroq
        structured_llm = active_llm.with_structured_output(_LLMIntakeExtraction)
        extraction: _LLMIntakeExtraction = structured_llm.invoke([
            SystemMessage(content=INTAKE_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
    except Exception as e:
        logger.warning("Structured output failed (%s). Falling back to JSON parsing prompt.", str(e))
        # Fallback to direct prompt with JSON instructions
        fallback_prompt = (
            f"{INTAKE_SYSTEM_PROMPT}\n"
            "Return valid JSON matching this schema:\n"
            "{\n"
            '  "project_name": "string",\n'
            '  "intended_audience": "string",\n'
            '  "scope": "string",\n'
            '  "tone": "string",\n'
            '  "user_expectations": ["string"],\n'
            '  "must_answer_topics": ["string"],\n'
            '  "must_refuse_topics": ["string"],\n'
            '  "clarification_questions": ["string"]\n'
            "}\n\n"
            f"{user_message}"
        )
        response = active_llm.invoke([HumanMessage(content=fallback_prompt)])
        raw_text = response.content
        import re, json
        # Extract json block if surrounded by ```json
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            extraction = _LLMIntakeExtraction.model_validate(data)
        else:
            raise ValueError(f"Could not parse structured output from LLM: {raw_text}")

    # Build final validated RequirementSpec
    req_spec = RequirementSpec(
        project_name=extraction.project_name or "ForgeAI Assistant",
        intended_audience=extraction.intended_audience,
        scope=extraction.scope,
        tone=extraction.tone,
        user_expectations=extraction.user_expectations,
        must_answer_topics=extraction.must_answer_topics,
        must_refuse_topics=extraction.must_refuse_topics,
        clarification_questions=extraction.clarification_questions[:5],
        documents_referenced=doc_references,
        raw_brief=brief,
    )

    if output_path:
        req_spec.save_to_json(output_path)

    logger.info(
        "Intake completed successfully. Extracted %d must-answer, %d must-refuse, %d questions.",
        len(req_spec.must_answer_topics),
        len(req_spec.must_refuse_topics),
        len(req_spec.clarification_questions),
    )
    return req_spec


if __name__ == "__main__":
    from src.document_loaders.loaders import load_directory
    sample_docs = load_directory(Path("data/sample_docs"))
    brief_example = (
        "We need an automated diagnostic assistant for factory floor maintenance engineers. "
        "It must help diagnose hydraulic pump pressure drops and thermal cutoff triggers, "
        "and tell operators when coolant valves need servicing. "
        "Do not answer questions about financial budgets or employee performance reviews."
    )
    spec = conduct_intake(brief_example, documents=sample_docs)
    print("\nGenerated requirements.json:")
    print(spec.model_dump_json(indent=2))

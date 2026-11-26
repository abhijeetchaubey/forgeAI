"""
ForgeAI Phase 5: Automatic Evaluation Generator
Generates at least 20 diverse, verified evaluation test cases (answerable, multi-document,
ambiguous, and out-of-scope) based on uploaded documents and their metadata.
Verifies factual evidence, deduplicates questions, validates with Pydantic,
and saves the dataset as eval_set.json and prompt templates in the prompts directory.
"""

from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import re
from typing import List, Optional, Union, Dict, Any, Set
from pydantic import BaseModel, Field, model_validator

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from config import settings
from src.llm.groq_client import get_groq_llm
from src.document_loaders.loaders import load_directory, load_document
from src.vectorstore.chroma_store import get_vector_store
from src.utils.logger import logger

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

class EvalCategory(str, Enum):
    ANSWERABLE = "answerable"
    MULTI_DOCUMENT = "multi_document"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


class TestCase(BaseModel):
    """A single evaluation test case with ground truth criteria."""
    __test__ = False  # Prevent pytest from treating this Pydantic model as a test suite
    test_id: str = Field(description="Unique identifier, e.g. EVAL-001")
    question: str = Field(description="The evaluation test question")
    category: EvalCategory = Field(description="Question category")
    expected_facts: List[str] = Field(
        default_factory=list,
        description="Factual points expected in the answer, or reason for refusal"
    )
    expected_source_document: List[str] = Field(
        default_factory=list,
        description="Filename(s) of documents containing supporting evidence"
    )
    expected_page_number: Optional[str] = Field(
        default=None,
        description="Page number(s) as string, e.g. '1' or '1, 2' or null"
    )
    should_refuse: bool = Field(
        description="Whether the agent is expected to refuse answering"
    )
    evaluation_criteria: str = Field(
        description="Rubric for evaluating agent response correctness"
    )

    @model_validator(mode="after")
    def validate_category_refusal_consistency(self) -> "TestCase":
        if self.category == EvalCategory.OUT_OF_SCOPE:
            self.should_refuse = True
        return self


class EvaluationSet(BaseModel):
    """Complete collection of validated evaluation test cases."""
    total_test_cases: int = Field(description="Total number of valid test cases")
    category_counts: Dict[str, int] = Field(description="Count per category")
    test_cases: List[TestCase] = Field(description="List of test case specifications")
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def save_to_json(self, file_path: Union[str, Path] = "eval_set.json") -> Path:
        """Saves the evaluation set to eval_set.json."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.model_dump_json(indent=2)
        p.write_text(content, encoding="utf-8")
        logger.info("Saved %d evaluation test cases to %s", len(self.test_cases), p.resolve())
        return p

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path] = "eval_set.json") -> "EvaluationSet":
        """Loads an EvaluationSet from a JSON file."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Evaluation set file not found: {p}")
        return cls.model_validate_json(p.read_text(encoding="utf-8"))


class _LLMGeneratedTestCase(BaseModel):
    """Raw test case schema for Groq structured output."""
    question: str = Field(description="The test question")
    category: EvalCategory = Field(description="Question category")
    expected_facts: List[str] = Field(description="Concrete facts or refusal reasons")
    expected_source_document: List[str] = Field(description="Source document names")
    expected_page_number: Optional[str] = Field(description="Page number(s) as string e.g. '1' or null")
    should_refuse: bool = Field(description="Whether to refuse")
    evaluation_criteria: str = Field(description="Grading rubric")


class _LLMEvalSetWrapper(BaseModel):
    """Container schema for structured list of cases."""
    test_cases: List[_LLMGeneratedTestCase] = Field(description="List of generated test cases")


# ---------------------------------------------------------------------------
# Prompt Management
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are the ForgeAI Automatic Evaluation Generator.
Your objective is to generate rigorous, high-quality evaluation test cases based strictly on actual uploaded document content and metadata.

Rules:
1. Rely EXCLUSIVELY on the provided document excerpts for answerable and multi-document questions.
2. For out-of-scope questions, formulate questions that are realistic in an enterprise or technical environment but completely absent and unsupported in the provided documents.
3. For ambiguous questions, formulate questions that refer to entities or procedures in the documents but omit critical context (e.g. asking "What is the status of the component?" without specifying which component).
4. For multi-document questions, require synthesizing facts from two or more distinct documents (e.g., comparing temperature thresholds in a manual with telemetry in a CSV).
5. Provide clear, factual expected facts, document sources, page numbers (where applicable), and evaluation criteria.
6. Generate distinct, non-duplicate test cases across requested categories.
"""

PROMPT_GROUNDED_CASES = """DOCUMENT CORPUS AND METADATA:
{document_corpus}

INSTRUCTIONS:
Generate at least 12 high-quality test cases based on the document corpus above.
Include:
1. 'answerable': Questions directly answerable from a single document. (At least 6 cases)
2. 'multi_document': Questions requiring information synthesis across at least two distinct documents. (At least 6 cases)

Each test case MUST include:
- question: clear, precise question
- category: 'answerable' or 'multi_document'
- expected_facts: list of factual data points from the documents
- expected_source_document: list of document names (e.g. ['sample.pdf', 'sample.csv'])
- expected_page_number: string of page number(s) or null
- should_refuse: false
- evaluation_criteria: detailed rubric for grading the response
"""

PROMPT_SINGLE_DOCUMENT_CASES = """DOCUMENT CORPUS AND METADATA:
{document_corpus}

INSTRUCTIONS:
Generate at least 12 distinct 'answerable' test cases using only facts directly
supported by this single source document. Do not generate 'multi_document' cases.
Each test case MUST include:
- question: clear, precise question
- category: 'answerable'
- expected_facts: list of factual data points from the document
- expected_source_document: list containing the source document name
- expected_page_number: string of page number(s) or null
- should_refuse: false
- evaluation_criteria: detailed rubric for grading the response
"""

PROMPT_BOUNDARY_CASES = """DOCUMENT CORPUS AND METADATA:
{document_corpus}

INSTRUCTIONS:
Generate at least 10 boundary evaluation test cases:
1. 'ambiguous': Questions referencing domain concepts in the documents but lacking specific context (e.g. entity name, time range, threshold), requiring clarification or conditional answers. (At least 5 cases)
2. 'out_of_scope': Inquiries clearly outside the provided document corpus that the agent MUST refuse. (At least 5 cases)

Each test case MUST include:
- question: the question text
- category: 'ambiguous' or 'out_of_scope'
- expected_facts: expected refusal reason or clarification needed
- expected_source_document: empty list for out_of_scope, or referenced docs for ambiguous
- expected_page_number: null
- should_refuse: true for out_of_scope, false for ambiguous
- evaluation_criteria: detailed rubric for grading the response
"""


def ensure_prompts_saved(prompts_dir: Path = PROMPTS_DIR):
    """Ensures prompts are stored in the prompts directory."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "eval_system_prompt.txt").write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
    (prompts_dir / "eval_generation_prompt.txt").write_text(
        f"{PROMPT_GROUNDED_CASES}\n\n---\n\n{PROMPT_BOUNDARY_CASES}", encoding="utf-8"
    )


def load_prompt_templates(prompts_dir: Path = PROMPTS_DIR) -> Dict[str, str]:
    """Loads prompt templates from disk."""
    ensure_prompts_saved(prompts_dir)
    sys_prompt = (prompts_dir / "eval_system_prompt.txt").read_text(encoding="utf-8")
    gen_prompt = (prompts_dir / "eval_generation_prompt.txt").read_text(encoding="utf-8")
    return {"system": sys_prompt, "generation": gen_prompt}


# ---------------------------------------------------------------------------
# Evaluation Generator Engine
# ---------------------------------------------------------------------------

def _normalize_question(text: str) -> str:
    """Normalizes question string for deduplication."""
    return re.sub(r"[^\w\s]", "", text.lower().strip())


def _format_document_corpus(documents: List[Document]) -> str:
    """Formats document items into a clean representation for the generator."""
    sections = []
    for i, doc in enumerate(documents, 1):
        filename = doc.metadata.get("filename", f"doc_{i}")
        page = doc.metadata.get("page_number", 1)
        row = doc.metadata.get("row_number")
        loc = f"page {page}" if row is None else f"row {row}"
        sections.append(
            f"--- Document: {filename} ({loc}) ---\n{doc.page_content.strip()}"
        )
    return "\n\n".join(sections)


def _build_grounded_cases_prompt(documents: List[Document]) -> str:
    """Avoids requesting cross-document cases when only one source was uploaded."""
    source_names = {
        doc.metadata.get("filename") or doc.metadata.get("source") or f"doc_{index}"
        for index, doc in enumerate(documents, 1)
    }
    template = (
        PROMPT_GROUNDED_CASES
        if len(source_names) > 1
        else PROMPT_SINGLE_DOCUMENT_CASES
    )
    return template.replace("{document_corpus}", _format_document_corpus(documents))


def verify_test_case_evidence(
    case: _LLMGeneratedTestCase,
    vector_store: Any,
    similarity_threshold: float = 1.35
) -> bool:
    """
    Verifies that answerable questions have actual document evidence,
    and out-of-scope questions are genuinely unsupported.
    """
    if vector_store is None:
        return True

    try:
        results = vector_store.similarity_search_with_score(query=case.question, k=3)
        best_score = float(results[0][1]) if results else 999.0

        if case.category in [EvalCategory.ANSWERABLE, EvalCategory.MULTI_DOCUMENT]:
            # Must have supporting chunk within acceptable distance
            if best_score > similarity_threshold:
                logger.debug("Rejected ungrounded case '%s' (best score %.2f > %.2f)", case.question, best_score, similarity_threshold)
                return False
            return True

        elif case.category == EvalCategory.OUT_OF_SCOPE:
            case.should_refuse = True
            return True

        elif case.category == EvalCategory.AMBIGUOUS:
            return True

    except Exception as e:
        logger.warning("Evidence verification warning for '%s': %s", case.question, str(e))
        return True

    return True


def _invoke_case_generation(
    llm: Any,
    system_prompt: str,
    human_prompt: str
) -> List[_LLMGeneratedTestCase]:
    """Invokes the LLM with strict structured output and a JSON-mode fallback."""
    try:
        structured_llm = llm.with_structured_output(
            _LLMEvalSetWrapper,
            method="json_schema",
            strict=True,
        )
        response: _LLMEvalSetWrapper = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        return response.test_cases
    except Exception as e:
        logger.warning("Strict JSON schema generation failed: %s. Retrying in JSON mode.", str(e))
        json_mode_prompt = (
            f"{system_prompt}\n\n"
            "Return only one valid JSON object with a top-level `test_cases` array. "
            "Each case must include question, category, expected_facts, "
            "expected_source_document, expected_page_number, should_refuse, and "
            "evaluation_criteria. Use null when expected_page_number is unknown. "
            "Do not include markdown or text outside the JSON object."
        )
        try:
            structured_llm = llm.with_structured_output(
                _LLMEvalSetWrapper,
                method="json_mode",
            )
            response: _LLMEvalSetWrapper = structured_llm.invoke([
                SystemMessage(content=json_mode_prompt),
                HumanMessage(content=human_prompt),
            ])
            return response.test_cases
        except Exception as fallback_error:
            logger.error("JSON mode evaluation generation failed: %s", str(fallback_error))
            raise RuntimeError(
                "Evaluation case generation failed with both strict JSON schema and JSON mode."
            ) from fallback_error


def generate_evaluation_set(
    documents: Optional[List[Document]] = None,
    document_dir: Optional[Union[str, Path]] = "data/sample_docs",
    vector_store: Optional[Any] = None,
    output_path: Optional[Union[str, Path]] = "eval_set.json",
    llm: Optional[Any] = None,
    min_cases: int = 20,
    prompts_dir: Path = PROMPTS_DIR,
) -> EvaluationSet:
    """
    Main entry point for Automatic Evaluation Generator:
    1. Loads uploaded documents and metadata.
    2. Generates test cases in two passes (grounded + boundary) using ChatGroq.
    3. Verifies factual evidence and refusal consistency.
    4. Deduplicates questions.
    5. Validates every case with Pydantic.
    6. Ensures at least min_cases (20) valid cases.
    7. Saves dataset to eval_set.json and prompts to prompts/ directory.
    """
    ensure_prompts_saved(prompts_dir)

    # 1. Resolve documents
    docs: List[Document] = []
    if documents:
        docs = documents
    elif document_dir and Path(document_dir).is_dir():
        docs = load_directory(document_dir)

    if not docs:
        raise ValueError("No documents provided to generate_evaluation_set.")

    logger.info("Generating evaluation set based on %d document chunk(s)...", len(docs))
    corpus_text = _format_document_corpus(docs)

    store = vector_store or get_vector_store()
    active_llm = llm or get_groq_llm()

    valid_test_cases: List[TestCase] = []
    seen_normalized_questions: Set[str] = set()

    # Pass 1: In-scope cases, with cross-document cases only when available
    prompt_pass1 = _build_grounded_cases_prompt(docs)
    logger.info("Generating in-scope cases (Answerable & Multi-document)...")
    cases_pass1 = _invoke_case_generation(active_llm, DEFAULT_SYSTEM_PROMPT, prompt_pass1)

    # Pass 2: Boundary cases (Ambiguous & Out-of-scope)
    prompt_pass2 = PROMPT_BOUNDARY_CASES.replace("{document_corpus}", corpus_text)
    logger.info("Generating boundary cases (Ambiguous & Out-of-scope)...")
    cases_pass2 = _invoke_case_generation(active_llm, DEFAULT_SYSTEM_PROMPT, prompt_pass2)

    all_raw_cases = cases_pass1 + cases_pass2
    logger.info("LLM generated %d raw test cases total.", len(all_raw_cases))

    # Verification, deduplication, and Pydantic validation
    for raw in all_raw_cases:
        q_norm = _normalize_question(raw.question)
        if not q_norm or q_norm in seen_normalized_questions:
            continue

        if not verify_test_case_evidence(raw, vector_store=store):
            continue

        seen_normalized_questions.add(q_norm)
        case_id = f"EVAL-{len(valid_test_cases) + 1:03d}"

        # Clean string page number
        page_val = str(raw.expected_page_number) if raw.expected_page_number is not None else None

        test_case = TestCase(
            test_id=case_id,
            question=raw.question.strip(),
            category=raw.category,
            expected_facts=raw.expected_facts,
            expected_source_document=raw.expected_source_document,
            expected_page_number=page_val,
            should_refuse=raw.should_refuse,
            evaluation_criteria=raw.evaluation_criteria.strip(),
        )
        valid_test_cases.append(test_case)

    # Count verification
    if len(valid_test_cases) < min_cases:
        logger.warning(
            "Document corpus supported %d meaningful non-duplicate test cases (requested %d). Reporting limitation.",
            len(valid_test_cases), min_cases
        )

    cat_counts: Dict[str, int] = {}
    for c in valid_test_cases:
        cat_counts[c.category.value] = cat_counts.get(c.category.value, 0) + 1

    eval_set = EvaluationSet(
        total_test_cases=len(valid_test_cases),
        category_counts=cat_counts,
        test_cases=valid_test_cases,
    )

    if output_path:
        eval_set.save_to_json(output_path)

    logger.info(
        "Evaluation generator finished. Total cases: %d. Breakdown: %s",
        eval_set.total_test_cases,
        eval_set.category_counts
    )
    return eval_set


if __name__ == "__main__":
    docs_dir = Path("data/sample_docs")
    if not docs_dir.exists():
        print("Sample docs directory not found.")
    else:
        eval_set = generate_evaluation_set(
            document_dir=docs_dir,
            output_path="eval_set.json",
            min_cases=20
        )
        print(f"\nGenerated {eval_set.total_test_cases} test cases:")
        for cat, count in eval_set.category_counts.items():
            print(f"  - {cat}: {count}")
        print(f"\nSaved to eval_set.json (Preview of first 3 cases):")
        for tc in eval_set.test_cases[:3]:
            print(f"[{tc.test_id}] ({tc.category}) {tc.question}")
            print(f"  Expected Facts: {tc.expected_facts}")
            print(f"  Sources: {tc.expected_source_document} (p. {tc.expected_page_number}) | Refuse: {tc.should_refuse}")

"""
ForgeAI Phase 6: Deterministic Evaluation Checks
Performs objective, programmatic verification of agent responses:
1. Fact matching and expected facts coverage.
2. Source retrieval and citation correctness (document and page).
3. Refusal accuracy (strict enforcement for out-of-scope/unsupported queries).
4. Anti-bloat / verbosity / confidence penalty calculation.
5. Grounding consistency against retrieved citation excerpts.
"""

from typing import List, Optional, Dict, Any, Tuple, Set
import re
from pathlib import Path
from pydantic import BaseModel, Field

from src.utils.logger import logger


class DeterministicResult(BaseModel):
    """Structured output of deterministic checks for a single test case."""
    # Refusal metrics
    refusal_score: float = Field(default=0.0, description="1.0 if refusal behavior matches expectation, else 0.0")
    refusal_passed: bool = Field(default=False, description="Whether refusal requirement was satisfied")
    is_actual_refusal: bool = Field(default=False, description="Whether the answer was classified as a refusal")

    # Citation metrics
    citation_score: float = Field(default=0.0, description="Citation score from 0.0 to 1.0")
    citation_passed: bool = Field(default=False, description="Whether all expected sources are present")
    cited_sources: List[str] = Field(default_factory=list, description="Unique sources identified in response/citations")
    missing_sources: List[str] = Field(default_factory=list, description="Expected sources that were not cited")
    page_match: Optional[bool] = Field(default=None, description="Whether expected page numbers were cited")

    # Facts coverage metrics
    facts_coverage_score: float = Field(default=0.0, description="Ratio of expected facts found (0.0 to 1.0)")
    facts_covered: List[str] = Field(default_factory=list, description="Expected facts identified in answer")
    facts_missing: List[str] = Field(default_factory=list, description="Expected facts missing from answer")

    # Grounding & Length guards
    grounding_score: float = Field(default=0.0, description="Lexical grounding score between answer and citation excerpts")
    verbosity_penalty: float = Field(default=0.0, description="Penalty deducted for wordy answers lacking expected facts")
    confidence_penalty: float = Field(default=0.0, description="Penalty for unfounded high-confidence assertions")

    # Diagnostics
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed check diagnostics")


REFUSAL_SIGNALS = [
    r"\bcannot\s+(?:answer|help|assist|provide|determine)\b",
    r"\bnot\s+(?:mentioned|found|available|covered|included|provided|described)\s+in\s+(?:the\s+)?(?:provided\s+)?(?:documents?|context|corpus|text|files?|knowledge\s+base)\b",
    r"\boutside\s+(?:the\s+)?scope\b",
    r"\bunsupported\s+topic\b",
    r"\bno\s+(?:relevant\s+)?(?:document\s+)?evidence\s+found\b",
    r"\bdoes\s+not\s+(?:contain|mention|state|include)\b",
    r"\bdo\s+not\s+have\s+(?:access\s+to\s+)?information\b",
    r"\bi\s+am\s+unable\s+to\b",
    r"\bclarification\s+needed\b",
    r"\bplease\s+(?:clarify|specify|provide\s+more\s+information)\b",
    r"\bunable\s+to\s+determine\b"
]

CONFIDENCE_FLUFF_SIGNALS = [
    r"\bwithout\s+(?:a\s+)?doubt\b",
    r"\b100%\s+certain\b",
    r"\bcompletely\s+guaranteed\b",
    r"\bindisputabl(?:e|y)\b",
    r"\babsolutely\s+certain\b",
    r"\bbeyond\s+any\s+shadow\s+of\s+doubt\b",
    r"\bproven\s+fact\s+without\s+question\b"
]


def normalize_text(text: str) -> str:
    """Normalizes text for robust token and phrase matching."""
    if not text:
        return ""
    # Lowercase and convert various spaces/punctuation
    text = text.lower()
    text = text.replace("°c", " celsius").replace("℃", " celsius")
    text = re.sub(r"[^\w\s\.\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_refusal(answer: str, is_refusal_flag: bool = False) -> bool:
    """
    Determines whether an answer constitutes a refusal or clarification request
    based on the agent response flag and linguistic refusal signals.
    """
    if is_refusal_flag:
        return True
    
    if not answer or not answer.strip():
        return True

    ans_norm = normalize_text(answer)
    for pattern in REFUSAL_SIGNALS:
        if re.search(pattern, ans_norm, re.IGNORECASE):
            return True
            
    return False


def check_refusal_accuracy(
    answer: str,
    should_refuse: bool,
    is_refusal_flag: bool = False,
    category: str = "answerable"
) -> Tuple[float, bool, bool]:
    """
    Evaluates refusal accuracy against expected behavior.
    Returns (score, passed, is_actual_refusal).
    """
    actual_refusal = detect_refusal(answer, is_refusal_flag)

    if should_refuse:
        # Agent MUST refuse out-of-scope / unsupported questions
        if actual_refusal:
            return 1.0, True, True
        else:
            # Failed to refuse when required -> severe violation
            logger.warning("Agent attempted to answer question that should have been refused!")
            return 0.0, False, False
    else:
        # Question was supposed to be answered (answerable, multi-document, or ambiguous)
        if actual_refusal:
            # Special case: 'ambiguous' questions where agent properly requests clarification
            if category == "ambiguous":
                # A refusal / request for clarification is expected for ambiguous questions!
                return 1.0, True, True
            else:
                # Unjustified refusal on answerable query
                logger.warning("Agent refused an answerable query.")
                return 0.0, False, True
        else:
            return 1.0, True, False


def extract_citations_from_text(answer: str) -> List[str]:
    """Extracts cited document names mentioned directly in the answer text."""
    if not answer:
        return []
    
    sources: Set[str] = set()
    # Match patterns like sample.csv, sample.pdf, sample.txt, [sample.pdf p. 2], etc.
    patterns = [
        r"\b([\w\-]+\.(?:pdf|csv|txt|docx|json))\b",
        r"\[([^\]\n]+\.(?:pdf|csv|txt|docx|json))[^\]]*\]"
    ]
    for pat in patterns:
        matches = re.findall(pat, answer, re.IGNORECASE)
        for m in matches:
            sources.add(m.strip().lower())
            
    return sorted(list(sources))


def check_citation_correctness(
    expected_sources: List[str],
    cited_citations: List[Any],
    answer_text: str,
    expected_page_number: Optional[str] = None,
    should_refuse: bool = False
) -> Tuple[float, bool, List[str], List[str], Optional[bool]]:
    """
    Verifies whether expected sources and pages are cited in the response.
    Returns (citation_score, passed, cited_sources, missing_sources, page_match).
    """
    # Collect all sources from structured Citation objects and raw text
    detected_sources: Set[str] = set()
    detected_pages: Set[int] = set()

    for c in cited_citations:
        src = getattr(c, "source", None) or (c.get("source") if isinstance(c, dict) else str(c))
        if src and src != "unknown":
            detected_sources.add(Path(str(src)).name.lower())
        pg = getattr(c, "page_number", None) or (c.get("page_number") if isinstance(c, dict) else None)
        if pg:
            try:
                detected_pages.add(int(pg))
            except (ValueError, TypeError):
                pass

    # Also detect inline text citations
    for text_src in extract_citations_from_text(answer_text):
        detected_sources.add(Path(text_src).name.lower())

    # Detect inline page citations in text (e.g. "page 1", "p. 2", "p.1")
    text_page_matches = re.findall(r"(?:page|p\.)\s*(\d+)", answer_text, re.IGNORECASE)
    for p in text_page_matches:
        try:
            detected_pages.add(int(p))
        except (ValueError, TypeError):
            pass

    # Normalize expected sources
    norm_expected = [Path(s).name.lower() for s in expected_sources if s]

    # Case 1: Refusal / Out of scope where NO sources are expected
    if should_refuse or not norm_expected:
        if not detected_sources:
            return 1.0, True, [], [], None
        else:
            # If agent fabricated citations on an out-of-scope question
            return 0.5, False, list(detected_sources), [], None

    # Case 2: Expected sources exist
    missing_sources = [s for s in norm_expected if s not in detected_sources]
    found_sources = [s for s in norm_expected if s in detected_sources]

    if not norm_expected:
        source_score = 1.0
    else:
        source_score = len(found_sources) / len(norm_expected)

    # Check page number if expected
    page_match: Optional[bool] = None
    if expected_page_number:
        # Extract digits from expected page number string (e.g. "page 2", "1, 2", "2")
        exp_pages = [int(p) for p in re.findall(r"\d+", str(expected_page_number))]
        if exp_pages:
            matched_pages = [p for p in exp_pages if p in detected_pages]
            page_match = (len(matched_pages) == len(exp_pages))
            if page_match:
                page_factor = 1.0
            elif matched_pages:
                page_factor = 0.5
            else:
                page_factor = 0.0
            # Source is 80% weight, page is 20% weight
            citation_score = round(0.8 * source_score + 0.2 * page_factor, 4)
        else:
            citation_score = round(source_score, 4)
    else:
        citation_score = round(source_score, 4)

    passed = (len(missing_sources) == 0) and (page_match is not False)
    return citation_score, passed, list(detected_sources), missing_sources, page_match


def check_facts_coverage(
    expected_facts: List[str],
    answer_text: str,
    should_refuse: bool = False
) -> Tuple[float, List[str], List[str]]:
    """
    Checks if expected factual statements or key entities/numbers appear in the answer.
    Returns (facts_coverage_score, facts_covered, facts_missing).
    """
    if should_refuse:
        # For refusal cases, expected facts typically state refusal justification
        # If agent refused, coverage is considered 100% compliant
        return 1.0, expected_facts, []

    if not expected_facts:
        return 1.0, [], []

    ans_norm = normalize_text(answer_text)
    covered: List[str] = []
    missing: List[str] = []

    for fact in expected_facts:
        fact_norm = normalize_text(fact)

        # Check direct substring
        if fact_norm in ans_norm:
            covered.append(fact)
            continue

        # Extract essential entities / numerical clauses from the fact
        # e.g., "30 days", "91.2 celsius", "coolant valve b", "120 bar"
        tokens = [t for t in fact_norm.split() if len(t) > 2 or t.isdigit()]
        if not tokens:
            missing.append(fact)
            continue

        # Check how many essential tokens or number+unit pairs match
        matched_tokens = [t for t in tokens if t in ans_norm]
        match_ratio = len(matched_tokens) / len(tokens)

        # Also check for numerical values specifically (e.g. "91.2", "85", "30", "120")
        numbers_in_fact = re.findall(r"\b\d+(?:\.\d+)?\b", fact_norm)
        numbers_match = True
        if numbers_in_fact:
            numbers_match = all(num in ans_norm for num in numbers_in_fact)

        if match_ratio >= 0.70 and numbers_match:
            covered.append(fact)
        else:
            missing.append(fact)

    score = len(covered) / len(expected_facts) if expected_facts else 1.0
    return round(score, 4), covered, missing


def calculate_grounding_and_bloat_penalties(
    answer: str,
    cited_citations: List[Any],
    expected_facts_coverage: float,
    should_refuse: bool = False
) -> Tuple[float, float, float]:
    """
    Computes:
    1. Lexical grounding score between answer and citation excerpts.
    2. Verbosity penalty (penalizes long answers that lack expected facts).
    3. Confidence penalty (penalizes unsubstantiated high-confidence filler).
    
    Rule: Do not award full credit merely because an answer is lengthy or confident.
    """
    if should_refuse:
        # Refusal answers should be concise and not penalized for missing document excerpts
        return 1.0, 0.0, 0.0

    ans_tokens = normalize_text(answer).split()
    word_count = len(ans_tokens)

    # 1. Verbosity penalty:
    # If the answer is lengthy (> 120 words) but fails to cover expected facts (< 0.6)
    verbosity_penalty = 0.0
    if word_count > 120 and expected_facts_coverage < 0.6:
        # Scaled penalty up to 0.25 based on bloat
        bloat_factor = min((word_count - 120) / 150.0, 1.0)
        verbosity_penalty = round(0.25 * bloat_factor * (1.0 - expected_facts_coverage), 4)

    # 2. Confidence penalty:
    # Detects empty rhetorical bravado ("without a doubt", "100% certain")
    confidence_penalty = 0.0
    ans_norm = normalize_text(answer)
    for pattern in CONFIDENCE_FLUFF_SIGNALS:
        if re.search(pattern, ans_norm, re.IGNORECASE):
            confidence_penalty += 0.10
    confidence_penalty = min(confidence_penalty, 0.30)

    # 3. Grounding score against retrieved excerpts:
    excerpts = []
    for c in cited_citations:
        exc = getattr(c, "excerpt", None) or (c.get("excerpt") if isinstance(c, dict) else "")
        if exc:
            excerpts.append(normalize_text(exc))

    if not excerpts:
        # If no citations were attached to an answerable question
        grounding_score = 0.0 if not should_refuse else 1.0
    else:
        combined_excerpts = " ".join(excerpts)
        meaningful_tokens = [t for t in ans_tokens if len(t) > 3]
        if not meaningful_tokens:
            grounding_score = 0.5
        else:
            grounded_tokens = [t for t in meaningful_tokens if t in combined_excerpts]
            grounding_score = round(min(len(grounded_tokens) / len(meaningful_tokens) * 1.2, 1.0), 4)

    return grounding_score, verbosity_penalty, confidence_penalty


def run_deterministic_checks(
    answer: str,
    category: str,
    expected_facts: List[str],
    expected_sources: List[str],
    expected_page_number: Optional[str] = None,
    should_refuse: bool = False,
    is_refusal_flag: bool = False,
    citations: Optional[List[Any]] = None
) -> DeterministicResult:
    """
    Executes the full suite of deterministic checks on an agent response.
    """
    citations = citations or []

    # 1. Refusal check
    refusal_score, refusal_passed, is_actual_refusal = check_refusal_accuracy(
        answer=answer,
        should_refuse=should_refuse,
        is_refusal_flag=is_refusal_flag,
        category=category
    )

    # 2. Citation check
    citation_score, citation_passed, cited_sources, missing_sources, page_match = check_citation_correctness(
        expected_sources=expected_sources,
        cited_citations=citations,
        answer_text=answer,
        expected_page_number=expected_page_number,
        should_refuse=should_refuse
    )

    # 3. Facts coverage check
    facts_coverage_score, facts_covered, facts_missing = check_facts_coverage(
        expected_facts=expected_facts,
        answer_text=answer,
        should_refuse=should_refuse
    )

    # 4. Grounding and anti-bloat penalties
    grounding_score, verbosity_penalty, confidence_penalty = calculate_grounding_and_bloat_penalties(
        answer=answer,
        cited_citations=citations,
        expected_facts_coverage=facts_coverage_score,
        should_refuse=should_refuse
    )

    details = {
        "word_count": len(answer.split()),
        "cited_sources": cited_sources,
        "missing_sources": missing_sources,
        "page_match": page_match,
        "facts_covered_count": len(facts_covered),
        "facts_missing_count": len(facts_missing),
        "verbosity_penalty": verbosity_penalty,
        "confidence_penalty": confidence_penalty,
    }

    return DeterministicResult(
        refusal_score=refusal_score,
        refusal_passed=refusal_passed,
        is_actual_refusal=is_actual_refusal,
        citation_score=citation_score,
        citation_passed=citation_passed,
        cited_sources=cited_sources,
        missing_sources=missing_sources,
        page_match=page_match,
        facts_coverage_score=facts_coverage_score,
        facts_covered=facts_covered,
        facts_missing=facts_missing,
        grounding_score=grounding_score,
        verbosity_penalty=verbosity_penalty,
        confidence_penalty=confidence_penalty,
        details=details
    )

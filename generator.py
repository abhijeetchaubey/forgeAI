"""
ForgeAI Phase 3: Dynamic Agent Generator
Generates declarative, domain-independent agent definitions from validated requirements,
groundability reports, document metadata, and available tool registries.
Validates the result with Pydantic with controlled retry loops, and saves agent.json.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from src.llm.groq_client import get_groq_llm
from src.utils.logger import logger
from intake import RequirementSpec
from groundability import GroundabilityReport, GroundingStatus


# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

class ToolSpec(BaseModel):
    """Specification of an available or selected agent tool."""
    name: str = Field(description="Unique identifier for the tool")
    description: str = Field(description="Functional description of what the tool does")
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="JSON schema or dictionary of tool parameters"
    )
    enabled: bool = Field(default=True, description="Whether the tool is active for the agent")


class RetrievalConfig(BaseModel):
    """Declarative retrieval and vector search parameters for the agent."""
    embedding_model: str = Field(description="Name or path of the embedding model")
    chunk_size: int = Field(gt=0, description="Document chunk size in characters")
    chunk_overlap: int = Field(ge=0, description="Chunk overlap in characters")
    top_k: int = Field(gt=0, description="Number of document chunks to retrieve per search")
    similarity_threshold: float = Field(
        default=1.35,
        description="Maximum distance threshold for considering a chunk relevant"
    )
    collection_name: str = Field(description="ChromaDB collection identifier")
    persist_directory: str = Field(description="Path to ChromaDB storage directory")

    @model_validator(mode="after")
    def validate_overlap(self) -> "RetrievalConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        return self


class RefusalPolicy(BaseModel):
    """Explicit refusal rules and fallback behaviors."""
    refusal_rules: List[str] = Field(description="Explicit operational boundaries and refusal conditions")
    unsupported_topics_handling: str = Field(
        description="Protocol for handling topics lacking document grounding"
    )
    must_refuse_topics_handling: str = Field(
        description="Protocol for handling prohibited or out-of-scope topics"
    )
    default_refusal_response: str = Field(
        description="Standard template message when refusing to answer"
    )


class OutputContract(BaseModel):
    """Expected response structure, formatting guidelines, and citation policies."""
    format: str = Field(default="markdown", description="Format type: markdown, json, or structured text")
    require_citations: bool = Field(default=True, description="Whether citations are strictly mandatory")
    citation_format: str = Field(
        default="[source: {filename}, page: {page_number}]",
        description="Standard syntax for referencing document sources"
    )
    tone_and_style: str = Field(description="Communication style constraints")
    constraints: List[str] = Field(default_factory=list, description="Output restrictions and guarantees")


class AgentDefinition(BaseModel):
    """
    Comprehensive, declarative, domain-independent agent definition schema.
    """
    agent_name: str = Field(description="Unique and descriptive name for the agent")
    description: str = Field(description="Overview of the agent's purpose and role")
    system_prompt: str = Field(description="Complete generated system prompt guiding agent behavior")
    audience: str = Field(description="Target audience or user personas")
    tone: str = Field(description="Tone and manner of interaction")
    supported_topics: List[str] = Field(
        description="Topics with verified document grounding that the agent is guaranteed to answer"
    )
    unsupported_topics: List[str] = Field(
        description="Topics blocked from guaranteed execution due to lack of ground truth or explicit refusal"
    )
    partially_supported_topics: List[str] = Field(
        default_factory=list,
        description="Topics with limited documentation that require explicit caveats"
    )
    retrieval_config: RetrievalConfig = Field(description="Vector retrieval and embedding parameters")
    selected_tools: List[ToolSpec] = Field(
        description="Subset of tools selected from the registry required for this agent's mission"
    )
    refusal_policy: RefusalPolicy = Field(description="Policy and protocol for declining questions")
    output_contract: OutputContract = Field(description="Guaranteed formatting and citation contract")
    version: str = Field(default="1.0.0", description="Configuration version")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def save_to_json(self, file_path: Union[str, Path] = "agent.json") -> Path:
        """Saves the validated agent definition to agent.json."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.model_dump_json(indent=2)
        p.write_text(content, encoding="utf-8")
        logger.info("Saved agent definition to %s", p.resolve())
        return p

    @classmethod
    def load_from_json(cls, file_path: Union[str, Path] = "agent.json") -> "AgentDefinition":
        """Loads and validates an AgentDefinition from JSON."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Agent definition file not found at: {p}")
        return cls.model_validate_json(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Default Tool Registry
# ---------------------------------------------------------------------------

DEFAULT_TOOL_REGISTRY: List[ToolSpec] = [
    ToolSpec(
        name="document_retriever",
        description="Searches and retrieves semantic document chunks from the local vectorized knowledge base.",
        parameters={
            "query": {"type": "string", "description": "Search query keywords or questions"},
            "top_k": {"type": "integer", "description": "Number of chunks to return", "default": 4}
        },
        enabled=True
    ),
    ToolSpec(
        name="tabular_data_inspector",
        description="Performs exact lookups and filters on structured CSV telemetry, equipment tables, or tabular data.",
        parameters={
            "table_name": {"type": "string", "description": "Name of the target table or CSV file"},
            "filter_key": {"type": "string", "description": "Column or attribute name to filter on"},
            "filter_value": {"type": "string", "description": "Value to match"}
        },
        enabled=True
    ),
    ToolSpec(
        name="engineering_calculator",
        description="Performs mathematical calculations, threshold evaluations, unit conversions, and formula checks.",
        parameters={
            "expression": {"type": "string", "description": "Mathematical formula or expression to calculate"}
        },
        enabled=True
    ),
    ToolSpec(
        name="incident_ticket_creator",
        description="Generates an escalation ticket or maintenance request in the system ledger when anomalies or critical thresholds are detected.",
        parameters={
            "title": {"type": "string", "description": "Incident title"},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "details": {"type": "string", "description": "Detailed explanation of the issue"}
        },
        enabled=True
    ),
    ToolSpec(
        name="external_web_search",
        description="Queries public internet search engines for external documentation or general world knowledge.",
        parameters={
            "query": {"type": "string", "description": "Web search query"}
        },
        enabled=False  # Disabled by default for closed-domain agents
    )
]


# ---------------------------------------------------------------------------
# Internal Extraction Schemas for Structured Generation
# ---------------------------------------------------------------------------

class _LLMGeneratedAgentComponents(BaseModel):
    """Pydantic schema for Groq structured output when synthesizing agent components."""
    agent_name: str = Field(description="Concise, professional name for the agent")
    description: str = Field(description="Role and objective summary")
    system_prompt: str = Field(
        description="Comprehensive system prompt defining persona, scope, grounding guidelines, and refusal policy"
    )
    selected_tool_names: List[str] = Field(
        description="List of tool names selected from the available tool registry that this agent requires"
    )
    refusal_rules: List[str] = Field(
        description="Specific refusal rules governing when the agent must decline to answer"
    )
    unsupported_topics_handling: str = Field(
        description="Clear statement on how questions lacking document support will be declined"
    )
    must_refuse_topics_handling: str = Field(
        description="Clear statement on how out-of-scope or prohibited topics will be declined"
    )
    default_refusal_response: str = Field(
        description="Standard template phrase used when declining an inquiry"
    )
    tone_and_style: str = Field(description="Detailed tone and communication style guidelines")
    output_constraints: List[str] = Field(
        description="Specific constraints on response format (e.g., citation requirements, conciseness)"
    )


GENERATOR_SYSTEM_PROMPT = """You are the ForgeAI Dynamic Agent Generator.
Your task is to synthesize a complete, declarative, domain-independent AI Agent Definition based strictly on:
1. Validated Requirements (project scope, audience, tone, must-answer, must-refuse topics).
2. Groundability Report (verified supported topics, partially supported topics, and blocked unsupported topics).
3. Document Metadata (available files, types, and sources).
4. Tool Registry (available capabilities).

CRITICAL DESIGN PRINCIPLES:
1. Domain-Independence: Do NOT assume or hardcode any domain (e.g. HR, manufacturing, finance, healthcare). Reflect ONLY the provided requirements and groundability data.
2. Grounding Integrity: The agent must ONLY guarantee answers for topics with verified document support (`guaranteed_capabilities`).
3. Refusal Enforcement: All unsupported topics and must-refuse topics MUST be enforced via the refusal policy and system prompt.
4. Tool Selection: Select ONLY the tools from the tool registry that are relevant to the agent's scope and document formats.
5. System Prompt Excellence: The generated system prompt must be thorough, specifying persona, capabilities, strict grounding on tools/retrieval, citation rules, and refusal handling."""


# ---------------------------------------------------------------------------
# Generation Logic with Controlled Retries
# ---------------------------------------------------------------------------

def _build_generation_prompt(
    requirements: RequirementSpec,
    groundability: GroundabilityReport,
    document_metadata: List[Dict[str, Any]],
    tool_registry: List[ToolSpec],
    error_feedback: Optional[str] = None
) -> str:
    """Constructs the prompt for the generator LLM."""
    tools_summary = "\n".join([
        f"- {t.name}: {t.description} (Parameters: {list(t.parameters.keys()) if t.parameters else 'none'})"
        for t in tool_registry
    ])

    docs_summary = "\n".join([
        f"- {m.get('filename', 'file')}: type={m.get('file_type', 'unknown')}, source={m.get('source', '')}"
        for m in document_metadata
    ]) if document_metadata else "No specific document metadata records provided."

    prompt = (
        f"REQUIREMENTS SPECIFICATION:\n"
        f"- Project Name: {requirements.project_name}\n"
        f"- Intended Audience: {requirements.intended_audience}\n"
        f"- Scope: {requirements.scope}\n"
        f"- Tone: {requirements.tone}\n"
        f"- User Expectations: {requirements.user_expectations}\n"
        f"- Must-Answer Topics: {requirements.must_answer_topics}\n"
        f"- Must-Refuse Topics: {requirements.must_refuse_topics}\n\n"
        f"GROUNDABILITY AUDIT:\n"
        f"- Guaranteed Supported Capabilities: {groundability.guaranteed_capabilities}\n"
        f"- Partially Supported Topics: {groundability.partially_supported_topics}\n"
        f"- Blocked Unsupported Topics: {groundability.blocked_unsupported_topics}\n"
        f"- Recommendations: {groundability.recommendations}\n\n"
        f"DOCUMENT METADATA:\n{docs_summary}\n\n"
        f"AVAILABLE TOOL REGISTRY:\n{tools_summary}\n\n"
    )

    if error_feedback:
        prompt += f"ATTENTION - PREVIOUS ATTEMPT FAILED WITH VALIDATION ERROR:\n{error_feedback}\nPlease correct the output to satisfy all schema constraints.\n\n"

    prompt += (
        "Generate the declarative agent specification by choosing appropriate tools from the registry, "
        "crafting a rigorous system prompt enforcing grounded topics and refusals, and defining "
        "the refusal policy and output contract."
    )
    return prompt


def generate_agent(
    requirements: Union[RequirementSpec, Dict[str, Any], Path, str],
    groundability: Union[GroundabilityReport, Dict[str, Any], Path, str],
    document_metadata: Optional[List[Dict[str, Any]]] = None,
    tool_registry: Optional[List[ToolSpec]] = None,
    output_path: Optional[Union[str, Path]] = "agent.json",
    llm: Optional[Any] = None,
    max_retries: int = 3,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> AgentDefinition:
    """
    Generates a declarative, validated AgentDefinition and saves it to agent.json.
    Handles invalid outputs or missing fields with controlled retries.
    """
    # 1. Resolve requirements
    if isinstance(requirements, RequirementSpec):
        req_spec = requirements
    elif isinstance(requirements, (str, Path)) and Path(requirements).is_file():
        req_spec = RequirementSpec.load_from_json(requirements)
    elif isinstance(requirements, dict):
        req_spec = RequirementSpec.model_validate(requirements)
    else:
        raise ValueError(f"Invalid requirements argument: {requirements}")

    # 2. Resolve groundability report
    if isinstance(groundability, GroundabilityReport):
        ground_rep = groundability
    elif isinstance(groundability, (str, Path)) and Path(groundability).is_file():
        ground_rep = GroundabilityReport.load_from_json(groundability)
    elif isinstance(groundability, dict):
        ground_rep = GroundabilityReport.model_validate(groundability)
    else:
        raise ValueError(f"Invalid groundability argument: {groundability}")

    # 3. Resolve tools and document metadata
    if tool_registry is not None:
        registry = tool_registry
    else:
        from registry import get_default_tool_registry
        dynamic_specs = get_default_tool_registry().to_tool_specs()
        existing_names = {t.name for t in DEFAULT_TOOL_REGISTRY}
        registry = list(DEFAULT_TOOL_REGISTRY) + [t for t in dynamic_specs if t.name not in existing_names]
    registry_map = {t.name: t for t in registry}
    doc_meta = document_metadata or []

    # If document_metadata is empty, populate from requirements.documents_referenced
    if not doc_meta and req_spec.documents_referenced:
        doc_meta = [{"filename": f, "source": f} for f in req_spec.documents_referenced]

    active_llm = llm or get_groq_llm()
    last_error: Optional[Exception] = None
    error_feedback: Optional[str] = None
    attempts_used = 0

    logger.info("Generating Agent Definition for '%s'...", req_spec.project_name)

    # 4. Controlled retry generation loop
    for attempt in range(1, max_retries + 1):
        attempts_used = attempt
        logger.info("Agent generation attempt %d of %d", attempt, max_retries)
        try:
            prompt_content = _build_generation_prompt(
                requirements=req_spec,
                groundability=ground_rep,
                document_metadata=doc_meta,
                tool_registry=registry,
                error_feedback=error_feedback
            )

            # Invoke ChatGroq with structured output
            structured_llm = active_llm.with_structured_output(_LLMGeneratedAgentComponents)
            components: _LLMGeneratedAgentComponents = structured_llm.invoke([
                SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content)
            ])

            # Select and validate tools
            selected_tools: List[ToolSpec] = []
            for t_name in components.selected_tool_names:
                if t_name in registry_map:
                    selected_tools.append(registry_map[t_name].model_copy(update={"enabled": True}))
                else:
                    logger.warning("Agent requested unknown tool '%s'; omitting.", t_name)

            # Ensure document_retriever is present if knowledge base documents exist
            if doc_meta and "document_retriever" in registry_map:
                if not any(t.name == "document_retriever" for t in selected_tools):
                    selected_tools.insert(0, registry_map["document_retriever"].model_copy(update={"enabled": True}))

            # Compose topics lists
            # Supported: strictly guaranteed capabilities
            supported_topics = list(ground_rep.guaranteed_capabilities)
            # Unsupported: blocked groundability topics + explicit must-refuse topics
            unsupported_set = set(ground_rep.blocked_unsupported_topics) | set(req_spec.must_refuse_topics)
            unsupported_topics = sorted(list(unsupported_set))
            partially_supported_topics = list(ground_rep.partially_supported_topics)

            # Build RetrievalConfig
            retrieval_cfg = RetrievalConfig(
                embedding_model=settings.EMBEDDING_MODEL_NAME,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                top_k=settings.TOP_K,
                similarity_threshold=1.35,
                collection_name=collection_name or settings.CHROMA_COLLECTION_NAME,
                persist_directory=str(persist_directory or settings.CHROMA_PERSIST_DIRECTORY),
            )

            # Build RefusalPolicy
            refusal_pol = RefusalPolicy(
                refusal_rules=components.refusal_rules,
                unsupported_topics_handling=components.unsupported_topics_handling,
                must_refuse_topics_handling=components.must_refuse_topics_handling,
                default_refusal_response=components.default_refusal_response,
            )

            # Build OutputContract
            output_contract = OutputContract(
                format="markdown",
                require_citations=True,
                citation_format="[source: {filename}, page: {page_number}]",
                tone_and_style=components.tone_and_style or req_spec.tone,
                constraints=components.output_constraints or [
                    "Only answer using verified retrieved evidence",
                    "Cite sources for all technical facts",
                    "Do not speculate on unsupported topics"
                ]
            )

            # Instantiate and validate final AgentDefinition
            agent_def = AgentDefinition(
                agent_name=components.agent_name or req_spec.project_name,
                description=components.description,
                system_prompt=components.system_prompt,
                audience=req_spec.intended_audience,
                tone=components.tone_and_style or req_spec.tone,
                supported_topics=supported_topics,
                unsupported_topics=unsupported_topics,
                partially_supported_topics=partially_supported_topics,
                retrieval_config=retrieval_cfg,
                selected_tools=selected_tools,
                refusal_policy=refusal_pol,
                output_contract=output_contract,
            )

            # Save to agent.json if requested
            if output_path:
                agent_def.save_to_json(output_path)

            logger.info("Successfully generated and validated AgentDefinition: '%s'", agent_def.agent_name)
            return agent_def

        except Exception as e:
            last_error = e
            if getattr(e, "status_code", None) == 429:
                logger.error("LLM rate limit reached; stopping generation retries.")
                break
            error_feedback = f"Error during attempt {attempt}: {str(e)}"
            logger.warning("Generation attempt %d failed: %s. Retrying...", attempt, str(e))

    attempt_label = "attempt" if attempts_used == 1 else "attempts"
    raise RuntimeError(
        f"Failed to generate valid AgentDefinition after {attempts_used} {attempt_label}. Last error: {last_error}"
    )


if __name__ == "__main__":
    req_path = Path("requirements.json")
    ground_path = Path("groundability_report.json")

    if not req_path.exists() or not ground_path.exists():
        print("Please ensure requirements.json and groundability_report.json exist.")
    else:
        agent = generate_agent(
            requirements=req_path,
            groundability=ground_path,
            output_path="agent.json"
        )
        print("\nGenerated agent.json:")
        print(agent.model_dump_json(indent=2))

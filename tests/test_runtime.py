import pytest
from pathlib import Path
from langchain_core.documents import Document

from config import settings
from src.embeddings.embeddings import get_embedding_model
from src.vectorstore.chroma_store import get_vector_store, add_documents
from generator import (
    AgentDefinition,
    RetrievalConfig,
    RefusalPolicy,
    OutputContract,
    ToolSpec,
)
from agent_loader import load_agent, LoadedAgentBundle
from chat_runner import AgentRuntime, AgentResponse


@pytest.fixture(scope="session")
def setup_two_domain_collections(tmp_path_factory):
    """
    Sets up two completely distinct ChromaDB collections representing two separate domains:
    1. Manufacturing / Machinery Diagnostics
    2. Information Security & Compliance
    """
    chroma_dir = tmp_path_factory.mktemp("runtime_chroma")
    embeddings = get_embedding_model()

    # Domain 1: Machinery
    store_machinery = get_vector_store(
        persist_directory=chroma_dir / "machinery",
        collection_name="machinery_docs",
        embedding_model=embeddings
    )
    machinery_docs = [
        Document(
            page_content="Robotic Cell Alpha operating pressure is 120 bar. Emergency cutoff triggers automatically at 85 Celsius.",
            metadata={"filename": "machinery_manual.pdf", "page_number": 2, "source": "machinery_manual.pdf"}
        ),
        Document(
            page_content="Hydraulic pump primary line operational. Maintenance cycle is 30 days.",
            metadata={"filename": "pumps.csv", "page_number": 1, "row_number": 1, "source": "pumps.csv"}
        )
    ]
    add_documents(machinery_docs, vector_store=store_machinery)

    # Domain 2: InfoSec Compliance
    store_infosec = get_vector_store(
        persist_directory=chroma_dir / "infosec",
        collection_name="infosec_docs",
        embedding_model=embeddings
    )
    infosec_docs = [
        Document(
            page_content="Password Policy: Passwords must be at least 16 characters long and must be rotated every 90 days.",
            metadata={"filename": "infosec_policy.pdf", "page_number": 5, "source": "infosec_policy.pdf"}
        ),
        Document(
            page_content="Security Incident Escalation: Any confirmed or suspected security incident must be reported to security-ops@corp.internal within 60 minutes.",
            metadata={"filename": "incident_response.pdf", "page_number": 12, "source": "incident_response.pdf"}
        )
    ]
    add_documents(infosec_docs, vector_store=store_infosec)

    return {
        "chroma_dir": chroma_dir,
        "store_machinery": store_machinery,
        "store_infosec": store_infosec,
    }


@pytest.fixture
def agent_machinery_def(setup_two_domain_collections):
    info = setup_two_domain_collections
    return AgentDefinition(
        agent_name="Industrial Machinery Diagnostic Assistant",
        description="Assists maintenance operators with machinery operating limits and emergency protocols.",
        system_prompt="You are an industrial machinery diagnostic assistant. Ground all answers strictly in retrieved machine documents. Include citations.",
        audience="Factory operators",
        tone="Precise, technical, safety-critical",
        supported_topics=["Robotic cell pressure and emergency cutoff temperature", "Hydraulic pump maintenance"],
        unsupported_topics=["Password policies", "Human resources", "Rocket trajectory"],
        retrieval_config=RetrievalConfig(
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            chunk_size=1000,
            chunk_overlap=200,
            top_k=2,
            similarity_threshold=1.35,
            collection_name="machinery_docs",
            persist_directory=str(info["chroma_dir"] / "machinery"),
        ),
        selected_tools=[
            ToolSpec(
                name="document_retriever",
                description="Retrieves machine manuals and telemetry.",
                parameters={"query": {"type": "string"}},
                enabled=True
            ),
            ToolSpec(
                name="engineering_calculator",
                description="Calculates temperature units.",
                parameters={"expression": {"type": "string"}},
                enabled=True
            )
        ],
        refusal_policy=RefusalPolicy(
            refusal_rules=["Refuse questions not about factory machinery."],
            unsupported_topics_handling="State that topic is outside machinery scope.",
            must_refuse_topics_handling="Decline immediately.",
            default_refusal_response="I cannot answer this question because it is outside machinery documentation.",
        ),
        output_contract=OutputContract(
            format="markdown",
            require_citations=True,
            citation_format="[source: {filename}, page: {page_number}]",
            tone_and_style="Technical and safety-critical",
            constraints=["Strict document citations mandatory"]
        )
    )


@pytest.fixture
def agent_infosec_def(setup_two_domain_collections):
    info = setup_two_domain_collections
    return AgentDefinition(
        agent_name="Corporate InfoSec Compliance Assistant",
        description="Assists enterprise employees with corporate information security policies and protocols.",
        system_prompt="You are an InfoSec compliance assistant. Answer questions strictly according to corporate security policies. Cite sources.",
        audience="Corporate staff",
        tone="Formal, authoritative, protective",
        supported_topics=["Password length and rotation requirements", "Security incident reporting escalation"],
        unsupported_topics=["Factory machinery", "Hydraulic pumps", "Employee salaries"],
        retrieval_config=RetrievalConfig(
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            chunk_size=1000,
            chunk_overlap=200,
            top_k=2,
            similarity_threshold=1.35,
            collection_name="infosec_docs",
            persist_directory=str(info["chroma_dir"] / "infosec"),
        ),
        selected_tools=[
            ToolSpec(
                name="document_retriever",
                description="Retrieves InfoSec policy documents.",
                parameters={"query": {"type": "string"}},
                enabled=True
            )
        ],
        refusal_policy=RefusalPolicy(
            refusal_rules=["Refuse non-security inquiries."],
            unsupported_topics_handling="State that topic is not in InfoSec policies.",
            must_refuse_topics_handling="Decline immediately.",
            default_refusal_response="I cannot assist with this topic as it is not part of InfoSec policies.",
        ),
        output_contract=OutputContract(
            format="markdown",
            require_citations=True,
            citation_format="[source: {filename}, page: {page_number}]",
            tone_and_style="Formal compliance style",
            constraints=["Must cite policy handbook page numbers"]
        )
    )


# ---------------------------------------------------------------------------
# Test Runtime with Agent 1 (Machinery Domain)
# ---------------------------------------------------------------------------

def test_runtime_agent_1_grounded_answer(setup_two_domain_collections, agent_machinery_def):
    info = setup_two_domain_collections
    bundle = load_agent(agent_source=agent_machinery_def, custom_vector_store=info["store_machinery"])
    runtime = AgentRuntime(bundle)

    assert runtime.agent_def.agent_name == "Industrial Machinery Diagnostic Assistant"
    assert len(runtime.tools) == 2  # document_retriever, engineering_calculator

    # Ask grounded question
    response: AgentResponse = runtime.ask("What is the emergency cutoff temperature for Robotic Cell Alpha?")
    assert not response.is_refusal
    assert "85" in response.answer or "cutoff" in response.answer.lower()
    assert len(response.citations) > 0
    assert response.citations[0].source == "machinery_manual.pdf"
    assert response.citations[0].page_number == 2


def test_runtime_agent_1_refuses_unsupported_question(setup_two_domain_collections, agent_machinery_def):
    info = setup_two_domain_collections
    bundle = load_agent(agent_source=agent_machinery_def, custom_vector_store=info["store_machinery"])
    runtime = AgentRuntime(bundle)

    # Ask out-of-domain question (infosec question sent to machinery agent)
    response: AgentResponse = runtime.ask("What are the corporate password rotation requirements?")
    assert response.is_refusal
    assert "cannot answer" in response.answer.lower() or "outside machinery" in response.answer.lower()
    assert response.citations == []


# ---------------------------------------------------------------------------
# Test Runtime with Agent 2 (InfoSec Domain)
# ---------------------------------------------------------------------------

def test_runtime_agent_2_grounded_answer(setup_two_domain_collections, agent_infosec_def):
    info = setup_two_domain_collections
    bundle = load_agent(agent_source=agent_infosec_def, custom_vector_store=info["store_infosec"])
    runtime = AgentRuntime(bundle)

    assert runtime.agent_def.agent_name == "Corporate InfoSec Compliance Assistant"
    assert len(runtime.tools) == 1  # Only document_retriever

    # Ask grounded question
    response: AgentResponse = runtime.ask("What is the required password length and how often must it be rotated?")
    assert not response.is_refusal
    assert "16" in response.answer
    assert "90" in response.answer
    assert len(response.citations) > 0
    assert response.citations[0].source == "infosec_policy.pdf"
    assert response.citations[0].page_number == 5


def test_runtime_agent_2_refuses_unsupported_question(setup_two_domain_collections, agent_infosec_def):
    info = setup_two_domain_collections
    bundle = load_agent(agent_source=agent_infosec_def, custom_vector_store=info["store_infosec"])
    runtime = AgentRuntime(bundle)

    # Ask out-of-domain question (machinery question sent to infosec agent)
    response: AgentResponse = runtime.ask("What is the operating pressure of the hydraulic pump?")
    assert response.is_refusal
    assert "cannot assist" in response.answer.lower() or "not part of infosec" in response.answer.lower()
    assert response.citations == []


def test_runtime_refuses_fictional_unsupported_topic(setup_two_domain_collections, agent_infosec_def):
    info = setup_two_domain_collections
    bundle = load_agent(agent_source=agent_infosec_def, custom_vector_store=info["store_infosec"])
    runtime = AgentRuntime(bundle)

    response = runtime.ask("How do I navigate a submarine through the Mariana Trench?")
    assert response.is_refusal
    assert response.citations == []

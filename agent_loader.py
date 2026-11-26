"""
ForgeAI Phase 4: Generic Agent Loader
Loads any valid agent.json specification, initializes the corresponding ChromaDB vector store,
loads local HuggingFace embeddings, initializes ChatGroq, and attaches only the tools
specified in the agent definition.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Callable
from pydantic import BaseModel
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_groq import ChatGroq

from config import settings
from src.embeddings.embeddings import get_embedding_model
from src.vectorstore.chroma_store import get_vector_store
from src.llm.groq_client import get_groq_llm
from src.utils.logger import logger
from generator import AgentDefinition, ToolSpec


class BuiltinToolRegistry:
    """Factory creating executable tool implementations matching ToolSpec definitions."""

    @staticmethod
    def create_tool(tool_spec: ToolSpec, vector_store: Any = None) -> Optional[BaseTool]:
        name = tool_spec.name

        # Check Phase 8 Dynamic ToolRegistry first for independent schema-validated tools
        from registry import get_default_tool_registry
        dyn_registry = get_default_tool_registry()
        if dyn_registry.has_tool(name):
            return dyn_registry.to_langchain_tool(name)

        if name == "document_retriever":
            def retrieve_docs(query: str, top_k: int = 4) -> str:
                """Searches the persistent vector database for relevant document chunks."""
                if not vector_store:
                    return "No vector store attached to retrieve documents."
                results = vector_store.similarity_search(query=query, k=top_k)
                if not results:
                    return "No matching document excerpts found."
                formatted = []
                for i, doc in enumerate(results, 1):
                    src = doc.metadata.get("filename", "unknown")
                    pg = doc.metadata.get("page_number", 1)
                    formatted.append(f"[{src} p.{pg}]: {doc.page_content}")
                return "\n\n".join(formatted)

            return StructuredTool.from_function(
                func=retrieve_docs,
                name="document_retriever",
                description=tool_spec.description
            )

        elif name == "tabular_data_inspector":
            def inspect_table(filter_key: str = "", filter_value: str = "", **kwargs) -> str:
                """Inspects tabular telemetry or structured records."""
                if not vector_store:
                    return "Tabular index not connected."
                # Search vector store for tabular row documents
                query = f"{filter_key} {filter_value}".strip() or "telemetry component status"
                results = vector_store.similarity_search(query=query, k=4)
                csv_matches = [d for d in results if d.metadata.get("file_type") == "csv"]
                docs_to_use = csv_matches if csv_matches else results
                formatted = [
                    f"Record (file: {d.metadata.get('filename')}, row: {d.metadata.get('row_number', 'N/A')}):\n{d.page_content}"
                    for d in docs_to_use
                ]
                return "\n---\n".join(formatted) if formatted else "No matching records found."

            return StructuredTool.from_function(
                func=inspect_table,
                name="tabular_data_inspector",
                description=tool_spec.description
            )

        elif name == "engineering_calculator":
            def calculate(expression: str) -> str:
                """Safely evaluates basic mathematical and unit expressions."""
                try:
                    import math
                    allowed_names = {
                        "abs": abs, "round": round, "min": min, "max": max,
                        "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos
                    }
                    # Safe arithmetic evaluation
                    code = compile(expression, "<calculator>", "eval")
                    for name in code.co_names:
                        if name not in allowed_names:
                            raise ValueError(f"Use of '{name}' is restricted in calculator.")
                    result = eval(code, {"__builtins__": {}}, allowed_names)
                    return f"Calculation Result: {expression} = {result}"
                except Exception as e:
                    return f"Calculation Error: {str(e)}"

            return StructuredTool.from_function(
                func=calculate,
                name="engineering_calculator",
                description=tool_spec.description
            )

        elif name == "incident_ticket_creator":
            def create_ticket(title: str, severity: str = "medium", details: str = "") -> str:
                """Generates a logged maintenance or incident ticket."""
                import uuid
                ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
                logger.warning("INCIDENT TICKET CREATED: [%s] (%s): %s", ticket_id, severity.upper(), title)
                return f"Success: Created ticket {ticket_id} with severity '{severity.upper()}'. Details: {details}"

            return StructuredTool.from_function(
                func=create_ticket,
                name="incident_ticket_creator",
                description=tool_spec.description
            )

        elif name == "external_web_search":
            def web_search(query: str) -> str:
                """Simulates external knowledge query."""
                return f"External web search result for: {query} (Offline simulated)"

            return StructuredTool.from_function(
                func=web_search,
                name="external_web_search",
                description=tool_spec.description
            )

        else:
            # Generic fallback tool
            def generic_tool(**kwargs) -> str:
                return f"Tool {name} invoked with {kwargs}"

            return StructuredTool.from_function(
                func=generic_tool,
                name=name,
                description=tool_spec.description
            )


class LoadedAgentBundle:
    """Encapsulates all initialized runtime components for an agent."""
    def __init__(
        self,
        definition: AgentDefinition,
        vector_store: Any,
        embedding_model: Any,
        llm: ChatGroq,
        tools: List[BaseTool],
    ):
        self.definition = definition
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm = llm
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}


def load_agent(
    agent_source: Union[str, Path, Dict[str, Any], AgentDefinition],
    llm: Optional[ChatGroq] = None,
    custom_vector_store: Optional[Any] = None,
) -> LoadedAgentBundle:
    """
    Generic, domain-independent loader that takes any valid AgentDefinition,
    initializes ChromaDB according to the agent's retrieval configuration,
    loads the local embedding model, prepares ChatGroq, and attaches only the
    specified tools.
    """
    # 1. Parse and validate AgentDefinition
    if isinstance(agent_source, AgentDefinition):
        agent_def = agent_source
    elif isinstance(agent_source, (str, Path)):
        p = Path(agent_source)
        if not p.is_file():
            raise FileNotFoundError(f"Agent definition file not found: {p}")
        agent_def = AgentDefinition.load_from_json(p)
    elif isinstance(agent_source, dict):
        agent_def = AgentDefinition.model_validate(agent_source)
    else:
        raise ValueError(f"Invalid agent_source type: {type(agent_source)}")

    logger.info("Loading agent '%s' (v%s)...", agent_def.agent_name, agent_def.version)

    # 2. Initialize embedding model from agent retrieval configuration
    retrieval_cfg = agent_def.retrieval_config
    embedding_model = get_embedding_model(model_name=retrieval_cfg.embedding_model)

    # 3. Initialize ChromaDB vector store
    if custom_vector_store is not None:
        vector_store = custom_vector_store
    else:
        vector_store = get_vector_store(
            persist_directory=retrieval_cfg.persist_directory,
            collection_name=retrieval_cfg.collection_name,
            embedding_model=embedding_model,
        )

    # 4. Initialize ChatGroq LLM
    active_llm = llm or get_groq_llm()

    # 5. Instantiate and attach ONLY the tools specified in the agent definition
    attached_tools: List[BaseTool] = []
    for tool_spec in agent_def.selected_tools:
        if tool_spec.enabled:
            tool_instance = BuiltinToolRegistry.create_tool(tool_spec, vector_store=vector_store)
            if tool_instance:
                attached_tools.append(tool_instance)
                logger.info("Attached tool '%s' to agent '%s'", tool_spec.name, agent_def.agent_name)

    logger.info(
        "Agent '%s' successfully loaded with %d tools and retrieval collection '%s'.",
        agent_def.agent_name,
        len(attached_tools),
        retrieval_cfg.collection_name,
    )

    return LoadedAgentBundle(
        definition=agent_def,
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm=active_llm,
        tools=attached_tools,
    )

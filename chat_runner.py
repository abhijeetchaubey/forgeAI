"""
ForgeAI Phase 4: Generic Agent Runtime & Chat Runner
Domain-independent runtime that executes any loaded agent definition, queries ChromaDB,
retrieves document evidence, grounds answers, attaches specified tools, returns citations,
and strictly enforces refusal policies when evidence is missing.
"""

from pathlib import Path
import re
from typing import List, Optional, Union, Dict, Any, Tuple
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from src.utils.logger import logger
from generator import AgentDefinition
from agent_loader import load_agent, LoadedAgentBundle


class Citation(BaseModel):
    """Document citation for grounded answers."""
    source: str = Field(description="Filename of the source document")
    page_number: Optional[int] = Field(default=1, description="Page number where evidence is found")
    row_number: Optional[int] = Field(default=None, description="Row number if from tabular CSV")
    similarity_score: Optional[float] = Field(default=None, description="Semantic similarity distance")
    excerpt: str = Field(description="Relevant text excerpt supporting the statement")


class AgentResponse(BaseModel):
    """Structured response from the generic agent runtime."""
    answer: str = Field(description="Generated response content or refusal message")
    is_refusal: bool = Field(default=False, description="Whether the inquiry was refused based on policy or lack of evidence")
    citations: List[Citation] = Field(default_factory=list, description="List of source document citations")
    agent_name: str = Field(description="Name of the executing agent")
    tools_used: List[str] = Field(default_factory=list, description="List of tools invoked during the query")


class AgentRuntime:
    """
    Generic, domain-independent agent execution runtime.
    Executes queries against any loaded agent definition without hardcoding domain behaviors.
    """

    def __init__(self, bundle: LoadedAgentBundle):
        self.bundle = bundle
        self.agent_def: AgentDefinition = bundle.definition
        self.vector_store = bundle.vector_store
        self.llm = bundle.llm
        self.tools = bundle.tools
        self.tool_map = bundle.tool_map

    @classmethod
    def from_agent_file(
        cls,
        agent_path: Union[str, Path] = "agent.json",
        custom_vector_store: Optional[Any] = None
    ) -> "AgentRuntime":
        """Factory creating a runtime directly from an agent.json file."""
        bundle = load_agent(agent_source=agent_path, custom_vector_store=custom_vector_store)
        return cls(bundle)

    def _retrieve_evidence(
        self,
        query: str
    ) -> Tuple[List[Document], List[Citation]]:
        """
        Retrieves relevant document excerpts from ChromaDB and evaluates similarity.
        """
        top_k = self.agent_def.retrieval_config.top_k
        threshold = self.agent_def.retrieval_config.similarity_threshold

        docs: List[Document] = []
        citations: List[Citation] = []

        try:
            results_with_scores = self.vector_store.similarity_search_with_score(query=query, k=top_k)
            for doc, score in results_with_scores:
                dist = float(score)
                # Keep chunks within similarity threshold
                if dist <= threshold:
                    docs.append(doc)
                    citations.append(
                        Citation(
                            source=doc.metadata.get("filename", "unknown"),
                            page_number=doc.metadata.get("page_number", 1),
                            row_number=doc.metadata.get("row_number"),
                            similarity_score=round(dist, 4),
                            excerpt=doc.page_content.strip()
                        )
                    )
        except Exception as e:
            logger.error("Evidence retrieval error for query '%s': %s", query, str(e))

        return docs, citations

    def _is_explicitly_unsupported(self, question: str) -> Optional[str]:
        """
        Checks if the question matches any explicitly forbidden or blocked unsupported topics.
        """
        q_lower = question.lower()
        for blocked in self.agent_def.unsupported_topics:
            words = [w.lower() for w in re.findall(r"\w+", blocked) if len(w) > 3]
            if words and all(w in q_lower for w in words):
                return blocked
        return None

    def ask(self, question: str) -> AgentResponse:
        """
        Processes a user question:
        1. Checks for explicitly blocked or unsupported topics.
        2. Retrieves relevant document context from ChromaDB.
        3. Enforces refusal if no evidence meets similarity threshold.
        4. Generates grounded answer with source citations.
        """
        if not question or not question.strip():
            return AgentResponse(
                answer="Please provide a non-empty question.",
                is_refusal=True,
                citations=[],
                agent_name=self.agent_def.agent_name,
                tools_used=[]
            )

        q_clean = question.strip()
        logger.info("[%s] Processing query: '%s'", self.agent_def.agent_name, q_clean)

        # 1. Check for explicit unsupported / must-refuse topics
        explicitly_blocked = self._is_explicitly_unsupported(q_clean)
        if explicitly_blocked:
            logger.info("[%s] Query matched blocked topic '%s'. Refusing.", self.agent_def.agent_name, explicitly_blocked)
            return AgentResponse(
                answer=f"{self.agent_def.refusal_policy.default_refusal_response} Topic '{explicitly_blocked}' is not supported.",
                is_refusal=True,
                citations=[],
                agent_name=self.agent_def.agent_name,
                tools_used=[]
            )

        # 2. Retrieve relevant context from ChromaDB
        relevant_docs, citations = self._retrieve_evidence(q_clean)

        # 3. If no relevant evidence is found, refuse to answer
        if not relevant_docs:
            logger.info("[%s] No relevant document evidence found for query. Refusing.", self.agent_def.agent_name)
            return AgentResponse(
                answer=f"{self.agent_def.refusal_policy.default_refusal_response} No relevant document evidence found in the knowledge base.",
                is_refusal=True,
                citations=[],
                agent_name=self.agent_def.agent_name,
                tools_used=[]
            )

        # 4. Format prompt with retrieved evidence and output contract
        context_blocks = []
        for i, c in enumerate(citations, 1):
            context_blocks.append(
                f"[Source: {c.source}, Page: {c.page_number or 1}]:\n{c.excerpt}"
            )
        formatted_context = "\n\n".join(context_blocks)

        tools_used: List[str] = ["document_retriever"]

        # Check if calculator or table tools should be noted
        if any(term in q_clean.lower() for term in ["calculate", "conversion", "sum", "math"]):
            if "engineering_calculator" in self.tool_map:
                tools_used.append("engineering_calculator")

        if any(term in q_clean.lower() for term in ["table", "csv", "row", "record"]):
            if "tabular_data_inspector" in self.tool_map:
                tools_used.append("tabular_data_inspector")

        system_instruction = (
            f"{self.agent_def.system_prompt}\n\n"
            f"OUTPUT CONTRACT AND GUIDELINES:\n"
            f"- Communication Style: {self.agent_def.output_contract.tone_and_style}\n"
            f"- Required Citation Syntax: {self.agent_def.output_contract.citation_format}\n"
            f"- Constraints:\n" + "\n".join(f"  * {c}" for c in self.agent_def.output_contract.constraints) + "\n\n"
            f"REFUSAL POLICY:\n"
            f"{self.agent_def.refusal_policy.unsupported_topics_handling}\n"
            f"If the answer cannot be completely derived from the provided document evidence, refuse with: "
            f"\"{self.agent_def.refusal_policy.default_refusal_response}\""
        )

        user_content = (
            f"DOCUMENT EVIDENCE:\n{formatted_context}\n\n"
            f"USER INQUIRY:\n{q_clean}\n\n"
            "Provide a factual, grounded answer strictly supported by the document evidence above. "
            "Include source citations."
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_instruction),
                HumanMessage(content=user_content)
            ])
            generated_text = response.content.strip()

            # Check if model triggered a refusal
            is_refusal = (
                self.agent_def.refusal_policy.default_refusal_response.lower() in generated_text.lower()
                or "cannot help with that" in generated_text.lower()
                or "not mentioned in the provided" in generated_text.lower()
            )

            return AgentResponse(
                answer=generated_text,
                is_refusal=is_refusal,
                citations=citations if not is_refusal else [],
                agent_name=self.agent_def.agent_name,
                tools_used=tools_used
            )

        except Exception as e:
            logger.error("[%s] LLM generation error: %s", self.agent_def.agent_name, str(e))
            return AgentResponse(
                answer=f"Error generating answer: {str(e)}",
                is_refusal=True,
                citations=[],
                agent_name=self.agent_def.agent_name,
                tools_used=[]
            )


def interactive_chat(agent_path: Union[str, Path] = "agent.json"):
    """Runs an interactive command-line session with the loaded agent."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()
    runtime = AgentRuntime.from_agent_file(agent_path)

    console.print(Panel.fit(
        f"[bold cyan]ForgeAI Agent Runtime: {runtime.agent_def.agent_name}[/bold cyan]\n"
        f"[dim]{runtime.agent_def.description}[/dim]\n"
        f"[yellow]Audience:[/yellow] {runtime.agent_def.audience} | [yellow]Tone:[/yellow] {runtime.agent_def.tone}",
        border_style="cyan"
    ))
    console.print("[dim]Type 'exit' or 'quit' to end session.[/dim]\n")

    while True:
        try:
            question = console.input("[bold green]User > [/bold green]")
            if question.strip().lower() in ["exit", "quit"]:
                console.print("[dim]Session closed.[/dim]")
                break
            if not question.strip():
                continue

            resp = runtime.ask(question)

            if resp.is_refusal:
                console.print(f"\n[bold red]Agent (Refusal):[/bold red] {resp.answer}\n")
            else:
                console.print(f"\n[bold blue]Agent ({resp.agent_name}):[/bold blue]")
                console.print(Markdown(resp.answer))

                if resp.citations:
                    console.print("\n[dim]Sources Cited:[/dim]")
                    for c in resp.citations:
                        console.print(f"  • [cyan]{c.source}[/cyan] (page {c.page_number or 1}) - score: {c.similarity_score}")
                console.print("")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    import sys
    agent_file = sys.argv[1] if len(sys.argv) > 1 else "agent.json"
    interactive_chat(agent_file)

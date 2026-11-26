"""
ForgeAI Phase 8: Dynamic Tool Registry
Central registry managing independent, schema-validated tools:
  1. csv_lookup
  2. calculator (safe AST-based arithmetic; no arbitrary Python execution)
  3. current_date
  4. open_meteo_weather (keyless public HTTP API)

Exposes tool metadata to the agent generator, filters/selects tools based on
requirements and actual use cases, validates inputs, handles execution errors,
and binds only selected tools into the runtime.
"""

from typing import Dict, List, Optional, Any
import re
from pydantic import BaseModel
from langchain_core.tools import StructuredTool

from src.tools import (
    RegisteredTool,
    csv_lookup_tool,
    calculator_tool,
    current_date_tool,
    open_meteo_tool,
)
from src.utils.logger import logger


class ToolRegistry:
    """Dynamic registry for independent, schema-validated tools."""

    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Registers an independent tool into the registry."""
        if not isinstance(tool, RegisteredTool):
            raise TypeError(f"Expected RegisteredTool, got {type(tool)}")
        self._tools[tool.name] = tool
        logger.debug("Registered tool '%s' in ToolRegistry.", tool.name)

    def unregister(self, name: str) -> None:
        """Removes a tool from the registry."""
        self._tools.pop(name, None)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered in ToolRegistry.")
        return self._tools[name]

    def list_tools(self) -> List[RegisteredTool]:
        return list(self._tools.values())

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_metadata_for_generator(self) -> List[Dict[str, Any]]:
        """
        Exposes tool metadata (name, description, input_schema, output_schema, use_cases)
        to the Agent Generator.
        """
        return [t.to_metadata_dict() for t in self._tools.values()]

    def to_tool_specs(self, tool_names: Optional[List[str]] = None) -> List[Any]:
        """
        Converts registered tools (or a selected subset) into generator.ToolSpec objects.
        """
        from generator import ToolSpec

        selected = (
            [self._tools[n] for n in tool_names if n in self._tools]
            if tool_names is not None
            else list(self._tools.values())
        )
        specs = []
        for t in selected:
            schema = t.input_schema.model_json_schema()
            props = schema.get("properties", {})
            specs.append(
                ToolSpec(
                    name=t.name,
                    description=t.description,
                    parameters=props,
                    enabled=True,
                )
            )
        return specs

    def select_tools_for_requirements(
        self,
        scope: str,
        must_answer_topics: Optional[List[str]] = None,
        user_expectations: str = "",
        document_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Deterministically determines which registry tools are genuinely needed
        for the given requirements and document formats.
        Ensures unrelated tools (e.g., weather or calculator for an HR agent)
        are NOT attached unnecessarily.
        """
        must_answer_topics = must_answer_topics or []
        document_metadata = document_metadata or []

        combined_text = " ".join([
            scope.lower(),
            user_expectations.lower(),
            " ".join(t.lower() for t in must_answer_topics),
        ])

        has_csv_doc = any(
            str(m.get("file_type", "")).lower() == "csv"
            or str(m.get("filename", "")).lower().endswith(".csv")
            for m in document_metadata
        )

        def _has_keyword(keywords: List[str]) -> bool:
            return any(re.search(rf"\b{re.escape(kw)}\b", combined_text) for kw in keywords)

        selected: List[str] = []

        # 1. CSV Lookup: attach if CSV documents exist OR requirements mention CSV/tables/procurement/inventory
        csv_keywords = ["csv", "table", "tabular", "spreadsheet", "procurement", "vendor", "inventory", "telemetry", "row", "catalog"]
        if has_csv_doc or _has_keyword(csv_keywords):
            if self.has_tool("csv_lookup"):
                selected.append("csv_lookup")

        # 2. Calculator: attach ONLY if arithmetic, budgets, totals, pricing, or unit conversions are needed
        calc_keywords = ["calculate", "calculation", "math", "arithmetic", "total", "sum", "budget", "cost", "price", "quote", "discount", "tax", "convert", "conversion", "threshold", "procurement"]
        if _has_keyword(calc_keywords):
            if self.has_tool("calculator"):
                selected.append("calculator")

        # 3. Current Date: attach ONLY if time/date/deadlines/schedules are needed
        date_keywords = ["current date", "today", "deadline", "due date", "schedule", "expiration", "expiry", "days remaining"]
        if _has_keyword(date_keywords):
            if self.has_tool("current_date"):
                selected.append("current_date")

        # 4. Open-Meteo Weather: attach ONLY if weather/climate/forecasting is explicitly in scope
        weather_keywords = ["weather", "forecast", "climate", "meteorolog", "windspeed", "open-meteo", "rain", "humidity"]
        if _has_keyword(weather_keywords):
            if self.has_tool("open_meteo_weather"):
                selected.append("open_meteo_weather")

        return selected

    def execute_tool(self, name: str, **kwargs) -> BaseModel:
        """
        Validates inputs and executes the named tool safely.
        Returns the tool's output_schema instance; never raises on bad tool input.
        """
        tool = self.get_tool(name)
        return tool.run(**kwargs)

    def to_langchain_tool(self, name: str) -> StructuredTool:
        """
        Wraps a RegisteredTool as a LangChain StructuredTool with strict args_schema
        for runtime attachment.
        """
        tool = self.get_tool(name)

        def _invoke(**kwargs) -> str:
            result = tool.run(**kwargs)
            return result.model_dump_json()

        return StructuredTool.from_function(
            func=_invoke,
            name=tool.name,
            description=tool.description,
            args_schema=tool.input_schema,
        )


def get_default_tool_registry() -> ToolRegistry:
    """Creates and returns a ToolRegistry populated with the 4 independent Phase 8 tools."""
    registry = ToolRegistry()
    registry.register(csv_lookup_tool)
    registry.register(calculator_tool)
    registry.register(current_date_tool)
    registry.register(open_meteo_tool)
    return registry

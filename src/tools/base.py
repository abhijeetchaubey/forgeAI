"""
Base definitions for ForgeAI Phase 8 Dynamic Tool Registry.
Defines the RegisteredTool contract with strict Pydantic input/output schemas
and safe execution wrapper.
"""

from typing import Callable, Type, List, Dict, Any
from pydantic import BaseModel, ValidationError


class RegisteredTool(BaseModel):
    """Metadata and execution binding for an independent tool."""
    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    execute: Callable[..., BaseModel]
    use_cases: List[str] = []

    def run(self, **kwargs) -> BaseModel:
        """
        Validates inputs against input_schema and executes the tool safely.
        Returns an instance of output_schema even on validation or runtime errors.
        """
        try:
            validated_input = self.input_schema(**kwargs)
        except ValidationError as ve:
            return self.output_schema(
                success=False,
                error=f"Input validation error for '{self.name}': {ve}"
            )
        try:
            return self.execute(validated_input)
        except Exception as exc:
            return self.output_schema(
                success=False,
                error=f"Execution error in '{self.name}': {exc}"
            )

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Exports JSON-serializable metadata for the agent generator."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema(),
            "use_cases": self.use_cases,
        }

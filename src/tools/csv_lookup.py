"""
Independent Tool 1: CSV Lookup Tool
Safely reads structured CSV files and performs optional column filtering
with strict input/output Pydantic validation.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.tools.base import RegisteredTool


class CSVLookupInput(BaseModel):
    """Input schema for CSV lookup tool."""
    file_path: str = Field(..., min_length=1, description="Path to the CSV file to inspect")
    filter_column: Optional[str] = Field(default=None, description="Optional column name to filter rows by")
    filter_value: Optional[str] = Field(default=None, description="Optional substring/value to match in filter_column (case-insensitive)")
    limit: int = Field(default=20, ge=1, le=200, description="Maximum number of matching rows to return")


class CSVLookupOutput(BaseModel):
    """Output schema for CSV lookup tool."""
    success: bool
    file_path: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None


def execute_csv_lookup(params: CSVLookupInput) -> CSVLookupOutput:
    """Reads and optionally filters a CSV file."""
    target = Path(params.file_path)
    if not target.is_file():
        return CSVLookupOutput(
            success=False,
            file_path=params.file_path,
            error=f"CSV file not found: {params.file_path}"
        )
    if target.suffix.lower() != ".csv":
        return CSVLookupOutput(
            success=False,
            file_path=params.file_path,
            error=f"Invalid file type '{target.suffix}'. Expected a .csv file."
        )

    try:
        with open(target, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            if not columns:
                return CSVLookupOutput(
                    success=False,
                    file_path=str(target),
                    error="CSV file has no header columns."
                )

            if params.filter_column and params.filter_column not in columns:
                return CSVLookupOutput(
                    success=False,
                    file_path=str(target),
                    columns=columns,
                    error=f"Column '{params.filter_column}' not found. Available columns: {columns}"
                )

            matched_rows: List[Dict[str, Any]] = []
            for row in reader:
                if params.filter_column and params.filter_value is not None:
                    cell = str(row.get(params.filter_column, ""))
                    if params.filter_value.lower() not in cell.lower():
                        continue
                matched_rows.append(dict(row))
                if len(matched_rows) >= params.limit:
                    break

            return CSVLookupOutput(
                success=True,
                file_path=str(target),
                columns=columns,
                rows=matched_rows,
                row_count=len(matched_rows),
            )
    except Exception as exc:
        return CSVLookupOutput(
            success=False,
            file_path=params.file_path,
            error=f"Failed to read CSV: {exc}"
        )


csv_lookup_tool = RegisteredTool(
    name="csv_lookup",
    description="Looks up and filters rows from structured CSV files (tables, telemetry, procurement records, inventories).",
    input_schema=CSVLookupInput,
    output_schema=CSVLookupOutput,
    execute=execute_csv_lookup,
    use_cases=["csv", "table", "tabular", "procurement", "inventory", "telemetry", "spreadsheet", "pricing", "records"],
)

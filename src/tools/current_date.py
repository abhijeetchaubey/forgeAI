"""
Independent Tool 3: Current Date & Time Tool
Returns the current date and timestamp in a requested timezone.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional
from pydantic import BaseModel, Field

from src.tools.base import RegisteredTool


class CurrentDateInput(BaseModel):
    """Input schema for current date tool."""
    timezone_name: str = Field(
        default="UTC",
        description="IANA timezone name, e.g. 'UTC', 'America/New_York', 'Asia/Kolkata'"
    )
    date_format: str = Field(
        default="%Y-%m-%d",
        description="strftime format string for the returned date"
    )


class CurrentDateOutput(BaseModel):
    """Output schema for current date tool."""
    success: bool
    current_date: Optional[str] = None
    current_datetime_iso: Optional[str] = None
    day_of_week: Optional[str] = None
    timezone_name: str = "UTC"
    error: Optional[str] = None


def execute_current_date(params: CurrentDateInput) -> CurrentDateOutput:
    """Fetches current date and time in the specified timezone."""
    try:
        tz = timezone.utc if params.timezone_name.upper() == "UTC" else ZoneInfo(params.timezone_name)
        now = datetime.now(tz)
        return CurrentDateOutput(
            success=True,
            current_date=now.strftime(params.date_format),
            current_datetime_iso=now.isoformat(),
            day_of_week=now.strftime("%A"),
            timezone_name=params.timezone_name,
        )
    except Exception as exc:
        return CurrentDateOutput(
            success=False,
            timezone_name=params.timezone_name,
            error=f"Invalid timezone or date format: {exc}",
        )


current_date_tool = RegisteredTool(
    name="current_date",
    description="Returns the current date, day of week, and ISO timestamp for scheduling, deadlines, or date-sensitive queries.",
    input_schema=CurrentDateInput,
    output_schema=CurrentDateOutput,
    execute=execute_current_date,
    use_cases=["date", "today", "deadline", "schedule", "due date", "maintenance", "leave", "holiday", "current time"],
)

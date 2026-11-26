"""
Independent Tool 4: Keyless Public HTTP API Tool (Open-Meteo)
Queries Open-Meteo's keyless public weather forecast API for current weather
conditions at given latitude/longitude coordinates.
"""

import json
from typing import Optional
import urllib.request
import urllib.parse
from pydantic import BaseModel, Field

from src.tools.base import RegisteredTool


class OpenMeteoInput(BaseModel):
    """Input schema for Open-Meteo weather API tool."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees (-180 to 180)")
    location_name: Optional[str] = Field(default=None, description="Optional human-readable location label")


class OpenMeteoOutput(BaseModel):
    """Output schema for Open-Meteo weather API tool."""
    success: bool
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    temperature_celsius: Optional[float] = None
    windspeed_kmh: Optional[float] = None
    weather_code: Optional[int] = None
    observation_time: Optional[str] = None
    error: Optional[str] = None


def execute_open_meteo(params: OpenMeteoInput) -> OpenMeteoOutput:
    """Calls Open-Meteo's keyless public HTTP API with timeout and error handling."""
    query_params = urllib.parse.urlencode({
        "latitude": params.latitude,
        "longitude": params.longitude,
        "current_weather": "true",
    })
    url = f"https://api.open-meteo.com/v1/forecast?{query_params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ForgeAI-ToolRegistry/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status != 200:
                return OpenMeteoOutput(
                    success=False,
                    location=params.location_name,
                    latitude=params.latitude,
                    longitude=params.longitude,
                    error=f"Open-Meteo API returned HTTP {resp.status}",
                )
            payload = json.loads(resp.read().decode("utf-8"))

        current = payload.get("current_weather", {})
        if not current:
            return OpenMeteoOutput(
                success=False,
                location=params.location_name,
                latitude=params.latitude,
                longitude=params.longitude,
                error="Open-Meteo response missing 'current_weather' payload.",
            )

        return OpenMeteoOutput(
            success=True,
            location=params.location_name,
            latitude=payload.get("latitude", params.latitude),
            longitude=payload.get("longitude", params.longitude),
            temperature_celsius=current.get("temperature"),
            windspeed_kmh=current.get("windspeed"),
            weather_code=current.get("weathercode"),
            observation_time=current.get("time"),
        )
    except Exception as exc:
        return OpenMeteoOutput(
            success=False,
            location=params.location_name,
            latitude=params.latitude,
            longitude=params.longitude,
            error=f"Open-Meteo HTTP request failed: {exc}",
        )


open_meteo_tool = RegisteredTool(
    name="open_meteo_weather",
    description="Queries the keyless Open-Meteo public HTTP API for live weather conditions (temperature, windspeed) by latitude and longitude.",
    input_schema=OpenMeteoInput,
    output_schema=OpenMeteoOutput,
    execute=execute_open_meteo,
    use_cases=["weather", "forecast", "temperature", "wind", "climate", "outdoor", "meteorology", "open-meteo"],
)

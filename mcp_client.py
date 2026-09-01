import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
import httpx
import re
import json


load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


client = MultiServerMCPClient(
  {
    "tavily": {
      "transport": "streamable_http",
      "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
    },

    "Aviationstack MCP": {
      "transport": "stdio", # we use stdio transport method for local mcp.
      "command": "uvx",
      "args": [
        "--with",
        "mcp<2",
        "aviationstack-mcp" ## gets the github repo to get the mcp.
      ],
      "env": {
        "AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY
      }
    },

    "weather": {
      "transport": "streamable_http",
      "url": "https://open-meteo.caseyjhand.com/mcp"
    }

  }
)


# Check if the client is connected to all the tools
async def get_all_tools():
  tools = await client.get_tools()
  print("Available tools in the api\n")

  for tool in tools:
    print(f"Name: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Args schema: {tool.args}")
    print("-" * 40)



########################################
# Tavily, Aviation and Weather Tools
########################################


search_tools = None
aviation_tools = {}

async def initialize_mcp():
  global search_tools
  global aviation_tools

  if search_tools is not None and aviation_tools:
    return

  tools = await client.get_tools()

  print("\nAvailbale MCP Tools:\n")

  for tool in tools:
    print(tool.name)

  search_tools = next(
    (tool for tool in tools if tool.name == "tavily_search"),
    None
  )

  aviation_tools = {
    tool.name: tool
    for tool in tools
    if tool.name != "tavily_search"
  }


async def tavily_mcp_search(query:str):
  await initialize_mcp()
  result = await search_tools.ainvoke(
    {
      "query": query
    }
  )
  return result


async def fetch_flight_reference_data():
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}

    for name in ("list_airports", "list_airlines"):
        if name not in tool_map:
            raise RuntimeError(
                f"Tool {name!r} not found. Available: {list(tool_map)}"
            )

    airports_result, airlines_result = await asyncio.gather(
        tool_map["list_airports"].ainvoke({}),
        tool_map["list_airlines"].ainvoke({}),
    )
    return airports_result, airlines_result



async def geocode_city(city: str):
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}

    if "openmeteo_search_locations" not in tool_map:
        raise RuntimeError(f"Tool not found. Available: {list(tool_map)}")

    result = await tool_map["openmeteo_search_locations"].ainvoke({
        "name": city,
        "count": 1
    })
    data = _extract_json(result)
    raw_text = data.get("raw_text", "") if isinstance(data, dict) else str(data)

    name_match = re.search(r"\*\*(.+?)\*\*\s*\(([A-Z]{2})\)", raw_text)
    coord_match = re.search(r"—\s*([\-\d.]+),\s*([\-\d.]+)\s*\|", raw_text)
    tz_match = re.search(r"timezone:\s*([\w/_+\-]+)", raw_text)

    if not name_match or not coord_match:
        raise ValueError(f"Could not geocode city: {city}. Raw response: {raw_text}")

    return {
        "name": name_match.group(1),
        "country": name_match.group(2),
        "latitude": float(coord_match.group(1)),
        "longitude": float(coord_match.group(2)),
        "timezone": tz_match.group(1) if tz_match else "auto",
    }


async def weather_mcp_forecast(city: str, days: int = 5):
    location = await geocode_city(city)

    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}

    if "openmeteo_get_forecast" not in tool_map:
        raise RuntimeError(f"Tool not found. Available: {list(tool_map)}")

    result = await tool_map["openmeteo_get_forecast"].ainvoke({
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "daily_variables": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weather_code",
            "wind_speed_10m_max",
        ],
        "forecast_days": days,
        "timezone": location["timezone"],
    })
    data = _extract_json(result)
    raw_text = data.get("raw_text", "") if isinstance(data, dict) else str(data)

    return _parse_forecast_markdown(location, raw_text)



# =========================
# Multi-city forecast (for country-level itineraries)
# =========================

async def weather_mcp_multi_forecast(cities: list[str], days: int = 5) -> dict[str, str]:
    """
    Fetch forecasts for multiple cities concurrently.
    Returns {city_name: forecast_string_or_error}.
    """
    async def safe_forecast(city: str):
        try:
            return city, await weather_mcp_forecast(city, days=days)
        except Exception as e:
            return city, f"Weather unavailable for {city}: {e}"

    results = await asyncio.gather(*(safe_forecast(c) for c in cities))
    return dict(results)



WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}



def _parse_forecast_markdown(location: dict, raw_text: str) -> str:
    # Matches lines like:
    # **2026-09-01** — temperature_2m_max: 22.3 | temperature_2m_min: 14.7 | precipitation_sum: 0 | weather_code: 3 | wind_speed_10m_max: 14.4
    pattern = re.compile(
        r"\*\*(\d{4}-\d{2}-\d{2})\*\*\s*—\s*"
        r"temperature_2m_max:\s*([\-\d.]+)\s*\|\s*"
        r"temperature_2m_min:\s*([\-\d.]+)\s*\|\s*"
        r"precipitation_sum:\s*([\-\d.]+)\s*\|\s*"
        r"weather_code:\s*(\d+)\s*\|\s*"
        r"wind_speed_10m_max:\s*([\-\d.]+)"
    )

    matches = pattern.findall(raw_text)
    if not matches:
        return f"Forecast for {location['name']}, {location['country']}:\n{raw_text}"

    lines = [f"Forecast for {location['name']}, {location['country']}:"]
    for date, t_max, t_min, precip, code, wind in matches:
        condition = WEATHER_CODES.get(int(code), f"code {code}")
        lines.append(
            f"{date}: {t_min}–{t_max}°C, {condition}, precipitation {precip}mm, wind up to {wind} km/h"
        )
    return "\n".join(lines)


def _extract_json(result):
    if isinstance(result, dict):
        return result

    if isinstance(result, list):
        text_blocks = [
            block.get("text", "") for block in result if isinstance(block, dict)
        ]
        combined = "\n".join(text_blocks)
    else:
        combined = str(result)

    try:
        return json.loads(combined)
    except json.JSONDecodeError:
        pass

    if "Response:" in combined:
        json_str = combined.split("Response:", 1)[1].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return {"raw_text": combined}



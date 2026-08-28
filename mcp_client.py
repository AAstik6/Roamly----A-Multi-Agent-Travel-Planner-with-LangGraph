import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")


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
    }
  }
)

# Check if the client is connected to all the tools
async def get_all_tools():
  tools = await client.get_tools()
  print("Available tools in the api\n")

  for tool in tools:
    print(tool.name)



################################
# Tavily and Aviation Tools
################################

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
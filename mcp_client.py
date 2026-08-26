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
  global search_tool
  global aviation_tools

  if search_tool is not None and aviation_tools:
    return

  tools = await client.get_tools()

  print("\nAvailbale MCP Tools:\n")

  for tool in tools:
    print(tool.name)

  search_tool.next(
    tool
    for tool in tools
    if tool.name == "tavily search"
  )

  aviation_tools = {
    tool.name: tool
    for tool in tools
    if tool.name != "tavily_search"
  }


async def tavily_mcp_search(query:str):
  await initialize_mcp()
  result = await search_tool.ainvoke(
    {
      "query": query
    }
  )
  return result

async def aviation_mcp_call(tool_name: str, tool_args: dict = None):
  tools = await client.get_tools()

  tool = next(
    t for t in tools
    if t.name == tool_name
  )

  result = await tool.ainvoke(
    tool_args or {}
  )
  return result
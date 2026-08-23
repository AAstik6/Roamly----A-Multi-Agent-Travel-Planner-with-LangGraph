import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = MultiServerMCPClient(
  {
    "tavily": {
      "transport" : "streamable_http", ## http is used for remote mcp servers.
      "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
    },
  }
)


## without async/await the program might freeze wating for the I/O operations.
tavily_search_tool = None

async def get_tavily_search_tool():
  global tavily_search_tool
  if tavily_search_tool is not None:
    return

  tools = await client.get_tools() # tells to execute concurrent tasks, don't sit idle.
  print("\nAvailable MCP Tools:")

  for tool in tools:
    print(tool.name)

  tavily_search_tool = next(
    tool
    for tool in tools
    if tool.name == "tavily_search"
  )

# this function can be used to call the tavily_search tool with a query in backend.py
async def tavily_mcp_search(query: str):
    await get_tavily_search_tool()
    result = await tavily_search_tool.ainvoke(
       {
          "query": query
       }
    )
    return result
    #print(result)
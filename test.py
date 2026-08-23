from mcp_client_test import tavily_mcp_search
import asyncio

if __name__ == "__main__":
  query = "Get me all the AI news"
  asyncio.run(tavily_mcp_search(query))
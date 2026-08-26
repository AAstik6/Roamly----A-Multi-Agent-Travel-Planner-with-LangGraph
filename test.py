from mcp_client_test import tavily_mcp_search
import asyncio
from mcp_client import get_all_tools


if __name__ == "__main__":
  asyncio.run(get_all_tools())
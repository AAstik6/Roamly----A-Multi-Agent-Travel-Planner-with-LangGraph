# test.py
import asyncio
from mcp_client import weather_mcp_forecast

if __name__ == "__main__":
    result = asyncio.run(weather_mcp_forecast("London", days=5))
    print(result)
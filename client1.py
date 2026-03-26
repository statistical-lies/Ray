import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import json

async def main():
    client = MultiServerMCPClient(
        {
            "ray-server": {
                "command": "python",
                "args": ["mcp_server1.py"],
                "transport": "stdio",
            }
        }
    )

    print("Connecting to MCP server...")

    tools = await client.get_tools()
    print("Tools loaded:", tools)

    # Call tool
    result = await tools[0].ainvoke({})

    print("\nRaw result:", result)

    # Extract actual content
    try:
        text_output = result[0]["text"]
        parsed = json.loads(text_output)

        print("\n NDVI RESULT (clean):")
        print(parsed)

    except Exception as e:
        print("\n Could not parse result:", e)

asyncio.run(main())
import json
import platform
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp_os_name")

@mcp.tool()
async def get_os_name() -> str:
    """OSの名前を取得します。
    """
    os_name = platform.system()
    return json.dumps({
        "type": "text",
        "text": os_name,
    })

if __name__ == "__main__":
    mcp.run(transport='stdio')

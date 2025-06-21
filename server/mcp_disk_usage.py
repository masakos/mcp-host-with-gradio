import json
import shutil

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp_disk_usage")

@mcp.tool()
async def get_disk_usage() -> str:
    """ディスク使用量情報を取得します。
    """
    total, used, free = shutil.disk_usage("/")
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    usage_percent = (used / total) * 100
    disk_info = {
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "usage_percent": round(usage_percent, 2)
    }
    result_text = (
        f"ディスク使用量:\n"
        f"  総容量: {disk_info['total_gb']} GB\n"
        f"  使用量: {disk_info['used_gb']} GB\n"
        f"  使用率: {disk_info['usage_percent']}%"
    )
    return json.dumps({
        "type": "text",
        "text": result_text,
    })

if __name__ == "__main__":
    mcp.run(transport='stdio')

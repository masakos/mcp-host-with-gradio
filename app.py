import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any

import gradio as gr
from anthropic import Anthropic
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


class MCPClient:
    """個別のMCPサーバーとの接続を管理するクラス"""

    def __init__(self, server_name: str):
        self.server_name = server_name
        self.session = None
        self.exit_stack = None
        self.tools = []
        self.tool_server_map = {}

    async def connect(self, server_path: str) -> str:
        """MCPサーバーに接続し、利用可能なツールを取得"""
        if self.exit_stack:
            await self.exit_stack.aclose()
        self.exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command="python",
            args=[server_path],
            env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        )

        # サーバープロセスを起動し、標準入出力経由でMCPサーバーと非同期に接続しセッションを初期化
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        # サーバーから利用可能なツール一覧を取得
        response = await self.session.list_tools()
        self.tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        self.tool_server_map = {tool.name: self.server_name for tool in response.tools}
        tool_names = [tool["name"] for tool in self.tools]
        return f"{self.server_name}と接続しました。利用可能なツール: {', '.join(tool_names)}"

class MultiMCPManager:
    """複数のMCPサーバーを統合管理し、Claude APIとの連携を行うメインクラス"""

    def __init__(self):
        self.os_client = MCPClient("mcp_os_name")
        self.disk_client = MCPClient("mcp_disk_usage")
        self.anthropic = Anthropic()
        self.all_tools = []
        self.tool_to_client = {}
        self.model_name = "claude-3-7-sonnet-20250219"

    def initialize_servers(self) -> str:
        """全サーバーへの接続"""
        return loop.run_until_complete(self._initialize_servers())

    async def _initialize_servers(self) -> str:
        servers = [
            (self.os_client, "server/mcp_os_name.py"),
            (self.disk_client, "server/mcp_disk_usage.py")
        ]
        tasks = [
            self._connect_client(client, path)
            for client, path in servers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return "\n".join(str(result) for result in results)

    async def _connect_client(self, client: MCPClient, server_path: str) -> str:
        """個別のクライアント接続処理"""
        try:
            result = await client.connect(server_path)
            self.all_tools.extend(client.tools)
            for tool_name in client.tool_server_map:
                self.tool_to_client[tool_name] = client
            return result
        except Exception as e:
            return f"Failed to connect to {server_path} server: {str(e)}"

    def process_message(
            self,
            message: str,
            history: list[dict[str, Any] | ChatMessage]
    ) -> tuple:
        """
        Args:
            message (str): ユーザーが入力した新しいメッセージ（質問など）。
            history (list[dict[str, Any] | ChatMessage]): これまでのチャット履歴（リスト形式）。
        Returns:
            tuple: 更新後のチャット履歴と、入力欄の状態
        """

        new_messages = loop.run_until_complete(self._process_query(message, history))
        # チャット履歴を更新
        updated_history = history + [{"role": "user", "content": message}] + new_messages
        textbox_reset = gr.Textbox(value="")
        return updated_history, textbox_reset

    async def _process_query(
            self,
            message: str,
            history: list[dict[str, Any] | ChatMessage]
    ) -> list[dict[str, Any]]:
        claude_messages = []
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role"), msg.get("content")

            if role in ["user", "assistant", "system"]:
                claude_messages.append({"role": role, "content": content})

        claude_messages.append({"role": "user", "content": message})

        # ユーザーからの質問を使用可能なツール情報を含めて、Claude API用の形式に変換して送信
        response = self.anthropic.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=claude_messages,
            tools=self.all_tools
        )
        result_messages = []

        # Claude APIからの応答を処理
        for content in response.content:
            if content.type == 'text':
                result_messages.append({
                    "role": "assistant",
                    "content": content.text
                })
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                client = self.tool_to_client.get(tool_name)

                # Claude API から使用を提示されたツールを実行
                client = self.tool_to_client.get(tool_name)
                result = await client.session.call_tool(tool_name, tool_args)
                result_text = str(result.content)
                result_messages.append({
                    "role": "assistant",
                    "content": "```\n" + result_text + "\n```",
                    "metadata": {
                        "parent_id": f"result_{tool_name}",
                        "id": f"raw_result_{tool_name}",
                        "title": "Raw Output"
                    }
                })

                # ツールの実行結果を含めて再度Claude API 呼び出し
                claude_messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n"
                        f"{result_text}"
                    )
                })
                next_response = self.anthropic.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=claude_messages,
                )
                if next_response.content and next_response.content[0].type == 'text':
                    result_messages.append({
                        "role": "assistant",
                        "content": next_response.content[0].text
                    })

        return result_messages

manager = MultiMCPManager()

def gradio_interface():
    with gr.Blocks(title="MCP Host Demo") as demo:
        gr.Markdown("# MCP Host Demo")
        # MCPサーバーに接続し、接続状況を表示
        gr.Textbox(
            label="MCP Server 接続状況",
            value=manager.initialize_servers(),
            interactive=False
        )
        chatbot = gr.Chatbot(
            value=[],
            height=500,
            type="messages",
            show_copy_button=True,
            avatar_images=("images/m_.jpeg", "images/robo.jpg"),
        )
        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                label="質問してください。",
                placeholder="Ask about OS information or disk usage",
                scale=4
            )
            clear_btn = gr.Button("Clear Chat", scale=1)

        msg.submit(manager.process_message, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)
    return demo

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY を .env ファイルに設定してください。")

    interface = gradio_interface()
    interface.launch(debug=True)

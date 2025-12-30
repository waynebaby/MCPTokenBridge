"""
MCPTokenBridge
===============

A single-file bridge that exposes an OpenAI-compatible chat completions HTTP API
and a stdin-based MCP agent tool named `hook`. The MCP side wraps chat
completion requests into sampling-friendly responses so VS Code Copilot can
forward conversations.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError


# --------------------------
# Data models / 数据模型
# --------------------------


class ChatMessage(BaseModel):
    """Single chat message record / 单条对话消息."""

    role: str = Field(..., description="Role such as user/assistant/system")
    content: str = Field(..., description="Message content text")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completions request / OpenAI 风格请求."""

    model: str = Field(..., description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    stream: bool = Field(False, description="Whether streaming is requested")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = Field("stop", description="Why generation stopped")


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    object: str = Field("chat.completion", description="Response type")
    choices: List[ChatCompletionChoice]


@dataclass
class MCPResponse:
    """Container for MCP replies / MCP 回复封装."""

    id: Optional[str]
    result: Dict[str, Any]
    error: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        payload = {"jsonrpc": "2.0", "id": self.id, "result": self.result}
        if self.error:
            payload["error"] = self.error
        return json.dumps(payload, ensure_ascii=False)


# --------------------------------
# Core bridge logic / 核心桥接逻辑
# --------------------------------


class MCPTokenBridge:
    """Bridge between HTTP chat completions and MCP stdin tool.

    桥接 HTTP Chat Completions 与 MCP stdin 工具。保持尽量简单明了，方便
    C# 背景的开发者理解。
    """

    def __init__(self, model_name: str = "mcp-bridge-demo") -> None:
        self.model_name = model_name

    def _generate_reply(self, messages: List[ChatMessage]) -> str:
        """Generate a simple echo-style reply.

        生成简单的回声式回复，便于演示。真实场景可替换为模型调用。
        """

        if not messages:
            return "Hello from MCPTokenBridge!"
        return f"Echo: {messages[-1].content}"

    def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        reply_text = self._generate_reply(request.messages)
        assistant_message = ChatMessage(role="assistant", content=reply_text)
        choice = ChatCompletionChoice(index=0, message=assistant_message)
        return ChatCompletionResponse(
            id="mcp-bridge-response",
            model=request.model,
            choices=[choice],
        )

    def handle_mcp_call(self, payload: Dict[str, Any]) -> MCPResponse:
        """Process a single MCP JSON-RPC payload.

        处理单条 MCP JSON-RPC 消息。支持初始化与 `hook` 工具调用。
        """

        message_id: Optional[str] = payload.get("id")
        method = payload.get("method")
        params = payload.get("params", {}) or {}

        if method == "initialize":
            return MCPResponse(
                id=message_id,
                result={
                    "protocolVersion": "2024-05-24",
                    "capabilities": {
                        "tools": {
                            "hook": {
                                "description": "Wrap chat completions for VS Code Copilot",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "model": {"type": "string"},
                                        "messages": {"type": "array"},
                                    },
                                    "required": ["messages", "model"],
                                },
                            }
                        }
                    },
                },
            )

        if method in {"callTool", "tools/call"}:
            tool_name = params.get("name") or params.get("tool")
            if tool_name != "hook":
                return MCPResponse(
                    id=message_id,
                    result={},
                    error={"code": -32000, "message": "Unknown tool"},
                )

            try:
                request = ChatCompletionRequest.model_validate(params.get("arguments", {}))
            except ValidationError as exc:  # bilingual message
                return MCPResponse(
                    id=message_id,
                    result={},
                    error={
                        "code": -32602,
                        "message": f"Invalid hook arguments 参数错误: {exc}",
                    },
                )

            completion = self.chat_completion(request)
            return MCPResponse(id=message_id, result=completion.model_dump())

        return MCPResponse(
            id=message_id,
            result={},
            error={"code": -32601, "message": f"Method not found: {method}"},
        )


# --------------------------
# HTTP API setup / HTTP 接口
# --------------------------

app = FastAPI(title="MCPTokenBridge", version="0.1.0")
bridge = MCPTokenBridge()


@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: ChatCompletionRequest) -> JSONResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported in demo")
    response = bridge.chat_completion(request)
    return JSONResponse(status_code=200, content=response.model_dump())


# ------------------------
# CLI entrypoint / 命令入口
# ------------------------


def run_stdio_loop() -> None:
    """Run a simple MCP stdin loop / 运行 MCP stdin 循环。"""

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            reply = bridge.handle_mcp_call(payload)
        except json.JSONDecodeError:
            reply = MCPResponse(id=None, result={}, error={"code": -32700, "message": "Invalid JSON"})
        sys.stdout.write(reply.to_json() + "\n")
        sys.stdout.flush()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MCPTokenBridge entrypoint")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--stdio", action="store_true", help="Run MCP stdin loop only")
    args = parser.parse_args(argv)

    if args.stdio:
        run_stdio_loop()
        return

    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

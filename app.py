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
import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

# 本地精简版 fastmcp 框架 / lightweight fastmcp-inspired harness
# 只负责 STDIN/STDOUT JSON-RPC 循环，保持与用户要求一致。


class StdioMCPServer:
    """Minimal STDIN/STDOUT server inspired by fastmcp / 精简版 fastmcp 服务."""

    def __init__(self, handler: "MCPTokenBridge") -> None:
        self.handler = handler

    def serve_forever(self) -> None:
        for line in sys.stdin:
            payload = line.strip()
            if not payload:
                continue
            try:
                message = json.loads(payload)
                reply = self.handler.handle_mcp_call(message)
            except json.JSONDecodeError:
                reply = MCPResponse(
                    id=None,
                    result={},
                    error={"code": -32700, "message": "Invalid JSON"},
                )
            sys.stdout.write(reply.to_json() + "\n")
            sys.stdout.flush()


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


@dataclass
class PendingHttpCall:
    """Pending HTTP call waiting for hook / 待处理 HTTP 请求."""

    request: ChatCompletionRequest
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[ChatCompletionResponse] = None
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


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
        self._requests: "Queue[PendingHttpCall]" = Queue()
        self._hook_thread = threading.Thread(
            target=self._hook_worker,
            name="hook-worker",
            daemon=True,
        )
        self._hook_thread.start()

    def _generate_reply(self, messages: List[ChatMessage]) -> str:
        """Generate a simple echo-style reply.

        生成简单的回声式回复，便于演示。真实场景可替换为模型调用。
        """

        if not messages:
            return "Hello from MCPTokenBridge!"
        return f"Echo: {messages[-1].content}"

    def _hook_worker(self) -> None:
        """Dedicated hook thread to process HTTP requests / 固定 Hook 线程轮询队列."""

        while True:
            pending = self._requests.get()
            try:
                completion = self.chat_completion(pending.request)
                pending.response = completion
                pending.headers.update(
                    {
                        "X-MCP-Bridge": "hook",
                        "X-MCP-Model": completion.model,
                    }
                )
            except Exception as exc:  # keep thread alive
                pending.error = f"Hook failure 钩子错误: {exc}"
            finally:
                pending.event.set()

    def submit_chat_request(self, request: ChatCompletionRequest, timeout: float = 30.0) -> Tuple[ChatCompletionResponse, Dict[str, str]]:
        """Queue a chat completion for the hook worker and wait for result.

        将请求送入 Hook 队列并阻塞等待返回，避免阻塞其他 HTTP 请求。
        """

        pending = PendingHttpCall(request=request)
        self._requests.put(pending)
        if not pending.event.wait(timeout=timeout):
            raise TimeoutError("Hook response timed out 钩子响应超时")
        if pending.error:
            raise RuntimeError(pending.error)
        assert pending.response is not None
        return pending.response, pending.headers

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

            # 通过 Hook 线程队列处理，确保工具永不结束 / route through hook queue
            completion, headers = self.submit_chat_request(request)
            result_payload = completion.model_dump()
            result_payload["_headers"] = headers  # expose headers back to caller
            return MCPResponse(id=message_id, result=result_payload)

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
    try:
        response, headers = bridge.submit_chat_request(request)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(status_code=200, content=response.model_dump(), headers=headers)


# ------------------------
# CLI entrypoint / 命令入口
# ------------------------


def run_stdio_loop() -> None:
    """Run a simple MCP stdin loop / 运行 MCP stdin 循环。

    单一入口：启动本进程后即监听 STDIN 并提供 MCP JSON-RPC 服务，
    同时通过后台线程运行 HTTP 服务器。
    """

    StdioMCPServer(bridge).serve_forever()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MCPTokenBridge entrypoint")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help=(
            "HTTP bind address; use 0.0.0.0 to expose all adapters or a specific IP "
            "(e.g., 192.168.55.10) to limit access / HTTP 绑定地址；使用 0.0.0.0 暴露"
            "全部网卡，指定具体 IP（如 192.168.55.10）可限制访问。"
        ),
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args(argv)

    # 启动 HTTP 服务线程 / Start HTTP server thread
    import uvicorn

    server_config = uvicorn.Config("app:app", host=args.host, port=args.port, reload=False, lifespan="on")
    server = uvicorn.Server(server_config)

    http_thread = threading.Thread(target=server.run, name="uvicorn-thread", daemon=True)
    http_thread.start()

    # 主线程运行 MCP STDIN 循环 / MCP loop on main thread
    run_stdio_loop()

    # 如果 STDIN 结束，确保 HTTP 线程退出 / Graceful stop when stdin ends
    if server.should_exit is False:
        server.should_exit = True
    http_thread.join(timeout=1)


if __name__ == "__main__":
    main()

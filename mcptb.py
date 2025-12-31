# -*- coding: utf-8 -*-
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
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import asyncio
import time
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
try:
    from fastmcp import FastMCP, Context  # type: ignore
    FASTMCP_AVAILABLE = True
except Exception:
    FastMCP = None  # type: ignore
    Context = None  # type: ignore
    FASTMCP_AVAILABLE = False

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
            logging.getLogger("mcptb").info(f"STDIO RX: {payload}")
            try:
                message = json.loads(payload)
                reply = self.handler.handle_mcp_call(message)
            except json.JSONDecodeError:
                reply = MCPResponse(
                    id=None,
                    result={},
                    error={"code": -32700, "message": "Invalid JSON"},
                )
            try:
                logging.getLogger("mcptb").info(f"STDIO TX: {reply.to_json()}")
            except Exception:
                logging.getLogger("mcptb").exception("Failed to serialize STDIO reply")
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
class PendingTask:
    """A pending HTTP message to be sampled by the MCP hook.

    由 MCP hook 线程执行采样的待处理 HTTP 消息。
    """

    request: ChatCompletionRequest
    future: asyncio.Future  # resolved with ChatCompletionResponse / 异步结果


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
        # Direct sampling: store ctx when hook starts / 直接采样：在 hook 启动时保存 ctx
        self.ctx: Optional["Context"] = None
        # Store the hook's event loop and internal queue / 保存 hook 的事件循环与内部队列
        self.hook_loop: Optional[asyncio.AbstractEventLoop] = None
        self.queue: Optional[asyncio.Queue[PendingTask]] = None

    # Local generation removed; bridge requires MCP hook
    # 已移除本地生成；桥接器要求 MCP hook。

    # Local hook worker removed; external MCP tool must drain queue
    # 已移除本地 Hook 线程；必须由外部 MCP 工具消费队列。

    async def _process_pending(self, pending: PendingTask) -> None:
        """Process one pending task on the hook loop.

        在 hook 事件循环中处理一条待采样任务。
        """
        assert self.ctx is not None
        req = pending.request
        parts: List[str] = [f"{m.role}: {m.content}" for m in req.messages]
        prompt = "\n".join(parts) if parts else "Hello from MCPTokenBridge!"
        try:
            result = await self.ctx.sample(messages=prompt, max_tokens=256)
            text = getattr(result, "text", "") or ""
            assistant_message = ChatMessage(role="assistant", content=text)
            choice = ChatCompletionChoice(index=0, message=assistant_message)
            completion = ChatCompletionResponse(
                id="mcp-fastmcp-response",
                model=req.model,
                choices=[choice],
            )
            pending.future.set_result(completion)
        except Exception as exc:
            pending.future.set_exception(exc)

    async def _await_future(self, fut: asyncio.Future) -> ChatCompletionResponse:
        return await fut  # helper to await hook-side futures from HTTP loop

    async def submit_via_hook(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Submit a request to the MCP hook queue and await its result.

        将请求提交到 MCP hook 队列，并等待其完成。
        """
        if self.ctx is None or self.hook_loop is None or self.queue is None:
            raise RuntimeError("Hook not active 钩子未运行")
        # Create a future on the hook loop
        fut = asyncio.run_coroutine_threadsafe(
            self._create_future_on_hook(), self.hook_loop
        ).result()
        pending = PendingTask(request=request, future=fut)
        # Enqueue on hook loop
        asyncio.run_coroutine_threadsafe(self.queue.put(pending), self.hook_loop)
        # Await completion by awaiting the hook future via the hook loop
        wrapped = asyncio.run_coroutine_threadsafe(self._await_future(fut), self.hook_loop)
        return await asyncio.wrap_future(wrapped)

    async def _create_future_on_hook(self) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        return loop.create_future()

    # submit_chat_request removed; HTTP calls sample_via_ctx directly
    # 移除 submit_chat_request；HTTP 直接调用 sample_via_ctx。

    # Removed direct local completion; all requests must go through MCP hook
    # 移除本地完成；所有请求必须通过 MCP hook。

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
                                        "input": {
                                            "type": "string",
                                            "description": "Simplest Copilot call: single user input"
                                        },
                                        "model": {"type": "string"},
                                        "messages": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "role": {"type": "string"},
                                                    "content": {"type": "string"}
                                                },
                                                "required": ["role", "content"]
                                            }
                                        }
                                    }
                                },
                            }
                        }
                    },
                },
            )

        # List declared tools per MCP spec / MCP 规范的工具枚举
        if method == "tools/list":
            return MCPResponse(
                id=message_id,
                result={
                    "tools": [
                        {
                            "name": "hook",
                            "description": "Wrap chat completions for VS Code Copilot",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "input": {
                                        "type": "string",
                                        "description": "Simplest Copilot call: single user input"
                                    },
                                    "model": {"type": "string"},
                                    "messages": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "role": {"type": "string"},
                                                "content": {"type": "string"}
                                            },
                                            "required": ["role", "content"]
                                        }
                                    }
                                }
                            },
                        }
                    ]
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

            arguments = params.get("arguments", {})
            # Allow simplest Copilot convention: only "input" string -> convert to request
            if isinstance(arguments, dict) and "input" in arguments and (
                "messages" not in arguments
            ):
                input_text = arguments.get("input", "")
                arguments = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": str(input_text)}],
                    "stream": False,
                }
            # Some MCP clients send a raw string as arguments; accept it
            elif isinstance(arguments, str):
                arguments = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": arguments}],
                    "stream": False,
                }

            try:
                request = ChatCompletionRequest.model_validate(arguments)
            except ValidationError as exc:  # bilingual message
                return MCPResponse(
                    id=message_id,
                    result={},
                    error={
                        "code": -32602,
                        "message": f"Invalid hook arguments 参数错误: {exc}",
                    },
                )

            # In MCP-only direct sampling mode, tools/call is not used here
            # MCP 仅直采样模式下，此处不处理工具调用
            return MCPResponse(
                id=message_id,
                result={},
                error={"code": -32001, "message": "Direct sampling only 仅支持直接采样"},
            )

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

# Basic console logging for inputs/outputs / 控制台日志配置
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger("mcptb")


@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: ChatCompletionRequest) -> JSONResponse:
    logger.info(f"HTTP RX: {request.model_dump()}")
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported in demo")
    if bridge.ctx is None or bridge.hook_loop is None or bridge.queue is None:
        raise HTTPException(status_code=503, detail="Hook not active 钩子未运行")
    try:
        response = await bridge.submit_via_hook(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sampling failed 采样失败: {exc}")
    payload = response.model_dump()
    logger.info(f"HTTP TX: {payload}")
    headers = {"X-MCP-Bridge": "hook", "X-MCP-Model": response.model}
    return JSONResponse(status_code=200, content=payload, headers=headers)


# Health endpoint removed with hook_active logic
# 已移除健康检查端点（随 hook_active 逻辑删除）


# ------------------------
# CLI entrypoint / 命令入口
# ------------------------


def run_stdio_loop() -> None:
    """Run a simple MCP stdin loop / 运行 MCP stdin 循环。

    单一入口：启动本进程后即监听 STDIN 并提供 MCP JSON-RPC 服务，
    同时通过后台线程运行 HTTP 服务器。
    """

    # Require FastMCP; no fallback / 强制使用 FastMCP；不提供回退
    if not FASTMCP_AVAILABLE or FastMCP is None:
        raise RuntimeError("FastMCP is required but not available. Please install fastmcp.")

    server = FastMCP(name="MCPTokenBridge")
    # Bridge requires external MCP hook; no local fallback / 桥接器要求 MCP hook，无本地回退

    @server.tool
    async def hook(
        input: str | None = None,
        messages: list[dict[str, str]] | None = None,
        model: str | None = None,
        ctx: "Context" = None,
    ) -> None:
        """Long-lived hook tool that never returns.

        长生命周期的 hook 工具，不返回结果：
        - English: Loop forever, drain HTTP queue, sample via ctx, set pending events.
        - 中文：无限循环，从 HTTP 队列取请求，调用 ctx 采样，设置待处理项的事件与响应。
        """

        # Capture ctx, create queue, and process tasks / 捕获 ctx，创建队列并处理任务
        bridge.ctx = ctx
        bridge.hook_loop = asyncio.get_running_loop()
        bridge.queue = asyncio.Queue()
        logger.info("FastMCP hook started; keeping session open")

        # Warm-up: try a minimal sample to establish session / 预热：尝试一次最小采样以建立会话
        for attempt in range(1, 21):
            try:
                _ = await ctx.sample(messages="warmup", max_tokens=1)
                logger.info("FastMCP session warm-up complete")
                break
            except Exception as exc:
                logger.warning(f"Warm-up attempt {attempt} failed: {exc}")
                await asyncio.sleep(0.5)
        while True:
            pending = await bridge.queue.get()
            await bridge._process_pending(pending)

    # Try common run methods; error if none / 尝试常见运行方法；若不可用则报错
    if hasattr(server, "run_stdio_server"):
        logger.info("Starting FastMCP stdio server")
        server.run_stdio_server()  # type: ignore
        return
    if hasattr(server, "run_stdio"):
        logger.info("Starting FastMCP run_stdio")
        server.run_stdio()  # type: ignore
        return
    if hasattr(server, "run"):
        logger.info("Starting FastMCP run()")
        server.run()  # type: ignore
        return
    raise RuntimeError("FastMCP server run method not found.")


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

    server_config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        lifespan="on",
        access_log=False,
        log_level="warning",
        use_colors=False,
    )
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

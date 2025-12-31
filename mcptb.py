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
from fastapi.responses import JSONResponse, StreamingResponse
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

# Minimal Anthropic-style input models (for /v1/messages shim)
class AnthropicContentBlock(BaseModel):
    type: str = Field(default="text")
    text: str = Field(default="")

class AnthropicMessage(BaseModel):
    role: str
    # Accept either list of blocks or simple string
    content: Any

class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    stream: bool = Field(False)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = Field("stop", description="Why generation stopped")


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    object: str = Field("chat.completion", description="Response type")
    choices: List[ChatCompletionChoice]


class ChatCompletionStreamChoice(BaseModel):
    """Streaming response choice / 流式响应选项."""
    index: int
    delta: ChatMessage  # Contains role/content increments
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming response / OpenAI 风格流式响应."""
    id: str
    model: str
    object: str = Field("chat.completion.chunk", description="Streaming response type")
    choices: List[ChatCompletionStreamChoice]


@dataclass
class StreamMsg:
    """Streaming queue message: fragment text or sentinel.

    流式队列消息：片段文本或结束哨兵（None）。
    """
    msgFrag: Optional[str]


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
        # Store the hook's event loop and internal queues / 保存 hook 的事件循环与内部队列
        self.hook_loop: Optional[asyncio.AbstractEventLoop] = None
        self.queue: Optional[asyncio.Queue[PendingTask]] = None
        # Per-request streaming bridge queue / 每请求流式桥接队列
        self.stream_queue: Optional[asyncio.Queue[Tuple[asyncio.Queue[StreamMsg], ChatCompletionRequest]]] = None
        # Track whether MCP session is ready / 跟踪 MCP 会话是否已准备好
        self.session_ready: bool = False

    # Local generation removed; bridge requires MCP hook
    # 已移除本地生成；桥接器要求 MCP hook。

    # Local hook worker removed; external MCP tool must drain queue
    # 已移除本地 Hook 线程；必须由外部 MCP 工具消费队列。

    async def _process_pending(self, pending: PendingTask) -> None:
        """Process one pending task on the hook loop.

        在 hook 事件循环中处理一条待采样任务。
        ALL MCP operations happen here in hook thread.
        所有 MCP 操作都在 hook 线程中进行。
        """
        assert self.ctx is not None
        req = pending.request
        parts: List[str] = [f"{m.role}: {m.content}" for m in req.messages]
        prompt = "\n".join(parts) if parts else "Hello from MCPTokenBridge!"
        try:
            logger.debug(f"Hook processing non-stream request: {prompt[:50]}...")
            result = await self.ctx.sample(messages=prompt, max_tokens=256)
            text = getattr(result, "text", "") or ""
            logger.debug(f"Hook got response: {len(text)} chars")
            assistant_message = ChatMessage(role="assistant", content=text)
            choice = ChatCompletionChoice(index=0, message=assistant_message)
            completion = ChatCompletionResponse(
                id="mcp-fastmcp-response",
                model=req.model,
                choices=[choice],
            )
            # Only set the future if it is still pending / 仅在未完成时设置结果
            if not pending.future.cancelled() and not pending.future.done():
                pending.future.set_result(completion)
            else:
                logger.warning("Hook future already completed/cancelled; skipping set_result")
        except Exception as exc:
            logger.error(f"Hook processing error: {exc}", exc_info=True)
            # Only set exception if future is still pending / 仅在未完成时设置异常
            if not pending.future.cancelled() and not pending.future.done():
                pending.future.set_exception(exc)
            else:
                logger.warning("Hook future already completed/cancelled; skipping set_exception")

    async def _process_pending_streaming(self, pending: PendingTask) -> None:
        """Process one pending task with streaming output.

        在 hook 事件循环中处理流式采样任务。
        Note: VS Code Copilot's ctx.sample() may not support streaming,
        so we perform regular sampling and let HTTP layer handle streaming chunks.
        
        注意：所有 MCP 操作都在 hook 线程中执行，不在 HTTP 线程中。
        """
        assert self.ctx is not None
        req = pending.request
        parts: List[str] = [f"{m.role}: {m.content}" for m in req.messages]
        prompt = "\n".join(parts) if parts else "Hello from MCPTokenBridge!"
        try:
            logger.debug(f"Hook processing stream request: {prompt[:50]}...")
            # Use regular sampling (not streaming) to get the full response
            # VS Code Copilot may not support streaming sampling at MCP level
            result = await self.ctx.sample(messages=prompt, max_tokens=256)
            text = getattr(result, "text", "") or ""
            logger.debug(f"Hook got streaming response: {len(text)} chars")
            # Return full text; HTTP layer will split into streaming chunks
            if not pending.future.cancelled() and not pending.future.done():
                pending.future.set_result(text)
            else:
                logger.warning("Hook future already completed/cancelled; skipping set_result (stream)")
        except Exception as exc:
            logger.error(f"Hook streaming error: {exc}", exc_info=True)
            if not pending.future.cancelled() and not pending.future.done():
                pending.future.set_exception(exc)
            else:
                logger.warning("Hook future already completed/cancelled; skipping set_exception (stream)")

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
        return await asyncio.wait_for(asyncio.wrap_future(wrapped), timeout=30.0)

    async def submit_via_hook_streaming(self, request: ChatCompletionRequest) -> str:
        """Submit a request to the MCP hook queue and await streaming result.

        将请求提交到 MCP hook 队列，并等待流式完成。返回文本响应。
        The hook thread decides whether to use streaming based on request.stream flag.
        Hook 线程根据 request.stream 标志决定是否使用流式处理。
        """
        if self.ctx is None or self.hook_loop is None or self.queue is None:
            raise RuntimeError("Hook not active 钩子未运行")
        # Create a future on the hook loop
        fut = asyncio.run_coroutine_threadsafe(
            self._create_future_on_hook(), self.hook_loop
        ).result()
        pending = PendingTask(request=request, future=fut)
        # Enqueue on hook loop - hook will process it based on stream flag
        asyncio.run_coroutine_threadsafe(self.queue.put(pending), self.hook_loop)
        # Await completion with timeout
        wrapped = asyncio.run_coroutine_threadsafe(self._await_future_streaming(fut), self.hook_loop)
        return await asyncio.wait_for(asyncio.wrap_future(wrapped), timeout=30.0)

    async def _await_future_streaming(self, fut: asyncio.Future) -> str:
        return await fut  # helper to await hook-side futures from HTTP loop (streaming)

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

# Streaming policy via env / 通过环境变量控制流式
def streaming_disabled() -> bool:
  
    return False;


@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: ChatCompletionRequest):
    logger.info(f"HTTP RX: {request.model_dump()}")
    
    # Accept clients that send "auto" or empty model by mapping to bridge default
    if not request.model or request.model == "auto":
        request.model = bridge.model_name
    
    # Check if hook is active and session is ready
    if bridge.ctx is None:
        logger.error("Hook not initialized - ctx is None")
        raise HTTPException(status_code=503, detail="Hook not initialized - MCP context missing")
    
    if bridge.hook_loop is None:
        logger.error("Hook not initialized - hook_loop is None")
        raise HTTPException(status_code=503, detail="Hook not initialized - event loop missing")
    
    if bridge.queue is None:
        logger.error("Hook not initialized - queue is None")
        raise HTTPException(status_code=503, detail="Hook not initialized - request queue missing")
    
    if not bridge.session_ready:
        logger.error("MCP session not ready - warmup may have failed")
        raise HTTPException(status_code=503, detail="MCP session not ready - VS Code Copilot not available")
    
    # Handle streaming requests via per-request queue / 通过每请求队列处理流式
    if request.stream and not streaming_disabled():
        # Create per-request out_queue and enqueue into stream bridge
        out_queue: asyncio.Queue[StreamMsg] = asyncio.Queue(maxsize=1024)
        if bridge.stream_queue is None:
            logger.error("Hook not initialized - stream queue is None")
            raise HTTPException(status_code=503, detail="Hook not initialized - stream queue missing")
        await bridge.stream_queue.put((out_queue, request))
        async def stream_generator():
            """Generate OpenAI-compatible streaming chunks.
            
            生成 OpenAI 兼容的流式块。
            """
            try:
                logger.info(f"[STREAM] Starting stream for model {request.model}")
                i = 0
                while True:
                    msg = await out_queue.get()
                    if msg.msgFrag is None:
                        break
                    delta = ChatMessage(role="assistant", content=msg.msgFrag)
                    choice = ChatCompletionStreamChoice(index=0, delta=delta, finish_reason=None)
                    response = ChatCompletionStreamResponse(id=f"mcp-stream-{i}", model=request.model, choices=[choice])
                    i += 1
                    yield f"data: {response.model_dump_json(exclude_none=True)}\n\n"
                final_choice = ChatCompletionStreamChoice(index=0, delta=ChatMessage(role="assistant", content=""), finish_reason="stop")
                final_response = ChatCompletionStreamResponse(id="mcp-stream-final", model=request.model, choices=[final_choice])
                yield f"data: {final_response.model_dump_json(exclude_none=True)}\n\n"
                yield "data: [DONE]\n\n"
                logger.info(f"[STREAM] Stream completed")
            except Exception as exc:
                logger.error(f"[STREAM] Streaming error: {exc}", exc_info=True)
                # Yield error in OpenAI-compatible format
                error_response = {
                    "error": {
                        "message": str(exc),
                        "type": type(exc).__name__
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    # Handle non-streaming: when streaming is enabled, unify via per-request queue
    if not request.stream and not streaming_disabled():
        out_queue: asyncio.Queue[StreamMsg] = asyncio.Queue(maxsize=1024)
        if bridge.stream_queue is None:
            logger.error("Hook not initialized - stream queue is None")
            raise HTTPException(status_code=503, detail="Hook not initialized - stream queue missing")
        await bridge.stream_queue.put((out_queue, request))
        try:
            first = await out_queue.get()
            _end = await out_queue.get()
            text = first.msgFrag or ""
            assistant_message = ChatMessage(role="assistant", content=text)
            choice = ChatCompletionChoice(index=0, message=assistant_message)
            response = ChatCompletionResponse(id="mcp-fastmcp-response", model=request.model, choices=[choice])
        except Exception as exc:
            logger.error(f"[NON-STREAM-Q] Sampling failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Sampling failed: {exc}")
    else:
        # Original non-stream future path
        try:
            if request.stream and streaming_disabled():
                logger.info("Streaming disabled via env; falling back to non-streaming")
            logger.info(f"[NON-STREAM] Starting request for model {request.model}")
            response = await bridge.submit_via_hook(request)
            logger.info(f"[NON-STREAM] Got response: {len(response.choices[0].message.content) if response.choices else 0} chars")
        except Exception as exc:
            logger.error(f"[NON-STREAM] Sampling failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Sampling failed: {exc}")
    
    payload = response.model_dump()
    # Convenience: top-level text for simpler clients
    try:
        payload["text"] = response.choices[0].message.content
    except Exception:
        payload["text"] = ""
    logger.info(f"HTTP TX: {len(payload.get('text', ''))} chars")
    headers = {"X-MCP-Bridge": "hook", "X-MCP-Model": response.model}
    return JSONResponse(status_code=200, content=payload, headers=headers)
# --------------------------
# Anthropic /v1/messages shim
# --------------------------

def _anthropic_to_chat_messages(messages: List[AnthropicMessage]) -> List[ChatMessage]:
    converted: List[ChatMessage] = []
    for m in messages:
        role = m.role
        content = m.content
        if isinstance(content, list):
            # list of blocks
            parts: List[str] = []
            for b in content:
                try:
                    block = AnthropicContentBlock.model_validate(b)
                    if block.type == "text":
                        parts.append(block.text)
                except Exception:
                    continue
            text = "".join(parts)
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        converted.append(ChatMessage(role=role, content=text))
    return converted


@app.post("/v1/messages")
async def anthropic_messages_endpoint(request: AnthropicMessagesRequest):
    # This endpoint mimics Anthropic Messages API using our MCP hook sampling
    logger.info(f"Anthropic HTTP RX: {request.model_dump()}")

    # Check hook/session readiness
    if bridge.ctx is None:
        raise HTTPException(status_code=503, detail="Hook not initialized - MCP context missing")
    if bridge.hook_loop is None:
        raise HTTPException(status_code=503, detail="Hook not initialized - event loop missing")
    if bridge.queue is None:
        raise HTTPException(status_code=503, detail="Hook not initialized - request queue missing")
    if not bridge.session_ready:
        raise HTTPException(status_code=503, detail="MCP session not ready - VS Code Copilot not available")

    # Translate Anthropic-style messages to our ChatMessage list
    chat_messages = _anthropic_to_chat_messages(request.messages)
    chat_req = ChatCompletionRequest(model=request.model, messages=chat_messages, stream=False)

    # If streaming enabled via env, support stream / 若启用流式则支持 stream
    if request.stream and not streaming_disabled():
        out_queue: asyncio.Queue[StreamMsg] = asyncio.Queue(maxsize=1024)
        if bridge.stream_queue is None:
            raise HTTPException(status_code=503, detail="Hook not initialized - stream queue missing")
        await bridge.stream_queue.put((out_queue, chat_req))
        async def anthropic_stream():
            try:
                # Emit SSE events from out_queue fragments

                # message_start
                start_event = {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{int(time.time()*1000)}",
                        "type": "message",
                        "role": "assistant",
                        "model": request.model,
                        "content": [
                            {"type": "text", "text": ""}
                        ],
                    },
                }
                yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"

                # content_block_start
                cbs_event = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {json.dumps(cbs_event)}\n\n"

                # content_block_delta (stream text fragments)
                while True:
                    msg = await out_queue.get()
                    if msg.msgFrag is None:
                        break
                    delta_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": msg.msgFrag},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

                # content_block_stop
                cbst_event = {"type": "content_block_stop", "index": 0}
                yield f"event: content_block_stop\ndata: {json.dumps(cbst_event)}\n\n"

                # message_delta
                md_event = {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}}
                yield f"event: message_delta\ndata: {json.dumps(md_event)}\n\n"

                # message_stop
                stop_event = {"type": "message_stop"}
                yield f"event: message_stop\ndata: {json.dumps(stop_event)}\n\n"

                # done
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(f"Anthropic streaming error: {exc}", exc_info=True)
                err = {"error": {"message": str(exc), "type": type(exc).__name__}}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(anthropic_stream(), media_type="text/event-stream")

    # Non-streaming: return Anthropic-style message object
    try:
        if not request.stream and not streaming_disabled():
            # Unified non-stream via per-request queue
            out_queue: asyncio.Queue[StreamMsg] = asyncio.Queue(maxsize=1024)
            if bridge.stream_queue is None:
                raise HTTPException(status_code=503, detail="Hook not initialized - stream queue missing")
            await bridge.stream_queue.put((out_queue, chat_req))
            first = await out_queue.get()
            _end = await out_queue.get()
            text = first.msgFrag or ""
        else:
            # Fallback to original non-stream
            if request.stream and streaming_disabled():
                logger.info("Anthropic stream requested but disabled via env; using non-stream response")
            completion = await bridge.submit_via_hook(chat_req)
            text = completion.choices[0].message.content if completion.choices else ""
        resp = {
            "id": f"msg_{int(time.time()*1000)}",
            "type": "message",
            "role": "assistant",
            "model": request.model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
        }
        return JSONResponse(status_code=200, content=resp)
    except Exception as exc:
        # Return a safe assistant message instead of 500 to avoid proxy None handling
        # 返回安全的助手消息而非 500，以避免代理层处理 None 失败
        logger.error(f"Anthropic non-streaming error: {exc}", exc_info=True)
        resp = {
            "id": f"msg_{int(time.time()*1000)}",
            "type": "message",
            "role": "assistant",
            "model": request.model,
            "content": [{"type": "text", "text": f"[Error] {exc}"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
        }
        return JSONResponse(status_code=200, content=resp)


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
        bridge.stream_queue = asyncio.Queue()
        logger.info("FastMCP hook started; keeping session open")

        # Warm-up: try a minimal sample to establish session / 预热：尝试一次最小采样以建立会话
        session_established = False
        for attempt in range(1, 31):  # Try up to 30 times
            try:
                logger.info(f"Session warm-up attempt {attempt}/30...")
                _ = await ctx.sample(messages="warmup", max_tokens=1)
                logger.info("✓ FastMCP session warm-up complete - ready for requests")
                session_established = True
                bridge.session_ready = True  # Mark session as ready
                break
            except Exception as exc:
                logger.warning(f"  Warm-up attempt {attempt} failed: {exc}")
                if attempt < 30:
                    await asyncio.sleep(1.0)  # Wait longer between attempts
                else:
                    logger.error(f"  Warm-up attempt {attempt}: {exc}")
        
        if not session_established:
            logger.error("✗ CRITICAL: Failed to establish MCP session after 30 attempts")
            logger.error("  The VS Code Copilot session may not be available")
            logger.error("  This usually means:")
            logger.error("  1. VS Code Copilot is not running")
            logger.error("  2. MCP context is not properly initialized")
            logger.error("  3. Network connection issue")
            # Don't crash - continue anyway and let requests fail with clear errors
            bridge.session_ready = False
        
        logger.info("Hook worker loop started - waiting for HTTP requests")
        while True:
            try:
                # Wait for either normal or streaming request
                normal_task = asyncio.create_task(bridge.queue.get())
                stream_task = asyncio.create_task(bridge.stream_queue.get())
                done, pending_tasks = await asyncio.wait({normal_task, stream_task}, return_when=asyncio.FIRST_COMPLETED)
                for t in pending_tasks:
                    t.cancel()
                task = next(iter(done))
                item = task.result()
                if isinstance(item, PendingTask):
                    logger.debug(f"Hook got NORMAL request: stream={item.request.stream}, model={item.request.model}")
                    if item.request.stream:
                        await bridge._process_pending_streaming(item)
                    else:
                        await bridge._process_pending(item)
                else:
                    out_queue, req = item  # type: ignore
                    parts: List[str] = [f"{m.role}: {m.content}" for m in req.messages]
                    prompt = "\n".join(parts) if parts else "Hello from MCPTokenBridge!"
                    try:
                        text_result = await ctx.sample(messages=prompt, max_tokens=256)
                        full_text = getattr(text_result, "text", "") or ""
                        if req.stream and not streaming_disabled():
                            # Streaming: emit fragments
                            for ch in full_text:
                                await out_queue.put(StreamMsg(msgFrag=ch))
                        else:
                            # Non-stream: emit full then sentinel
                            await out_queue.put(StreamMsg(msgFrag=full_text))
                    finally:
                        await out_queue.put(StreamMsg(msgFrag=None))
            except Exception as exc:
                logger.error(f"Hook worker error: {exc}", exc_info=True)

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

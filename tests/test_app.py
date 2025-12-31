import asyncio
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcptb import (
    ChatCompletionRequest,
    MCPTokenBridge,
    chat_completions_endpoint,
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionChoice,
    bridge as global_bridge,
)

import threading


def start_echo_consumer(b) -> threading.Thread:
    def worker():
        pending = b._requests.get()
        # Build echo response for last message content
        last = ""
        for m in reversed(pending.request.messages):
            if m.content:
                last = m.content
                break
        assistant_message = ChatMessage(role="assistant", content=f"Echo: {last}")
        choice = ChatCompletionChoice(index=0, message=assistant_message)
        completion = ChatCompletionResponse(
            id="test-echo",
            model=pending.request.model,
            choices=[choice],
        )
        pending.response = completion
        pending.headers.update({"X-MCP-Bridge": "hook", "X-MCP-Model": completion.model})
        pending.event.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def test_chat_completion_endpoint_echoes_last_message() -> None:
    payload = ChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "ping"}],
        stream=False,
    )
    # Simulate MCP hook consumer for one request
    start_echo_consumer(global_bridge)
    response = asyncio.run(chat_completions_endpoint(payload))
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["choices"][0]["message"]["content"] == "Echo: ping"
    assert response.headers["X-MCP-Bridge"] == "hook"


def test_mcp_initialize_and_hook_call() -> None:
    bridge = MCPTokenBridge()

    init_reply = bridge.handle_mcp_call({"id": "1", "method": "initialize"})
    init_payload: Dict[str, object] = json.loads(init_reply.to_json())
    assert init_payload["result"]["capabilities"]["tools"]["hook"]["description"]

    arguments = {"model": "demo-model", "messages": [{"role": "user", "content": "hi"}]}
    # Simulate MCP hook consumer for one request
    start_echo_consumer(bridge)
    hook_reply = bridge.handle_mcp_call(
        {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/call",
            "params": {"name": "hook", "arguments": arguments},
        }
    )
    hook_payload: Dict[str, object] = json.loads(hook_reply.to_json())
    assert hook_payload["result"]["choices"][0]["message"]["content"] == "Echo: hi"
    assert hook_payload["result"]["_headers"]["X-MCP-Bridge"] == "hook"


def test_mcp_hook_call_with_input_only() -> None:
    bridge = MCPTokenBridge()

    # Simulate MCP hook consumer for one request
    start_echo_consumer(bridge)
    hook_reply = bridge.handle_mcp_call(
        {
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tools/call",
            "params": {"name": "hook", "arguments": {"input": "hello"}},
        }
    )
    hook_payload: Dict[str, object] = json.loads(hook_reply.to_json())
    assert hook_payload["result"]["choices"][0]["message"]["content"] == "Echo: hello"


def test_mcp_hook_string_arguments() -> None:
    bridge = MCPTokenBridge()

    # Simulate MCP hook consumer for one request
    start_echo_consumer(bridge)
    hook_reply = bridge.handle_mcp_call(
        {
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tools/call",
            "params": {"name": "hook", "arguments": "hello"},
        }
    )
    hook_payload: Dict[str, object] = json.loads(hook_reply.to_json())
    assert hook_payload["result"]["choices"][0]["message"]["content"] == "Echo: hello"
    assert hook_payload["result"]["_headers"]["X-MCP-Bridge"] == "hook"

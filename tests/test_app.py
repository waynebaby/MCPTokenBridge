import asyncio
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcptb import ChatCompletionRequest, MCPTokenBridge, chat_completions_endpoint


def test_chat_completion_endpoint_echoes_last_message() -> None:
    payload = ChatCompletionRequest(
        model="demo-model",
        messages=[{"role": "user", "content": "ping"}],
        stream=False,
    )
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

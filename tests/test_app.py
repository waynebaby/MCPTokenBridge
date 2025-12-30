import json
from typing import Dict

from fastapi.testclient import TestClient

from app import MCPTokenBridge, app


client = TestClient(app)


def test_chat_completion_endpoint_echoes_last_message() -> None:
    payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Echo: ping"


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

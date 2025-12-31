# MCPTokenBridge

## Overview (English)
MCPTokenBridge is a single-file Python bridge that simultaneously runs an OpenAI-compatible `/v1/chat/completions` HTTP endpoint (FastAPI + uvicorn) and a stdin-based MCP tool named `hook`. The `hook` worker thread never stops; it drains an internal queue so web requests are forwarded to MCP and returned with `X-MCP-*` headers. The process has only one entrypoint and starts both services together.

## 概览（中文）
MCPTokenBridge 是一个单文件的 Python 桥接程序，同时运行 OpenAI 风格的 `/v1/chat/completions` HTTP 接口（FastAPI + uvicorn）和基于 STDIN 的 MCP 工具 `hook`。`hook` 工作者线程常驻，持续从队列中取出请求并返回带有 `X-MCP-*` 头的结果；整个流程只有一个入口，启动即同时运行两种服务。

## Getting Started / 快速开始
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run: single entry (HTTP + MCP) / 运行方式：单一入口（HTTP + MCP）
```bash
python mcptb.py --host 0.0.0.0 --port 8000
# MCP stdin is handled on the same process; for example:
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | python mcptb.py --host 0.0.0.0 --port 8000
```

## Network binding guidance / 网络绑定说明
- **English**: Use `--host 0.0.0.0` when you want every network adapter (e.g., `192.168.55.10:8888` and `192.168.1.11:8888`) to be reachable on the same port. Use a specific host (e.g., `--host 192.168.55.10 --port 8888`) to bind only that adapter; other adapters will not accept connections.
- **中文**：当需要让所有网卡（例如 `192.168.55.10:8888` 与 `192.168.1.11:8888`）都可访问时，使用 `--host 0.0.0.0`。若只想绑定某个适配器，请指定具体地址（如 `--host 192.168.55.10 --port 8888`）；此时其他适配器将无法访问。

## Flow (English)
1. HTTP `POST /v1/chat/completions` enqueues the request and waits for the `hook` worker reply.
2. The `hook` tool never terminates; it runs on a fixed background thread and returns MCP headers along with the chat response.
3. **Streaming support** (new): Set `"stream": true` in the request to receive Server-Sent Events (SSE) formatted responses with `Content-Type: text/event-stream`. Chunks are sent character-by-character in the format `data: {json}\n\n` followed by a final `data: [DONE]\n\n`.

## 工作流（中文）
1. HTTP `POST /v1/chat/completions` 会将请求入队，等待 `hook` 工作者返回结果。
2. `hook` 工具常驻后台线程，不会结束，同时返回 MCP 相关的响应头与聊天内容。
3. **流式支持**（新功能）：在请求中设置 `"stream": true` 可以接收 Server-Sent Events (SSE) 格式的响应，Content-Type 为 `text/event-stream`。每个块按字符逐个发送，格式为 `data: {json}\n\n`，最后以 `data: [DONE]\n\n` 结束。

## Project Layout / 目录结构
- `mcptb.py` — single runtime module for both HTTP and MCP modes / 兼作 HTTP 与 MCP 入口的单文件模块
- `requirements.txt` — dependencies / 依赖声明
- `tests/` — pytest suite / 单元测试
- `logs/`, `bin/` — outputs and helper scripts / 输出与辅助脚本

## Streaming API Usage Example / 流式 API 使用示例

### Non-streaming request (traditional) / 非流式请求（传统）
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mcp-bridge-demo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
# Response: {"id":"mcp-fastmcp-response","model":"mcp-bridge-demo","object":"chat.completion",...}
```

### Streaming request with SSE / 流式请求（SSE）
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mcp-bridge-demo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
# Response stream:
# data: {"id":"mcp-stream-0","model":"mcp-bridge-demo","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"H"}}]}
# data: {"id":"mcp-stream-1","model":"mcp-bridge-demo","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"e"}}]}
# ... (more chunks) ...
# data: [DONE]
```

## VS Code MCP configuration (English)
Place `mcp.json` under `.vscode/` (or your global MCP directory) so Copilot Chat launches the combined entrypoint and keeps `hook` alive:
```json
{
  "mcpServers": {
    "mcp-token-bridge": {
      "command": "python",
      "args": ["mcptb.py", "--host", "127.0.0.1", "--port", "8000"],
      "env": {},
      "enabled": true
    }
  }
}
```
- Ensure `python` points to your virtual environment; the MCP stdin server and HTTP server start together on the specified host/port.
- Keep the process running; the `hook` tool never terminates and forwards all chat completions through the queue.

## VS Code MCP 配置（中文）
在 `.vscode/`（或全局 MCP 目录）下放置 `mcp.json`，让 Copilot Chat 启动组合入口并保持 `hook` 持续运行：
```json
{
  "mcpServers": {
    "mcp-token-bridge": {
      "command": "python",
      "args": ["mcptb.py", "--host", "127.0.0.1", "--port", "8000"],
      "env": {},
      "enabled": true
    }
  }
}
```
- 确认 `python` 指向虚拟环境；MCP stdin 与 HTTP 服务器会在指定的主机和端口同时启动。
- 进程需保持常驻；`hook` 工具不会结束，所有 Chat Completion 请求都会通过后台队列转发。

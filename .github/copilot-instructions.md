# Copilot Instructions for MCPTokenBridge

This repo implements a single-module MCP-to-HTTP bridge that exposes OpenAI- and Anthropic-compatible chat endpoints while sampling via a long-lived MCP `hook` tool in VS Code Copilot. The goal is simplicity: one runtime module, strongly typed models, and minimal deps.

## Architecture & Flow
- **Runtime module**: `mcptb.py` (single-file). No package tree.
- **MCP hook**: `FastMCP` server defines a `hook` tool which captures `ctx` and drains an internal queue. All sampling uses `ctx.sample(...)` in the hook loop.
- **HTTP server**: FastAPI app started via `uvicorn` on a background thread from `main()`. Endpoints:
  - `/v1/chat/completions`: OpenAI-style (non-stream JSON only)
  - `/v1/messages`: Anthropic-style shim (non-stream JSON only)
- **Session readiness**: The hook warm-ups the MCP/Copilot session; until ready, HTTP requests return `503`.
- **Streaming policy**: Streaming is disabled globally; `stream=true` requests fall back to non-stream JSON.

## Key Files & Conventions
- `mcptb.py`: strongly typed models via Pydantic (`ChatMessage`, `ChatCompletionRequest/Response`, Anthropic shim types). Dataclasses for MCP responses.
- `AGENTS.md`: project guidelines: single-module runtime, typed code, bilingual comments for public logic, minimal deps in `requirements.txt`.
- `requirements.txt`: `fastapi`, `uvicorn`, `pydantic`, `fastmcp`.
- `tests/`: example async tests and integration helpers; some tests assume an async pytest plugin.

## Requests & Examples
- **OpenAI non-stream**:
  ```json
  {"model":"auto","stream":false,
   "messages":[{"role":"user","content":"Hello"}]}
  ```
  POST to `/v1/chat/completions`.
- **Anthropic non-stream**:
  ```json
  {"model":"auto","stream":false,
   "messages":[{"role":"user","content":"Hello"}]}
  ```
  POST to `/v1/messages`.
- Returns top-level JSON with assistant text in `choices[0].message.content` (OpenAI) or `content[0].text` (Anthropic).

## MCP Integration (Claude Code / Copilot)
- Launch via stdio with `mcp.json`:
  ```json
  {
    "mcpServers": {
      "MCPTokenBridge": {
        "transport": {"type":"stdio"},
        "command": "python",
        "args": ["mcptb.py","--host","0.0.0.0","--port","8000"],
        "env": {"MCPTB_HTTP_TIMEOUT":"none"},
        "cwd": "C:\\ws\\wayne.wang\\MCPTokenBridge"
      }
    }
  }
  ```
- Notes:
  - The bridge requires a live VS Code Copilot session; warm-up may take several attempts.
  - If `503` occurs, ensure Copilot is active and the MCP context is available.

## Timeouts & Env Vars
- `MCPTB_HTTP_TIMEOUT`: server-side per-request wait (seconds). Use `none`/`0`/`infinite` for no timeout.
- Streaming env (`MCPTB_DISABLE_STREAMING`) is ignored; streaming is permanently disabled.

## Developer Workflows
- **Run (Windows)**:
  ```powershell
  python mcptb.py --host 0.0.0.0 --port 8000
  ```
- **Quick HTTP check**:
  ```powershell
  $url = "http://127.0.0.1:8000/v1/messages"
  $body = '{"model":"auto","stream":false,"messages":[{"role":"user","content":"hi"}]}'
  curl -X POST $url -H "Content-Type: application/json" -d $body
  ```
- **Tests**: run `python -m pytest -q`. Async tests may require `pytest-asyncio` or `anyio`.

## Patterns & Pitfalls
- All sampling happens on the MCP hook loop; HTTP handlers submit tasks and await futures.
- Futures are guarded against invalid states (cancelled/done) when setting results.
- Return 503 if MCP hook/session not ready; do not attempt local fallback generation.

## Style
- Strong typing with Pydantic/`dataclass`; minimal dependencies.
- Bilingual comments for public logic and non-obvious flows.
- Preserve single-module layout; auxiliary artifacts in `tests/`, `bin/`, `logs/`.

If any sections are unclear (e.g., your client’s `mcp.json` placement, async test plugin expectations), tell me and I’ll tighten the guidance with concrete examples from your setup.

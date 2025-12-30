# MCPTokenBridge

Single-file Python bridge that offers:

- An OpenAI-compatible `/v1/chat/completions` HTTP endpoint (FastAPI + uvicorn).
- A stdin-based MCP agent exposing a `hook` tool for VS Code Copilot-style chats.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run HTTP API

```bash
python app.py --host 0.0.0.0 --port 8000
```

### Run MCP stdin mode

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | python app.py --stdio
```

## Project Layout

- `app.py` — single runtime module for both HTTP and MCP modes
- `requirements.txt` — dependencies
- `tests/` — pytest suite
- `logs/`, `bin/` — reserved for outputs and helper scripts

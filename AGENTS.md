# MCPTokenBridge Agent Guidelines

- Prefer a **single self-contained Python module** in the repository root for runtime code; auxiliary artifacts (tests, logs, binaries) belong in their own directories (`tests/`, `logs/`, `bin/`).
- Code should be **strongly typed**, using type hints and dataclasses/Pydantic models as appropriate. Avoid fanciful patterns; keep the flow straightforward (the requestor is primarily a C# developer).
- Write concise bilingual comments (Chinese + English) for non-obvious logic and public-facing structures.
- HTTP API should follow **OpenAI-compatible chat completions** semantics; MCP/stdin handling should expose a tool named **`hook`** that wraps chat completion requests into sampling-compatible responses.
- Streaming is **disabled** globally; all `stream=true` requests must return non-stream JSON responses.
- Keep dependencies minimal; declare them in `requirements.txt` alongside the main module.
- Tests live in `tests/` and should exercise both HTTP handlers and MCP translation logic.
- Some tests require an async plugin (e.g., `pytest-asyncio` or `anyio`). Update or skip any streaming-related tests.
- CI or helper scripts may go in `bin/`. Treat `logs/` as writable output; do not place code there.
- When adding documentation, prefer README updates in the repo root.
- Final artifacts must preserve the single-module runtime layout (no sprawling package tree).

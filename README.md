# MCPTokenBridge
----------------

English Version · see Chinese version → [中文版本](#中文版本)


What it is:
- A single‑file bridge that lets you use private subscription power through public endpoints—without leaking secrets.
- OpenAI‑compatible HTTP: `/v1/chat/completions`
- Anthropic‑style shim: `/v1/messages`
- An MCP `hook` that stays alive and drains queues.

Why it exists:
- Convert private tokens (e.g., Copilot/MCP) into safe calls from any OpenAI/Anthropic client.
- Stream replies with SSE; log RX/TX with per‑request GUIDs and full content on stream end.

Killer scenarios:
- Resource sharing: team subscription, dev environments, no real token exposure.
- Multi‑tenant proxy: temporary public tokens, scoped access, secrets sealed.
- Audit + rate‑limit: RX/TX logs per call for monitoring and throttling.
- Third‑party integrations: controlled, expiring access for partners.
- Test/Sandbox: safe non‑prod tokens to validate features.

Quick start (Windows) — MCP‑driven launch only:
```bash
# 1) Create venv and install deps
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt

# 2) Add an MCP config so Copilot starts the bridge
#    VS Code Settings or your MCP config file (example):
```
```json
{
  "mcpServers": {
    "MCPTokenBridge": {
      "transport": {"type": "stdio"},
      "command": "python",
      "args": ["mcptb.py", "--host", "0.0.0.0", "--port", "8000"],
      "env": {"MCPTB_HTTP_TIMEOUT": "none"}
    }
  }
}
```
```bash
# 3) Open Copilot Chat in VS Code to activate MCP (hook warms up)
# 4) Verify HTTP is up (after warm‑up)
curl -s http://127.0.0.1:8000/v1/messages -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

Note: launching mcptb.py directly from the shell won’t attach to Copilot; the MCP must start it.

Force colors (optional):
```bash
$env:MCPTB_FORCE_COLOR="1"
```

Streaming:
- OpenAI SSE `data:` chunks with `choices[].delta`
- Anthropic events: `message_start → content_block_delta → message_stop → [DONE]`
- The bridge collects fragments during streaming; full text is logged at completion.

Endpoints:
- OpenAI non‑stream
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```
- OpenAI stream
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"Stream"}],"stream":true}'
```
- Anthropic non‑stream
```bash
curl -X POST http://127.0.0.1:8000/v1/messages -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```
- Anthropic stream
```bash
curl -N -X POST http://127.0.0.1:8000/v1/messages -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"Stream"}],"stream":true}'
```

MCP setup:
```json
{
  "mcpServers": {
    "MCPTokenBridge": {
      "transport": {"type": "stdio"},
      "command": "python",
      "args": ["mcptb.py", "--host", "0.0.0.0", "--port", "8000"],
      "env": {"MCPTB_HTTP_TIMEOUT": "none"}
    }
  }
}
```

Notes:
- Single‑file runtime (`mcptb.py`), strong typing, minimal deps.
- MCP warm‑up required; until ready, HTTP returns 503.
- GUID‑tagged RX/TX logs; set `MCPTB_FORCE_COLOR=1` for ANSI colors.
- Unified per‑request queues for stream/non‑stream; final TX logs include full content.

WSL2 → Windows:
```bash
ip route | awk '/default/ {print $3}'
grep nameserver /etc/resolv.conf | awk '{print $2}'
```

License:
- Operational clarity over legal boilerplate. Use responsibly.


中文版本
--------

查看英文版本 → [English Version](#english-version)

项目简介：
- 单文件桥接：把私有订阅能力通过公开接口安全暴露。
- OpenAI 兼容 HTTP：`/v1/chat/completions`
- Anthropic 风格兼容：`/v1/messages`
- 常驻 MCP `hook`：持续消费队列并返回结果。

存在意义：
- 将私有 Token（如 Copilot/MCP）转换为可被 OpenAI/Anthropic 客户端调用的安全请求。
- 支持流式 SSE；RX/TX 日志带 GUID，流结束时记录完整文本。

高价值场景：
- 订阅资源共享：团队订阅，开发/测试环境复用，避免暴露真实凭证。
- 多租户安全代理：按项目签发临时公开 Token，控制访问范围，核心凭证不外泄。
- 审计与限流：对每次调用记录 RX/TX（含 GUID），实现精细化监控与限流。
- 第三方集成：为合作方提供可控、带有效期的访问权限，而非企业主 Token。
- 测试与沙箱：在非生产环境用受控公开 Token 安全验证功能。

快速开始（Windows）— 必须由 MCP 驱动启动：
```bash
# 1）创建虚拟环境并安装依赖
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt

# 2）在 VS Code 中配置 MCP，让 Copilot 启动桥接器
```
```json
{
  "mcpServers": {
    "MCPTokenBridge": {
      "transport": {"type": "stdio"},
      "command": "python",
      "args": ["mcptb.py", "--host", "0.0.0.0", "--port", "8000"],
      "env": {"MCPTB_HTTP_TIMEOUT": "none"}
    }
  }
}
```
```bash
# 3）打开 VS Code 的 Copilot Chat 以激活 MCP（hook 预热就绪）
# 4）预热完成后，使用 curl 验证 HTTP 服务
curl -s http://127.0.0.1:8000/v1/messages -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

注意：直接在命令行启动 mcptb.py 无法挂接到 Copilot，必须由 MCP 启动。

可选：在非 TTY 输出中强制彩色日志：
```bash
$env:MCPTB_FORCE_COLOR="1"
```

流式说明：
- OpenAI：SSE `data:` 行携带 `choices[].delta`
- Anthropic：`message_start → content_block_delta → message_stop → [DONE]`
- 桥接器在流式过程中累计片段，结束时输出完整文本到 TX 日志。

接口示例：
- OpenAI 非流式
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```
- OpenAI 流式
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"Stream"}],"stream":true}'
```
- Anthropic 非流式
```bash
curl -X POST http://127.0.0.1:8000/v1/messages -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```
- Anthropic 流式
```bash
curl -N -X POST http://127.0.0.1:8000/v1/messages -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"Stream"}],"stream":true}'
```

MCP 配置：
```json
{
  "mcpServers": {
    "MCPTokenBridge": {
      "transport": {"type": "stdio"},
      "command": "python",
      "args": ["mcptb.py", "--host", "0.0.0.0", "--port", "8000"],
      "env": {"MCPTB_HTTP_TIMEOUT": "none"}
    }
  }
}
```

注意事项：
- 单文件运行（`mcptb.py`）、强类型、最小依赖。
- MCP 会话需预热；未就绪时 HTTP 返回 503。
- RX/TX 日志带 GUID；设置 `MCPTB_FORCE_COLOR=1` 可强制 ANSI 彩色。
- 流式与非流式通过每请求队列统一管理；最终 TX 日志包含完整内容。

WSL2 访问 Windows：
```bash
ip route | awk '/default/ {print $3}'
grep nameserver /etc/resolv.conf | awk '{print $2}'
```

许可证：
- 更关注可运行性与清晰性；请负责任地使用。

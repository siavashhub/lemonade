# MCP Gateway

Lemonade exposes its inference capabilities as a Model Context Protocol (MCP) server, so any MCP-compatible client (Claude Desktop, MCP Inspector, Cursor, the `mcp` Python client, etc.) can call your locally running models as tools.

The gateway implements the **MCP "Streamable HTTP" transport** (spec version `2025-06-18`) with the `tools` capability only. All traffic flows through a single endpoint:

| Endpoint | Status | Notes |
|----------|--------|-------|
| `POST /mcp` | Supported | JSON-RPC 2.0 envelope. Accepts a single message or a batch array. |
| `GET /mcp` | `405 Method Not Allowed` | Server-initiated SSE channel is not supported. |

> **Why a single path?** The MCP specification mandates one endpoint URL per server, so `/mcp` is an intentional exception to Lemonade's quad-prefix convention. The same precedent applies to Ollama (`/api/*`) and Anthropic (`POST /api/messages`).

## Authentication

`/mcp` is treated as a regular API route, so it honors `LEMONADE_API_KEY` exactly like `/api/v1/chat/completions`:

```bash
curl -s http://localhost:13305/mcp \
    -H "Authorization: Bearer $LEMONADE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"ping"}'
```

## Supported methods

| Method | Purpose |
|--------|---------|
| `initialize` | Negotiate protocol version, return server identity and capabilities. |
| `notifications/initialized` | Client acknowledgement; silently accepted. |
| `tools/list` | Return the catalogue of callable tools (with JSON Schemas). |
| `tools/call` | Invoke one of the tools below. |
| `ping` | Liveness probe; returns `{}`. |

## Tools

All tools auto-load (and download, if missing) the requested model on first call, exactly like `POST /v1/chat/completions`. Errors are returned as MCP results with `"isError": true` rather than JSON-RPC errors, matching the spec's guidance for tool failures.

### `lemonade_chat`

Chat completion against any LLM in the registry.

```json
{
  "name": "lemonade_chat",
  "arguments": {
    "model": "Qwen3-0.6B-GGUF",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Summarize MCP in one line."}
    ],
    "max_tokens": 64,
    "temperature": 0.2,
    "tools": [],
    "tool_choice": "auto"
  }
}
```

Returns one text block with the assistant content. If the model emits tool calls, a second text block containing `tool_calls: <json>` is appended.

### `lemonade_embed`

Generate embedding vectors for one or more strings.

```json
{
  "name": "lemonade_embed",
  "arguments": {
    "model": "Qwen3-Embedding-0.6B-GGUF",
    "input": ["hello world", "another sentence"]
  }
}
```

Returns a single text block whose payload is a JSON-stringified `{model, data: [{embedding, index, object}], usage}` (MCP has no first-class embedding content type, so the structured payload is embedded as text).

### `lemonade_transcribe_audio`

Transcribe a base64-encoded audio clip with a Whisper-class model.

```json
{
  "name": "lemonade_transcribe_audio",
  "arguments": {
    "model": "Whisper-Tiny",
    "audio_base64": "UklGR…",
    "filename": "clip.wav",
    "response_format": "verbose_json"
  }
}
```

Returns two text blocks: the bare transcript, followed by the full OpenAI-shaped response (for callers that need timestamps or segments).

### `lemonade_generate_speech`

Synthesize speech from text. Audio is returned inline as base64.

```json
{
  "name": "lemonade_generate_speech",
  "arguments": {
    "model": "Kokoro-82M",
    "input": "Hello from Lemonade.",
    "voice": "af_heart",
    "response_format": "mp3"
  }
}
```

Returns one audio content block (`{"type":"audio", "data":"<base64>", "mimeType":"audio/mpeg"}`).

### `lemonade_generate_image`

Generate one or more PNGs from a prompt. Always returns inline base64 (the gateway forces `response_format=b64_json`).

```json
{
  "name": "lemonade_generate_image",
  "arguments": {
    "model": "Stable-Diffusion-1.5",
    "prompt": "a lemon-shaped car driving across the moon",
    "size": "512x512",
    "n": 1
  }
}
```

Returns one image content block per generated image (`{"type":"image", "data":"<base64>", "mimeType":"image/png"}`).

## Error model

| Code | Meaning |
|------|---------|
| `-32700` | Body was not valid JSON. |
| `-32600` | Request was not a JSON-RPC object (or batch was empty / missing `method`). |
| `-32601` | Unknown JSON-RPC method (e.g. `resources/list`). |
| `-32602` | Invalid `params` for a known method. |
| `-32603` | Internal server error (an exception escaped a handler). |

Tool-level failures (bad arguments, model load errors, backend exceptions) are returned as **successful** JSON-RPC results with `"isError": true` and a text content block describing the failure, so MCP-aware models can self-correct.

## Limitations (MVP)

- No server-initiated SSE (GET /mcp returns 405). Tools return their full result in the POST response.
- No session resumption (`Mcp-Session-Id` header is not issued).
- `resources/*` and `prompts/*` capabilities are not implemented.
- Streaming chat output is not exposed via MCP — `stream=true` is ignored. Use `POST /v1/chat/completions` directly for streamed tokens.

## Quick test with curl

```bash
# 1. Initialize
curl -s http://localhost:13305/mcp -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}'

# 2. List tools
curl -s http://localhost:13305/mcp -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

# 3. Call lemonade_chat
curl -s http://localhost:13305/mcp -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"lemonade_chat","arguments":{"model":"Qwen3-0.6B-GGUF","messages":[{"role":"user","content":"hi"}],"max_tokens":16}}}'
```

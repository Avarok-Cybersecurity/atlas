# Tool Calling & Streaming

Atlas supports OpenAI-compatible **function calling** across three wire formats and full **SSE streaming** (OpenAI + Anthropic conventions). This chapter is the operator reference for running agents against Atlas — how to enable tools, stream responses, handle multi-turn tool results, and recognise the failure modes that used to bite real agents.

## Enable tools

Tool calling is on by default — just include `tools: [...]` in your request. Atlas auto-selects the wire format from the model's `MODEL.toml`. Overriding: `--tool-call-parser <FORMAT>`.

| Parser | Wire format | Models |
|---|---|---|
| `hermes` | `<tool_call>{...}</tool_call>` JSON | Qwen3-VL, Qwen3-Next, MiniMax |
| `qwen3_coder` | XML-in-tool-call with `<function=...><parameter=...>` | Qwen3.5-27B/35B/122B, Nemotron-H, Qwen3.6 |
| `mistral` | JSON after `[TOOL_CALLS]` prefix | Mistral-Small-4 |

All three formats are parsed on the server and emitted to the client as standard OpenAI `tool_calls` blocks — you do not need to handle the wire format in your client.

## Minimal tool-call request

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 512
  }'
```

Response (abridged):

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_00000000",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Multi-turn: sending tool results back

Standard OpenAI pattern — append the assistant's `tool_calls` message and then a `role: "tool"` message with the tool's output:

```json
{
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": null, "tool_calls": [{
      "id": "call_00000000", "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
    }]},
    {"role": "tool", "tool_call_id": "call_00000000", "name": "get_weather",
     "content": "{\"temperature\": 15, \"condition\": \"cloudy\"}"}
  ],
  "tools": [...]
}
```

Atlas expands the multi-turn conversation through the chat template and runs a fresh forward.

## `tool_choice`

| Value | Meaning |
|---|---|
| `"auto"` (default) | Model decides |
| `"none"` | Disable tool calling for this request |
| `"required"` | Force the model to call any tool |
| `{"type": "function", "function": {"name": "X"}}` | Force a specific tool |

`"required"` is implemented via the XGrammar grammar (see [XGrammar](../deep-dives/xgrammar.md)) — the grammar masks the "no-tool-call" path, so the sampler can only produce a valid tool-call opening.

## Streaming

Streaming is enabled with `"stream": true`:

```bash
curl -sN http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"atlas","messages":[...],"stream":true}'
```

Output is standard OpenAI SSE:

```
data: {"choices":[{"delta":{"role":"assistant","content":"Once "}}]}

data: {"choices":[{"delta":{"content":"upon "}}]}

...

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Tool calls stream as `delta.tool_calls` chunks:

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_00000000","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"location"}}]}}]}

...

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

## Anthropic Messages

For clients built against Anthropic's API:

```bash
curl -s http://localhost:8888/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "max_tokens": 500,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming uses Anthropic's event conventions — `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`. Atlas populates `stop_sequence` on `message_delta` when a stop token was hit (fixed in wave-12 — earlier builds left it null).

Tool use on `/v1/messages` uses Anthropic's nested content-block format:

```json
{
  "content": [
    {"type": "text", "text": "Let me check that."},
    {"type": "tool_use", "id": "toolu_...", "name": "get_weather",
     "input": {"location": "Paris"}}
  ],
  "stop_reason": "tool_use"
}
```

## Reasoning / `<think>` blocks

Models that emit `<think>` (Qwen3.5, Nemotron-H, MiniMax) stream reasoning content as a separate channel:

```
data: {"choices":[{"delta":{"reasoning":"Let me think step by step. First, ..."}}]}

data: {"choices":[{"delta":{"reasoning":" the user is asking about..."}}]}

data: {"choices":[{"delta":{"content":"The answer is 42."}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
```

This matches OpenAI's `o1` family convention. Clients that don't parse `reasoning` chunks will ignore them cleanly.

`--max-thinking-budget` caps the total reasoning tokens; `--disable-thinking` strips them entirely. For agent workloads that want reasoning but don't want unbounded think time, a budget of 2048–4096 is typical.

## Vision requests

Qwen3-VL and Qwen3.6 accept images in OpenAI content-parts format:

```json
{
  "model": "atlas",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<DATA>"}}
    ]
  }],
  "max_tokens": 512
}
```

`image_url` accepts `data:` URLs (base64-encoded) or `http(s):` URLs (the server fetches them). Multiple images per message are supported.

## Known pitfalls and how Atlas addresses them

- **Tool-call hallucination inside markdown fences.** Older builds' parsers ate the surrounding fence characters. Fixed in wave-1 (markdown-fence aware parser) + XGrammar grammar enforcement.
- **Broken tool-call XML (Qwen3-coder format).** The parser now tolerates literal `</tool_call>` inside JSON string values, missing `</parameter>` tags, and empty `{}` tool-call bodies (wave-7).
- **Streaming Responses store tool_calls.** The server now persists tool calls to the Responses-API session store mid-stream so multi-turn conversations across the Responses API see them on the next turn (wave-11).
- **Balanced markdown URL parens.** The citation extractor used to choke on Wikipedia URLs containing parentheses; now uses a balanced parser (wave-11).
- **Template-forced thinking false-positive.** Qwen3.6's `<think>\n\n</think>\n\n` template prologue was triggering the reasoning parser; now requires the opening `<think>` to be unclosed (wave-4).
- **Spontaneous `<think>` outside the template position.** Qwen3.6 occasionally emits `<think>` in response mid-stream; Atlas now detects this in all four affected code paths (wave-3).

All of these have regression tests under `crates/spark-server/src/tool_parser.rs` and `reasoning_parser.rs`.

## Running against real agents

Minimum Atlas config for running Claude Code, OpenCode, Cline, or nanobot:

- `--max-seq-len 16384` or higher (agents regularly exceed 4k).
- `--enable-prefix-caching` (massive TTFT win on tool schemas).
- `--scheduling-policy slai` (keeps streaming smooth).
- `--speculative --mtp-quantization nvfp4` if the model supports it (agents are 50%+ tool calls; MTP + constrained decoding = +37% throughput).
- `--auto-compact 0.85` so long agent sessions don't crash into the seq-len wall.

## Files to read

- `crates/spark-server/src/tool_parser.rs` — the three parser impls.
- `crates/spark-server/src/reasoning_parser.rs` — `<think>` detection + extraction.
- `crates/spark-server/src/openai.rs`, `anthropic.rs` — request/response structs.
- `crates/spark-server/src/api.rs` — the HTTP handlers.
- `docs/design/tool-calling-gap-analysis.md` — authoritative tool-call design note.
- [XGrammar deep dive](../deep-dives/xgrammar.md) for the constrained-decoding side.

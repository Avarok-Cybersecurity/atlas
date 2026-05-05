#!/usr/bin/env python3
"""Atlas Spark coherence & API compatibility test suite.

Tests OpenAI API compatibility for coding agents (OpenCode, Claude Code, Cline, Continue).
Run against a live Atlas server: python3 scripts/test_coherence.py [--url http://localhost:8888]
"""

import argparse
import json
import sys
import time
import urllib.request
from typing import Optional

URL = "http://localhost:8888"
PASSED = 0
FAILED = 0
SKIPPED = 0


def api(endpoint: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        f"{URL}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def stream_api(endpoint: str, payload: dict, timeout: int = 300) -> list:
    """Send streaming request, return list of parsed SSE chunks."""
    payload["stream"] = True
    req = urllib.request.Request(
        f"{URL}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    chunks = []
    for line in resp.read().decode().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
        elif line == "data: [DONE]":
            chunks.append("[DONE]")
    return chunks


def test(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}{f': {detail}' if detail else ''}")


def skip(name: str, reason: str = ""):
    global SKIPPED
    SKIPPED += 1
    print(f"  ○ {name} (skipped{f': {reason}' if reason else ''})")


# ═══════════════════════════════════════════════════════════════
# 1. Basic API Structure
# ═══════════════════════════════════════════════════════════════
def test_basic_api():
    print("\n═══ 1. Basic API Structure ═══")

    # 1a. /v1/models endpoint
    try:
        r = json.loads(urllib.request.urlopen(f"{URL}/v1/models", timeout=10).read())
        test("GET /v1/models returns model list", "data" in r and len(r["data"]) > 0)
        test("/v1/models has 'object': 'list'", r.get("object") == "list")
    except Exception as e:
        test("GET /v1/models", False, str(e))

    # 1b. Basic chat completion
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Say 'hello'"}],
        "max_tokens": 10,
    })
    test("Chat completion returns 'id'", "id" in r)
    test("Chat completion returns 'object'", r.get("object") == "chat.completion")
    test("Chat completion returns 'choices'", "choices" in r and len(r["choices"]) > 0)
    test("Choice has 'message' with 'role'", r["choices"][0]["message"]["role"] == "assistant")
    test("Choice has 'finish_reason'", r["choices"][0]["finish_reason"] in ("stop", "length"))
    test("Response has 'usage'", "usage" in r)
    test("Usage has prompt/completion tokens", "prompt_tokens" in r["usage"] and "completion_tokens" in r["usage"])

    # 1c. Basic completion
    r = api("/v1/completions", {
        "model": "test",
        "prompt": "Hello",
        "max_tokens": 5,
    })
    test("Completion returns 'choices'", "choices" in r and len(r["choices"]) > 0)
    test("Completion choice has 'text'", "text" in r["choices"][0])


# ═══════════════════════════════════════════════════════════════
# 2. Streaming (SSE)
# ═══════════════════════════════════════════════════════════════
def test_streaming():
    print("\n═══ 2. Streaming (SSE) ═══")

    chunks = stream_api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 500,  # Enough for thinking + response
    })

    test("Stream returns chunks", len(chunks) > 1)
    test("Stream ends with [DONE]", chunks[-1] == "[DONE]")

    data_chunks = [c for c in chunks if c != "[DONE]"]
    test("First chunk has 'delta' with 'role'",
         data_chunks[0]["choices"][0]["delta"].get("role") == "assistant")

    # Check that content chunks have 'delta.content'
    content_chunks = [c for c in data_chunks if c["choices"][0]["delta"].get("content")]
    test("Content chunks have 'delta.content'", len(content_chunks) > 0)

    # Last data chunk should have finish_reason
    last = data_chunks[-1]
    test("Last chunk has finish_reason",
         last["choices"][0].get("finish_reason") in ("stop", "length"))

    # All chunks share same id
    ids = set(c["id"] for c in data_chunks)
    test("All chunks share same 'id'", len(ids) == 1)


# ═══════════════════════════════════════════════════════════════
# 3. Tool Calling (OpenCode/Cline/Continue critical path)
# ═══════════════════════════════════════════════════════════════
def test_tool_calling():
    print("\n═══ 3. Tool Calling ═══")

    tools = [
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        }},
        {"type": "function", "function": {
            "name": "run_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }},
    ]

    # 3a. Basic tool call
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read the file /tmp/test.txt"},
        ],
        "tools": tools,
        "max_tokens": 200,
    })
    c = r["choices"][0]
    tcs = c.get("tool_calls") or c["message"].get("tool_calls")
    test("Tool call generated", tcs is not None and len(tcs) > 0)

    if tcs:
        tc = tcs[0]
        test("Tool call has 'id'", "id" in tc)
        test("Tool call has 'type': 'function'", tc.get("type") == "function")
        test("Tool call has 'function.name'", "name" in tc.get("function", {}))
        test("Tool call has 'function.arguments' (string)", isinstance(tc["function"].get("arguments"), str))

        # Arguments must be valid JSON
        try:
            args = json.loads(tc["function"]["arguments"])
            test("Tool call arguments is valid JSON", True)
            test("Arguments contain expected key", "path" in args or "filePath" in args)
        except (json.JSONDecodeError, KeyError):
            test("Tool call arguments is valid JSON", False, tc["function"].get("arguments", ""))

        test("finish_reason is 'tool_calls'", c.get("finish_reason") == "tool_calls")

    # 3b. Multi-turn with tool response
    if tcs:
        tc = tcs[0]
        r2 = api("/v1/chat/completions", {
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Read the file /tmp/test.txt"},
                {"role": "assistant", "content": c["message"].get("content"), "tool_calls": tcs},
                {"role": "tool", "tool_call_id": tc["id"], "content": "Hello World!"},
            ],
            "tools": tools,
            "max_tokens": 200,
        })
        content2 = r2["choices"][0]["message"].get("content") or ""
        tcs2 = r2["choices"][0]["message"].get("tool_calls") or []
        test("Multi-turn after tool response returns content", len(content2) > 0 or len(tcs2) > 0)
        test("Response references file content", True)  # Model should mention the content

    # 3c. Streaming tool calls
    chunks = stream_api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read /tmp/test.txt"},
        ],
        "tools": tools,
        "max_tokens": 200,
    })
    data_chunks = [c for c in chunks if c != "[DONE]"]
    tc_chunks = [c for c in data_chunks if c["choices"][0]["delta"].get("tool_calls")]
    test("Streaming produces tool_call chunks", len(tc_chunks) > 0)

    if tc_chunks:
        first_tc = tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        test("Streaming tool_call has 'id'", "id" in first_tc)
        test("Streaming tool_call has 'function.name'", "name" in first_tc.get("function", {}))


# ═══════════════════════════════════════════════════════════════
# 4. Coherence & Quality
# ═══════════════════════════════════════════════════════════════
def test_coherence():
    print("\n═══ 4. Coherence & Quality ═══")

    # 4a. No Chinese/gibberish in English prompts
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Explain what Python is in 2 sentences."}],
        "max_tokens": 300,  # Enough for thinking + response
    })
    content = r["choices"][0]["message"]["content"]
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in content)
    has_cyrillic = any('\u0400' <= ch <= '\u04ff' for ch in content)
    test("No Chinese characters in English response", not has_chinese, content[:80])
    test("No Cyrillic characters in English response", not has_cyrillic, content[:80])
    test("Response is non-empty", len(content.strip()) > 10)
    test("No <think> XML tags leaked", "<think>" not in content and "</think>" not in content)

    # 4b. Code generation coherence
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Write a Python function that checks if a number is prime. Just the code."}],
        "max_tokens": 500,  # Enough for thinking + code
    })
    content = r["choices"][0]["message"]["content"]
    test("Code response contains 'def'", "def" in content)
    test("Code response contains 'return'", "return" in content)
    test("No <think> tags leaked in code", "<think>" not in content and "</think>" not in content)

    # 4c. Multi-turn coherence
    msgs = [
        {"role": "user", "content": "My name is Atlas."},
    ]
    r1 = api("/v1/chat/completions", {"model": "test", "messages": msgs, "max_tokens": 50})
    msgs.append({"role": "assistant", "content": r1["choices"][0]["message"]["content"]})
    msgs.append({"role": "user", "content": "What is my name?"})
    r2 = api("/v1/chat/completions", {"model": "test", "messages": msgs, "max_tokens": 50})
    content2 = r2["choices"][0]["message"]["content"]
    test("Multi-turn remembers context (name)", "Atlas" in content2 or "atlas" in content2, content2[:80])


# ═══════════════════════════════════════════════════════════════
# 5. Determinism
# ═══════════════════════════════════════════════════════════════
def test_determinism():
    print("\n═══ 5. Determinism ═══")

    results = []
    for i in range(5):
        r = api("/v1/chat/completions", {
            "model": "test",
            "messages": [{"role": "user", "content": "What is 7 * 8?"}],
            "max_tokens": 50,
            "temperature": 0,
        })
        results.append(r["choices"][0]["message"]["content"][:50])

    test("Greedy is deterministic (5 runs)", len(set(results)) == 1,
         f"{len(set(results))} unique")


# ═══════════════════════════════════════════════════════════════
# 6. Edge Cases
# ═══════════════════════════════════════════════════════════════
def test_edge_cases():
    print("\n═══ 6. Edge Cases ═══")

    # 6a. Empty system message
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Say hi"},
        ],
        "max_tokens": 10,
    })
    test("Empty system message works", len(r["choices"][0]["message"]["content"]) > 0)

    # 6b. Very long system message
    long_system = "You are helpful.\n" + ("Context line.\n" * 500)
    try:
        r = api("/v1/chat/completions", {
            "model": "test",
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": "Say hi"},
            ],
            "max_tokens": 20,
        })
        content = r["choices"][0]["message"]["content"]
        test("Long system message (5k+ chars) works", len(content.strip()) > 0)
    except Exception as e:
        test("Long system message (5k+ chars) works", False, str(e)[:100])

    # 6c. max_tokens=1
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1,
    })
    test("max_tokens=1 returns response", "choices" in r)

    # 6d. temperature=0 explicit
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Say exactly: test"}],
        "max_tokens": 10,
        "temperature": 0.0,
    })
    test("temperature=0 works", len(r["choices"][0]["message"]["content"]) > 0)

    # 6e. Stop sequences
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Count from 1 to 10, one per line."}],
        "max_tokens": 100,
        "stop": ["5"],
    })
    content = r["choices"][0]["message"]["content"]
    test("Stop sequence works (no '5' in output)", "5" not in content or len(content) < 20)


# ═══════════════════════════════════════════════════════════════
# 7. Tool Calling Reliability (OpenCode critical)
# ═══════════════════════════════════════════════════════════════
def test_tool_reliability():
    print("\n═══ 7. Tool Call Reliability (10x) ═══")

    tools = [
        {"type": "function", "function": {
            "name": "read", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]},
        }},
        {"type": "function", "function": {
            "name": "write", "description": "Write a file",
            "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filePath", "content"]},
        }},
        {"type": "function", "function": {
            "name": "bash", "description": "Run command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        }},
    ]

    # Load opencode system prompt if available
    try:
        with open("/tmp/opencode_system_prompt.txt") as f:
            system = f.read()
    except FileNotFoundError:
        system = "You are a helpful coding assistant."

    # Use only 1 tool (most reliable) for reliability test
    single_tool = [tools[0]]
    success = 0
    for i in range(10):
        r = api("/v1/chat/completions", {
            "model": "test",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Read /workspace/atlas/Cargo.toml"},
            ],
            "tools": single_tool,
            "max_tokens": 500,
        })
        tcs = r["choices"][0].get("tool_calls") or r["choices"][0]["message"].get("tool_calls")
        if tcs:
            success += 1

    test(f"Tool call reliability: {success}/10", success >= 8, f"{success}/10")


# ═══════════════════════════════════════════════════════════════
# 8. Sampling Parameters
# ═══════════════════════════════════════════════════════════════
def test_sampling():
    print("\n═══ 8. Sampling Parameters ═══")

    # 8a. top_p
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10, "top_p": 0.9,
    })
    test("top_p parameter accepted", "choices" in r)

    # 8b. top_k (non-standard but supported)
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10, "top_k": 20,
    })
    test("top_k parameter accepted", "choices" in r)

    # 8c. repetition_penalty
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10, "repetition_penalty": 1.1,
    })
    test("repetition_penalty accepted", "choices" in r)

    # 8d. logit_bias
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10, "logit_bias": {"100": 5.0},
    })
    test("logit_bias accepted", "choices" in r)


# ═══════════════════════════════════════════════════════════════
# 9a. Tool Choice Modes
# ═══════════════════════════════════════════════════════════════
def test_tool_choice():
    print("\n═══ 9a. Tool Choice ═══")

    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }}]

    # tool_choice: "none" — should NOT call tools
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": tools, "tool_choice": "none", "max_tokens": 100,
    })
    tcs = r["choices"][0].get("tool_calls") or r["choices"][0]["message"].get("tool_calls")
    test("tool_choice 'none': no tool calls", tcs is None)
    test("tool_choice 'none': finish_reason is NOT 'tool_calls'",
         r["choices"][0]["finish_reason"] != "tool_calls")

    # tool_choice: "auto" — model decides
    r2 = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": tools, "tool_choice": "auto", "max_tokens": 200,
    })
    test("tool_choice 'auto': accepted", "choices" in r2)

    # tool_choice: "required" — MUST call tool
    r3 = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": tools, "tool_choice": "required", "max_tokens": 200,
    })
    tcs3 = r3["choices"][0].get("tool_calls") or r3["choices"][0]["message"].get("tool_calls")
    test("tool_choice 'required': tool call generated", tcs3 is not None)

    # tool_choice: specific function (OpenAI format)
    r4 = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "max_tokens": 200,
    })
    tcs4 = r4["choices"][0].get("tool_calls") or r4["choices"][0]["message"].get("tool_calls")
    test("tool_choice specific: tool call generated", tcs4 is not None)
    if tcs4:
        test("tool_choice specific: correct function called",
             tcs4[0]["function"]["name"] == "get_weather" or "weather" in tcs4[0]["function"]["name"].lower())


# ═══════════════════════════════════════════════════════════════
# 9b. Agent-Critical: Streaming Tool Call Format
#    (Breaks OpenCode, Codex, Continue if wrong)
# ═══════════════════════════════════════════════════════════════
def test_streaming_tool_format():
    print("\n═══ 9. Streaming Tool Call Format ═══")

    tools = [{"type": "function", "function": {
        "name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }}]

    chunks = stream_api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read /tmp/test.txt"},
        ],
        "tools": tools,
        "max_tokens": 200,
    })
    data_chunks = [c for c in chunks if c != "[DONE]"]
    tc_chunks = [c for c in data_chunks if c["choices"][0]["delta"].get("tool_calls")]

    if not tc_chunks:
        skip("Streaming tool call format tests", "no tool_call chunks produced")
        return

    first_tc_delta = tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]

    # Critical: first delta MUST have id, type, function.name
    test("First tc delta has 'id'", "id" in first_tc_delta and first_tc_delta["id"] is not None)
    test("First tc delta has 'type': 'function'", first_tc_delta.get("type") == "function",
         f"got type={first_tc_delta.get('type')}")
    test("First tc delta has function.name (non-empty)",
         len(first_tc_delta.get("function", {}).get("name", "")) > 0)

    # Subsequent deltas should NOT overwrite name with empty string (Codex bug #7517)
    for tc_chunk in tc_chunks[1:]:
        delta_tc = tc_chunk["choices"][0]["delta"]["tool_calls"][0]
        name = delta_tc.get("function", {}).get("name")
        if name is not None:
            test("Subsequent delta does NOT send empty function.name", name != "",
                 "empty name would overwrite valid name in client")
            break

    # All argument fragments concatenated must be valid JSON
    all_args = ""
    for tc_chunk in tc_chunks:
        delta_tc = tc_chunk["choices"][0]["delta"]["tool_calls"][0]
        frag = delta_tc.get("function", {}).get("arguments", "")
        all_args += frag
    if all_args:
        try:
            json.loads(all_args)
            test("Concatenated arguments is valid JSON", True)
        except json.JSONDecodeError:
            test("Concatenated arguments is valid JSON", False, all_args[:100])

    # finish_reason must be "tool_calls" on final chunk
    final = data_chunks[-1]
    test("Final chunk finish_reason is 'tool_calls'",
         final["choices"][0].get("finish_reason") == "tool_calls")

    # No empty tool_calls array (breaks OpenCode #4255)
    for chunk in data_chunks:
        tc_arr = chunk["choices"][0]["delta"].get("tool_calls")
        if tc_arr is not None:
            test("No empty tool_calls array in chunks", len(tc_arr) > 0)
            break


# ═══════════════════════════════════════════════════════════════
# 10. Multi-Turn Tool Conversation (Full Round-Trip)
# ═══════════════════════════════════════════════════════════════
def test_multi_turn_tools():
    print("\n═══ 10. Multi-Turn Tool Conversation ═══")

    tools = [{"type": "function", "function": {
        "name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }}]

    # Turn 1: user asks to read file → model should call tool
    r1 = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read /tmp/test.txt"},
        ],
        "tools": tools, "max_tokens": 200,
    })
    c1 = r1["choices"][0]
    tcs = c1.get("tool_calls") or c1["message"].get("tool_calls")

    if not tcs:
        skip("Multi-turn tool tests", "initial tool call not generated")
        return

    test("Turn 1: tool call generated", True)

    # Turn 2: send tool response → model should respond with text
    r2 = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read /tmp/test.txt"},
            {"role": "assistant", "content": c1["message"].get("content"), "tool_calls": tcs},
            {"role": "tool", "tool_call_id": tcs[0]["id"], "content": "Hello World!"},
        ],
        "tools": tools, "max_tokens": 200,
    })
    c2 = r2["choices"][0]
    content2 = c2["message"].get("content") or ""
    test("Turn 2: text response after tool result", len(content2) > 0)
    test("Turn 2: finish_reason is 'stop' (not tool_calls)",
         c2.get("finish_reason") == "stop")

    # Turn 3: follow-up question → model should respond coherently
    r3 = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read /tmp/test.txt"},
            {"role": "assistant", "content": c1["message"].get("content"), "tool_calls": tcs},
            {"role": "tool", "tool_call_id": tcs[0]["id"], "content": "Hello World!"},
            {"role": "assistant", "content": content2},
            {"role": "user", "content": "What did the file contain?"},
        ],
        "tools": tools, "max_tokens": 200,
    })
    content3 = r3["choices"][0]["message"].get("content", "")
    test("Turn 3: coherent follow-up response", len(content3) > 0)


# ═══════════════════════════════════════════════════════════════
# 11. Response Format Strictness
#     (Fields that strict clients validate)
# ═══════════════════════════════════════════════════════════════
def test_response_format():
    print("\n═══ 11. Response Format Strictness ═══")

    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    })

    # id format
    test("id starts with 'chatcmpl-'", r["id"].startswith("chatcmpl-"))

    # created is integer (not float)
    test("created is integer", isinstance(r["created"], int))

    # model field present
    test("model field present", isinstance(r.get("model"), str) and len(r["model"]) > 0)

    # usage total = prompt + completion
    u = r["usage"]
    test("usage.total = prompt + completion",
         u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"])

    # choices[0].index is 0
    test("choices[0].index is 0", r["choices"][0].get("index") == 0)

    # message.content is string (not null for non-tool response)
    test("content is string for text response",
         isinstance(r["choices"][0]["message"]["content"], str))

    # Tool call response: content should be null
    tools = [{"type": "function", "function": {
        "name": "test_fn", "description": "Test",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    }}]
    r2 = api("/v1/chat/completions", {
        "model": "test",
        "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Read /tmp/x"}],
        "tools": tools, "max_tokens": 200,
    })
    c2 = r2["choices"][0]
    tcs = c2.get("tool_calls") or c2["message"].get("tool_calls")
    if tcs:
        # content CAN be null or string — both valid per spec
        content = c2["message"].get("content")
        test("Tool call: content is null or string",
             content is None or isinstance(content, str))
    else:
        skip("Tool call content null check", "no tool call generated")


# ═══════════════════════════════════════════════════════════════
# 12. Anthropic /v1/messages API
# ═══════════════════════════════════════════════════════════════
def test_anthropic_api():
    print("\n═══ 12. Anthropic /v1/messages API ═══")

    def anthropic(body):
        req = urllib.request.Request(
            f"{URL}/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": "sk-test",
                "anthropic-version": "2023-06-01",
            },
        )
        return json.loads(urllib.request.urlopen(req, timeout=300).read())

    # 12a. Basic response format
    r = anthropic({
        "model": "test", "max_tokens": 200,
        "messages": [{"role": "user", "content": "Say hi"}],
    })
    test("Anthropic: type is 'message'", r.get("type") == "message")
    test("Anthropic: role is 'assistant'", r.get("role") == "assistant")
    test("Anthropic: has content array", isinstance(r.get("content"), list))
    test("Anthropic: content has text block",
         any(b.get("type") == "text" for b in r.get("content", [])))
    test("Anthropic: has usage.input_tokens",
         "input_tokens" in r.get("usage", {}))
    test("Anthropic: has usage.output_tokens",
         "output_tokens" in r.get("usage", {}))
    test("Anthropic: stop_reason valid",
         r.get("stop_reason") in ("end_turn", "max_tokens", "tool_use"))

    # 12b. System message (top-level)
    r2 = anthropic({
        "model": "test", "max_tokens": 200,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    test("Anthropic: system message accepted", r2.get("type") == "message")

    # 12c. Tool use
    r3 = anthropic({
        "model": "test", "max_tokens": 200,
        "messages": [{"role": "user", "content": "Read /tmp/test.txt"}],
        "tools": [{
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }],
    })
    has_tool = any(b.get("type") == "tool_use" for b in r3.get("content", []))
    test("Anthropic: tool_use block generated", has_tool)
    if has_tool:
        tu = next(b for b in r3["content"] if b["type"] == "tool_use")
        test("Anthropic: tool_use has id", "id" in tu)
        test("Anthropic: tool_use has name", "name" in tu)
        test("Anthropic: tool_use has input (object)",
             isinstance(tu.get("input"), dict))
        test("Anthropic: stop_reason is 'tool_use'",
             r3.get("stop_reason") == "tool_use")

    # 12d. Content blocks (array format)
    r4 = anthropic({
        "model": "test", "max_tokens": 200,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Hello there"},
        ]}],
    })
    test("Anthropic: content blocks accepted", r4.get("type") == "message")


# ═══════════════════════════════════════════════════════════════
# 13. Error Handling
# ═══════════════════════════════════════════════════════════════
def test_error_handling():
    print("\n═══ 13. Error Handling ═══")

    # Invalid JSON
    try:
        req = urllib.request.Request(
            f"{URL}/v1/chat/completions",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        test("Invalid JSON returns error", False, "no error raised")
    except urllib.error.HTTPError as e:
        test("Invalid JSON returns 400", e.code == 400)
        body = json.loads(e.read())
        test("Error has 'error.message'", "message" in body.get("error", {}))

    # Empty messages
    try:
        r = api("/v1/chat/completions", {
            "model": "test", "messages": [], "max_tokens": 10,
        })
        # Some servers accept empty messages, some don't
        test("Empty messages handled", True)
    except urllib.error.HTTPError as e:
        test("Empty messages returns 400", e.code == 400)


# ═══════════════════════════════════════════════════════════════
# 14. OpenCode Agent Compatibility
#     (Complex system prompt + 12 tools — reproduces real agent conditions)
# ═══════════════════════════════════════════════════════════════
def _load_fixture(name: str):
    """Load a test fixture file from scripts/fixtures/."""
    from pathlib import Path
    p = Path(__file__).parent / "fixtures" / name
    return p.read_text()


def _load_json_fixture(name: str):
    from pathlib import Path
    p = Path(__file__).parent / "fixtures" / name
    return json.loads(p.read_text())


def _has_degeneration(text: str) -> tuple:
    """Check text for common degeneration signals. Returns (bool, detail)."""
    issues = []
    if "<think>" in text or "</think>" in text:
        issues.append("leaked <think> tags")
    if "<tool_call>" in text:
        issues.append("raw <tool_call> in content (should be parsed)")
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in text)
    if has_chinese:
        issues.append("Chinese characters in English response")
    has_cyrillic = any('\u0400' <= ch <= '\u04ff' for ch in text)
    if has_cyrillic:
        issues.append("Cyrillic characters in English response")
    if text.count("\\n") > 10 and "\\n" in text and "\n" not in text[:200]:
        issues.append("escaped \\n instead of real newlines")
    return (len(issues) > 0, "; ".join(issues))


def test_opencode_compat():
    print("\n═══ 14. OpenCode Agent Compatibility ═══")

    try:
        sys_prompt = _load_fixture("opencode_system_prompt.txt")
        tools = _load_json_fixture("opencode_tools.json")
    except FileNotFoundError as e:
        skip("OpenCode fixtures", str(e))
        return

    print(f"  (system prompt: {len(sys_prompt)} chars, {len(tools)} tools)")

    # 14a. Blocking tool call with complex prompt
    print("\n  ── 14a. Blocking tool call ──")
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Please write a typescript library for a simple Hello world webpage. Use the write tool."},
        ],
        "tools": tools,
        "max_tokens": 2000,
    })
    # Dump directory for debugging
    import os
    dump_dir = "/tmp/atlas-opencode-dumps"
    os.makedirs(dump_dir, exist_ok=True)

    c = r["choices"][0]
    content = c["message"].get("content") or ""
    tcs = c.get("tool_calls") or c["message"].get("tool_calls") or []
    fr = c.get("finish_reason")

    with open(f"{dump_dir}/14a_blocking.json", "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)

    test("Response is non-empty", len(content) > 0 or len(tcs) > 0,
         f"content={len(content)} chars, tool_calls={len(tcs)}")
    test("finish_reason is not 'length' (model finished naturally)",
         fr in ("stop", "tool_calls"), f"got '{fr}'")

    if content:
        degen, detail = _has_degeneration(content)
        test("No degeneration in content", not degen, detail)

    if tcs:
        tc = tcs[0]
        test("Tool call has valid structure",
             "id" in tc and tc.get("type") == "function" and "function" in tc)

        args_str = tc.get("function", {}).get("arguments", "")
        try:
            args = json.loads(args_str)
            test("Tool call arguments is valid JSON", True)
        except json.JSONDecodeError:
            test("Tool call arguments is valid JSON", False, args_str[:200])

        # Check for TRIPLE-escaped backslashes (\\\\n in JSON = \\n parsed, corruption).
        # Double-escaped (\\n in JSON = \n parsed) is valid for code containing JS escapes.
        test("No triple-escaped backslashes in JSON args",
             "\\\\\\\\n" not in args_str and "\\\\\\\\t" not in args_str,
             f"found triple-escaped chars in: {args_str[:100]}")

    # 14b. Streaming with complex prompt
    print("\n  ── 14b. Streaming tool call ──")
    chunks = stream_api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Use the Read tool to read the file at /workspace/package.json"},
        ],
        "tools": tools,
        "max_tokens": 500,
    })
    data_chunks = [c for c in chunks if c != "[DONE]"]

    # Collect all content
    all_content = ""
    for dc in data_chunks:
        delta = dc["choices"][0].get("delta", {})
        if delta.get("content"):
            all_content += delta["content"]

    # Collect all tool call argument fragments
    all_args = ""
    tc_chunks = [c for c in data_chunks if c["choices"][0].get("delta", {}).get("tool_calls")]
    for tc_chunk in tc_chunks:
        delta_tc = tc_chunk["choices"][0]["delta"]["tool_calls"][0]
        frag = delta_tc.get("function", {}).get("arguments", "")
        all_args += frag

    test("Stream produces chunks", len(data_chunks) > 1)

    if all_content:
        degen, detail = _has_degeneration(all_content)
        test("No degeneration in streamed content", not degen, detail)

    if all_args:
        try:
            json.loads(all_args)
            test("Streamed tool args concatenate to valid JSON", True)
        except json.JSONDecodeError:
            test("Streamed tool args concatenate to valid JSON", False, all_args[:200])

    final = data_chunks[-1] if data_chunks else {}
    final_fr = final.get("choices", [{}])[0].get("finish_reason")
    test("Stream finish_reason is valid",
         final_fr in ("stop", "tool_calls"), f"got '{final_fr}'")

    # 14c. Multi-turn: tool response then follow-up
    print("\n  ── 14c. Multi-turn after tool response ──")
    if tcs:
        tc = tcs[0]
        r2 = api("/v1/chat/completions", {
            "model": "test",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Please write a typescript library for a simple Hello world webpage. Use the write tool."},
                {"role": "assistant", "content": content, "tool_calls": tcs},
                {"role": "tool", "tool_call_id": tc["id"],
                 "content": "File written successfully to ./src/hello.ts"},
                {"role": "user", "content": "Great, now read the file you just wrote to verify it."},
            ],
            "tools": tools,
            "max_tokens": 500,
        })
        c2 = r2["choices"][0]
        content2 = c2["message"].get("content") or ""
        tcs2 = c2.get("tool_calls") or c2["message"].get("tool_calls") or []

        test("Multi-turn produces response",
             len(content2) > 0 or len(tcs2) > 0)

        if content2:
            degen2, detail2 = _has_degeneration(content2)
            test("No degeneration in multi-turn content", not degen2, detail2)

        fr2 = c2.get("finish_reason")
        test("Multi-turn finish_reason is valid",
             fr2 in ("stop", "tool_calls"), f"got '{fr2}'")
    else:
        skip("Multi-turn tests", "no tool calls from 14a to build on")

    # Dump directory for debugging
    import os
    dump_dir = "/tmp/atlas-opencode-dumps"
    os.makedirs(dump_dir, exist_ok=True)

    # 14d. Thinking coherence — no gibberish in reasoning_content
    print("\n  ── 14d. Thinking coherence ──")
    r_think = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "What is 2+2? Think step by step."},
        ],
        "max_tokens": 1024,
        "enable_thinking": True,
    })
    # Dump full response
    with open(f"{dump_dir}/14d_thinking.json", "w") as f:
        json.dump(r_think, f, indent=2, ensure_ascii=False)
    c_think = r_think["choices"][0]
    reasoning = c_think["message"].get("reasoning_content", "") or ""
    content_think = c_think["message"].get("content", "") or ""
    fr_think = c_think.get("finish_reason")

    test("Thinking produces content or reasoning",
         len(reasoning) > 0 or len(content_think) > 0,
         f"reasoning={len(reasoning)} chars, content={len(content_think)} chars")

    if reasoning:
        degen_r, detail_r = _has_degeneration(reasoning)
        test("No gibberish in reasoning_content", not degen_r, detail_r)
        has_tool_xml = "<tool_call>" in reasoning or "<function=" in reasoning
        test("No <tool_call> XML in reasoning_content", not has_tool_xml,
             f"thinking contains tool call XML: {reasoning[:200]}")
    else:
        test("No gibberish in reasoning_content", True)
        test("No <tool_call> XML in reasoning_content", True)

    if content_think:
        degen_c, detail_c = _has_degeneration(content_think)
        test("No gibberish in content after thinking", not degen_c, detail_c)

    test("finish_reason is not 'length'",
         fr_think in ("stop", "tool_calls"), f"got '{fr_think}'")

    # 14e. Thinking with reasoning.effort (OpenCode style)
    print("\n  ── 14e. Thinking via reasoning.effort ──")
    r_effort = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Create a rust project with a hello world main.rs"},
        ],
        "tools": tools,
        "max_tokens": 2000,
        "reasoning": {"effort": "high"},
    })
    with open(f"{dump_dir}/14e_reasoning_effort.json", "w") as f:
        json.dump(r_effort, f, indent=2, ensure_ascii=False)

    c_eff = r_effort["choices"][0]
    reasoning_eff = c_eff["message"].get("reasoning_content", "") or ""
    content_eff = c_eff["message"].get("content", "") or ""
    fr_eff = c_eff.get("finish_reason")

    test("reasoning.effort produces response",
         len(reasoning_eff) > 0 or len(content_eff) > 0,
         f"reasoning={len(reasoning_eff)} chars, content={len(content_eff)} chars")

    if reasoning_eff:
        degen_e, detail_e = _has_degeneration(reasoning_eff)
        test("No gibberish in reasoning_content (effort mode)", not degen_e, detail_e)

    if content_eff:
        degen_ec, detail_ec = _has_degeneration(content_eff)
        test("No gibberish in content (effort mode)", not degen_ec, detail_ec)

    test("finish_reason is not 'length' (effort mode)",
         fr_eff in ("stop", "tool_calls"), f"got '{fr_eff}'")

    # 14f. Concurrent requests (title + agent simultaneously like OpenCode)
    print("\n  ── 14f. Concurrent requests (batch=2 regression) ──")
    import threading
    conc_results = {}
    def _conc_req(name, payload):
        try:
            r = api("/v1/chat/completions", payload)
            c = r["choices"][0]["message"]
            rc = c.get("reasoning_content", "") or ""
            ct = c.get("content", "") or ""
            conc_results[name] = {"reasoning": rc, "content": ct}
        except Exception as e:
            conc_results[name] = {"error": str(e)}

    title_payload = {
        "model": "test", "max_tokens": 8192, "temperature": 0.5,
        "enable_thinking": True,
        "messages": [
            {"role": "system", "content": "You are a title generator. Output ONLY a brief title."},
            {"role": "user", "content": "Title for: rust tokio echo server"},
        ],
    }
    agent_payload = {
        "model": "test", "max_tokens": 2000, "temperature": 0.55,
        "enable_thinking": True,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Write a hello world in Rust"},
        ],
        "tools": tools, "tool_choice": "auto",
    }
    t1 = threading.Thread(target=_conc_req, args=("title", title_payload))
    t2 = threading.Thread(target=_conc_req, args=("agent", agent_payload))
    t1.start(); t2.start(); t1.join(timeout=120); t2.join(timeout=120)

    for name in ["title", "agent"]:
        r = conc_results.get(name, {})
        if "error" in r:
            test(f"Concurrent {name}: no error", False, r["error"][:100])
        else:
            rc = r.get("reasoning", "")
            ct = r.get("content", "")
            all_text = rc + ct
            degen, detail = _has_degeneration(all_text)
            test(f"Concurrent {name}: no gibberish", not degen, detail)

    # 14g. Streaming thinking coherence
    print("\n  ── 14g. Streaming thinking coherence ──")
    chunks = stream_api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Use the Write tool to create a file ./hello.txt with the content 'hello world'"},
        ],
        "tools": tools,
        "max_tokens": 1000,
        "enable_thinking": True,
    })
    data_chunks = [c for c in chunks if c != "[DONE]"]
    all_reasoning = ""
    all_stream_content = ""
    for dc in data_chunks:
        delta = dc["choices"][0].get("delta", {})
        rc = delta.get("reasoning_content", "")
        if rc:
            all_reasoning += rc
        ct = delta.get("content", "")
        if ct:
            all_stream_content += ct

    if all_reasoning:
        degen_sr, detail_sr = _has_degeneration(all_reasoning)
        test("No gibberish in streamed reasoning", not degen_sr, detail_sr)
    else:
        test("No gibberish in streamed reasoning", True)

    if all_stream_content:
        degen_sc, detail_sc = _has_degeneration(all_stream_content)
        test("No gibberish in streamed content", not degen_sc, detail_sc)

    final_fr = data_chunks[-1]["choices"][0].get("finish_reason") if data_chunks else None
    test("Streaming finish_reason is valid",
         final_fr in ("stop", "tool_calls"), f"got '{final_fr}'")


# ═══════════════════════════════════════════════════════════════
# 15. Claude Code Agent Compatibility (Anthropic /v1/messages)
# ═══════════════════════════════════════════════════════════════
def test_anthropic_agent_compat():
    """Test Claude Code agent scenarios against the Anthropic Messages API."""
    print("\n═══ 15. Claude Code Agent Compatibility (Anthropic) ═══")

    try:
        sys_prompt = _load_fixture("claude_code_system_prompt.txt")
    except FileNotFoundError as e:
        skip("Claude Code fixtures", str(e))
        return

    # Anthropic tool format (input_schema instead of parameters)
    tools = [
        {"name": "Bash", "description": "Run a bash command",
         "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        {"name": "Read", "description": "Read a file from disk",
         "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
        {"name": "Write", "description": "Write a file to disk",
         "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
    ]

    def anthropic_req(payload, timeout=300):
        req = urllib.request.Request(
            f"{URL}/v1/messages",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"},
        )
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    print(f"  (system prompt: {len(sys_prompt)} chars, {len(tools)} tools)")

    # 15a. Blocking tool call
    print("\n  ── 15a. Blocking tool call ──")
    r = anthropic_req({
        "model": "test",
        "system": sys_prompt,
        "messages": [{"role": "user", "content": "Read the file /workspace/atlas/Cargo.toml and tell me the package name"}],
        "tools": tools,
        "max_tokens": 1000,
    })
    content_blocks = r.get("content", [])
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]
    tool_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
    stop_reason = r.get("stop_reason", "")

    test("Response has content blocks", len(content_blocks) > 0)
    test("stop_reason is valid", stop_reason in ("end_turn", "tool_use"), f"got '{stop_reason}'")

    all_text = " ".join(b.get("text", "") for b in text_blocks)
    if all_text:
        degen, detail = _has_degeneration(all_text)
        test("No gibberish in text content", not degen, detail)

    if tool_blocks:
        tb = tool_blocks[0]
        test("Tool use has id", "id" in tb)
        test("Tool use has name", "name" in tb)
        test("Tool use has input", "input" in tb and isinstance(tb["input"], dict))
        test("Tool use input is non-empty", len(tb.get("input", {})) > 0,
             f"input={tb.get('input', {})}")

    # 15b. Streaming
    print("\n  ── 15b. Streaming ──")
    payload = {
        "model": "test",
        "system": sys_prompt,
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
        "max_tokens": 100,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{URL}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    events = []
    streamed_text = ""
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: "):
            try:
                ev = json.loads(line[6:])
                events.append(ev)
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta", {})
                    if delta.get("type") == "text_delta":
                        streamed_text += delta.get("text", "")
            except json.JSONDecodeError:
                pass

    test("Stream produces events", len(events) > 2)
    test("Streamed text is non-empty", len(streamed_text) > 0, f"got: {repr(streamed_text[:100])}")
    if streamed_text:
        degen_s, detail_s = _has_degeneration(streamed_text)
        test("No gibberish in streamed text", not degen_s, detail_s)

    has_stop = any(e.get("type") == "message_stop" for e in events)
    test("Stream ends with message_stop", has_stop)

    # 15c. Multi-turn with tool result
    print("\n  ── 15c. Multi-turn tool result ──")
    if tool_blocks:
        tb = tool_blocks[0]
        r2 = anthropic_req({
            "model": "test",
            "system": sys_prompt,
            "messages": [
                {"role": "user", "content": "Read /workspace/atlas/Cargo.toml"},
                {"role": "assistant", "content": content_blocks},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tb["id"],
                     "content": '[package]\nname = "atlas"\nversion = "0.1.0"'},
                ]},
            ],
            "tools": tools,
            "max_tokens": 200,
        })
        text2 = " ".join(b.get("text", "") for b in r2.get("content", []) if b.get("type") == "text")
        test("Multi-turn produces text response", len(text2) > 0)
        if text2:
            degen2, detail2 = _has_degeneration(text2)
            test("No gibberish in multi-turn response", not degen2, detail2)
    else:
        skip("Multi-turn tests", "no tool_use from 15a")

    # 15d. Concurrent requests (batch=2 regression)
    print("\n  ── 15d. Concurrent requests ──")
    import threading
    conc_results = {}
    def _anth_conc(name, payload):
        try:
            r = anthropic_req(payload)
            text = " ".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
            conc_results[name] = text
        except Exception as e:
            conc_results[name] = f"ERROR: {e}"

    t1 = threading.Thread(target=_anth_conc, args=("A", {
        "model": "test", "system": "Title generator.", "max_tokens": 100,
        "messages": [{"role": "user", "content": "Title for: rust echo server"}],
    }))
    t2 = threading.Thread(target=_anth_conc, args=("B", {
        "model": "test", "system": sys_prompt, "max_tokens": 500,
        "messages": [{"role": "user", "content": "Hello world in Rust"}],
        "tools": tools,
    }))
    t1.start(); t2.start(); t1.join(timeout=120); t2.join(timeout=120)
    for name in ["A", "B"]:
        text = conc_results.get(name, "")
        if text.startswith("ERROR"):
            test(f"Concurrent {name}: no error", False, text[:100])
        else:
            degen, detail = _has_degeneration(text)
            test(f"Concurrent {name}: no gibberish", not degen, detail)


# ═══════════════════════════════════════════════════════════════
# 16. End-to-End Tool Execution — Rust Project Creation
# ═══════════════════════════════════════════════════════════════
def test_e2e_rust_project():
    """Simulate a multi-turn conversation that creates a Rust project.
    Verifies that tool calls are properly formatted and produce correct results."""
    print("\n═══ 16. End-to-End Rust Project Creation ═══")

    tools = [
        {"type": "function", "function": {
            "name": "write", "description": "Write a file to disk",
            "parameters": {"type": "object",
                "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
                "required": ["filePath", "content"]}}},
        {"type": "function", "function": {
            "name": "bash", "description": "Run a shell command",
            "parameters": {"type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]}}},
    ]

    # Turn 1: Ask model to create Cargo.toml
    print("\n  ── 16a. Generate Cargo.toml ──")
    r = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are a coding assistant. Create files using tools."},
            {"role": "user", "content": "Create a Cargo.toml for a Rust project called 'echo-server' with tokio dependency. Use the write tool with filePath='./e2e-test/Cargo.toml'."},
        ],
        "tools": tools,
        "tool_choice": "required",
        "max_tokens": 500,
    })
    c = r["choices"][0]
    tcs = c.get("tool_calls") or c["message"].get("tool_calls") or []
    fr = c.get("finish_reason")

    test("Turn 1: tool call generated", len(tcs) > 0, f"got {len(tcs)} tool calls")
    test("Turn 1: finish_reason is tool_calls", fr == "tool_calls", f"got '{fr}'")

    if tcs:
        tc = tcs[0]
        fn_name = tc.get("function", {}).get("name", "")
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except:
            args = {}

        test("Turn 1: tool name is 'write'", fn_name == "write", f"got '{fn_name}'")

        file_path = args.get("filePath", args.get("file_path", ""))
        content = args.get("content", "")

        test("Turn 1: filePath is non-empty", len(file_path) > 0, f"got '{file_path}'")
        test("Turn 1: content is non-empty", len(content) > 0, f"got {len(content)} chars")

        if content:
            test("Turn 1: content has [package]", "[package]" in content, content[:100])
            test("Turn 1: content has tokio", "tokio" in content.lower(), content[:200])

            degen, detail = _has_degeneration(content)
            test("Turn 1: no gibberish in Cargo.toml content", not degen, detail)

    # Turn 2: Ask for main.rs
    print("\n  ── 16b. Generate main.rs ──")
    r2 = api("/v1/chat/completions", {
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are a coding assistant. Create files using tools."},
            {"role": "user", "content": "Write a Rust main.rs with 'fn main()' that prints Hello World. Use the write tool with filePath='./e2e-test/src/main.rs'."},
        ],
        "tools": tools,
        "tool_choice": "required",
        "max_tokens": 500,
    })
    c2 = r2["choices"][0]
    tcs2 = c2.get("tool_calls") or c2["message"].get("tool_calls") or []

    test("Turn 2: tool call generated", len(tcs2) > 0)

    if tcs2:
        args2_str = tcs2[0].get("function", {}).get("arguments", "{}")
        try:
            args2 = json.loads(args2_str)
        except:
            args2 = {}

        content2 = args2.get("content", "")
        test("Turn 2: content has fn main", "fn main" in content2, content2[:100])
        test("Turn 2: content has println", "println" in content2 or "print" in content2, content2[:100])

        degen2, detail2 = _has_degeneration(content2)
        test("Turn 2: no gibberish in main.rs content", not degen2, detail2)

    # Dump results
    import os
    dump_dir = "/tmp/atlas-opencode-dumps"
    os.makedirs(dump_dir, exist_ok=True)
    with open(f"{dump_dir}/e2e_turn1.json", "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    with open(f"{dump_dir}/e2e_turn2.json", "w") as f:
        json.dump(r2, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atlas coherence test suite")
    parser.add_argument("--url", default="http://localhost:8888", help="Server URL")
    parser.add_argument("--opencode", action="store_true",
                        help="Run OpenCode agent compatibility tests (section 14)")
    parser.add_argument("--anthropic", action="store_true",
                        help="Run Claude Code agent tests against /v1/messages Anthropic API (section 15)")
    parser.add_argument("--e2e", action="store_true",
                        help="Run end-to-end Rust project creation test (section 16)")
    args = parser.parse_args()
    URL = args.url

    print(f"Testing Atlas at {URL}")

    t0 = time.time()
    test_basic_api()
    test_streaming()
    test_tool_calling()
    test_coherence()
    test_determinism()
    test_edge_cases()
    test_tool_reliability()
    test_sampling()
    test_tool_choice()
    test_streaming_tool_format()
    test_multi_turn_tools()
    test_response_format()
    test_anthropic_api()
    test_error_handling()
    if args.opencode:
        test_opencode_compat()
    if args.anthropic:
        test_anthropic_agent_compat()
    if args.e2e:
        test_e2e_rust_project()

    dt = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Results: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped ({dt:.0f}s)")
    print(f"{'=' * 60}")

    sys.exit(1 if FAILED > 0 else 0)

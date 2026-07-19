#!/usr/bin/env python3
"""Fire tool-calling prompts at a serve and classify the output as ok/mangled/empty.

"Mangled" = the known Atlas failure signature: raw tool-call markup leaking into
the assistant *content* (<function=...>, <function_calls>, </parameters>, <tool_call>)
instead of being parsed into tool_calls, or content that is junk/degenerate.
"""
import json
import re
import sys
import urllib.request

HOSTPORT, DRAFTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
# Accept either "port" (host defaults to 0.0.0.0, for local serves) or "host:port"
# (for probing a remote serve, e.g. "10.10.10.2:8888" from another box).
if ":" in HOSTPORT:
    HOST, PORT = HOSTPORT.rsplit(":", 1)
else:
    HOST, PORT = "0.0.0.0", HOSTPORT
URL = f"http://{HOST}:{PORT}/v1/chat/completions"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["location"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}]

PROMPTS = [
    "What is the weather in Paris?",
    "What is the weather in Tokyo in celsius?",
    "Add 17 and 42.",
    "What's the weather in Berlin?",
    "Add 100 and 250, then tell me the result.",
    "Weather in New York, in fahrenheit please.",
    "What is the weather in Cairo?",
    "Add 7 and 8.",
]

MANGLE = re.compile(r"<function=|<function_calls>|</parameters>|<tool_call>|</function>|<parameter=")

results = {"n": 0, "tool_ok": 0, "mangled": 0, "empty": 0, "mean_toks": 0.0, "examples": []}
toks = []
for p in PROMPTS:
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": p}],
        "tools": TOOLS,
        "max_tokens": 400,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
    except Exception as e:
        results["n"] += 1
        results["examples"].append({"verdict": "ERROR", "snippet": f"{type(e).__name__}: {e}"})
        continue

    results["n"] += 1
    ch = d["choices"][0]
    msg = ch.get("message", {}) or {}
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    toks.append((d.get("usage", {}) or {}).get("completion_tokens", 0))

    if MANGLE.search(content):
        verdict = "MANGLED"          # tool markup leaked into content
        results["mangled"] += 1
    elif tcs:
        verdict = "TOOL_OK"          # parsed into a real tool_call
        results["tool_ok"] += 1
    elif not content.strip():
        verdict = "EMPTY"
        results["empty"] += 1
    else:
        verdict = "PROSE_NO_CALL"    # answered in prose instead of calling
    snippet = (content or json.dumps(tcs))[:200].replace("\n", "\\n")
    results["examples"].append({"verdict": verdict, "prompt": p, "snippet": snippet})
    print(f"[mtp{DRAFTS}] {verdict:14} {p[:34]!r} -> {snippet[:70]!r}")

results["mean_toks"] = sum(toks) / len(toks) if toks else 0.0
json.dump(results, open(OUT, "w"), indent=2)
print(f"[mtp{DRAFTS}] tool_ok={results['tool_ok']}/{results['n']} "
      f"mangled={results['mangled']}/{results['n']} empty={results['empty']}/{results['n']}")

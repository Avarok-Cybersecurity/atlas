#!/usr/bin/env python3
"""Capture serial (batch=1, MTP) multi-turn agentic outputs for A/B vs Marconi.

Run twice against two server configs (Marconi ON vs OFF), identical seeds /
greedy, and diff the per-turn token streams. At batch=1 MTP is active, so this
is the exact documented path: --enable-prefix-caching + --speculative on a
hybrid SSM model, warm Marconi restores across turns. Marconi-OFF (full SSM
recompute each turn) is the proven-correct reference. Greedy ⇒ identical.

Outputs JSON {case: [turn_tokens,...]} to the path in argv[1] so a second run
can diff. Includes tool-call turns (duplicated-tool-arg-fragment symptom).
"""
import json
import os
import sys
import urllib.request

URL = os.environ.get("ATLAS_URL", "http://localhost:8888")
MODEL = os.environ.get("ATLAS_MODEL", "model")
SYS = "You are a helpful assistant. Be thorough and precise."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Create a file with given path and content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
}]

# Each case is a list of user turns. Mix of long prose (cross decode-ckpt
# boundaries → decode-era snapshots) and a tool-call turn (arg fragments).
CASES = [
    [
        "Write a detailed 250-word explanation of how DNS resolution works, "
        "from the stub resolver through recursive and authoritative servers.",
        "Now summarize that into five bullet points.",
        "Create a file at /tmp/dns.md containing a 200-word markdown writeup "
        "of the same topic with a title and three sections.",
        "Good. Now list the three sections you used, one line each.",
    ],
    [
        "Explain in 250 words how a TCP three-way handshake establishes a "
        "connection, including SYN, SYN-ACK, ACK and sequence numbers.",
        "Summarize the handshake in exactly three sentences.",
        "Create a file at /tmp/tcp.json whose content is a JSON object with "
        "keys 'steps' (array of 3 strings) and 'purpose' (a sentence).",
        "Now restate the purpose field in your own words.",
    ],
    [
        "Describe in 250 words how the Calvin cycle fixes carbon during "
        "photosynthesis, naming the key enzyme and the energy carriers used.",
        "List the three phases of the cycle, one sentence each.",
        "Create a file at /tmp/calvin.txt containing a numbered list of the "
        "three phases with a two-sentence description for each phase.",
        "Which phase consumes the most ATP, and why?",
    ],
]


def chat(messages, tools=None, max_tokens=384):
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "logprobs": True, "top_logprobs": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    c = r["choices"][0]
    msg = c["message"]
    lp = (c.get("logprobs") or {}).get("content") or []
    tc = msg.get("tool_calls") or []
    # Serialize tool-call arguments deterministically for the diff.
    tc_repr = [f"{t['function']['name']}({t['function']['arguments']})" for t in tc]
    return {
        "tokens": [e.get("token") for e in lp],
        "content": msg.get("content") or "",
        "tool_calls": tc_repr,
        "finish_reason": c.get("finish_reason"),
    }


def main():
    out_path = sys.argv[1]
    results = {}
    for ci, turns in enumerate(CASES):
        msgs = [{"role": "system", "content": SYS}]
        turn_outs = []
        for ti, u in enumerate(turns):
            msgs.append({"role": "user", "content": u})
            use_tools = "Create a file" in u
            o = chat(msgs, tools=TOOLS if use_tools else None)
            turn_outs.append(o)
            # Append assistant turn to history (content + tool calls).
            asst = {"role": "assistant", "content": o["content"]}
            msgs.append(asst)
            if o["tool_calls"]:
                # Feed a tool result so the conversation can continue.
                msgs.append({"role": "tool", "content": "OK, file created."})
        results[f"C{ci}"] = turn_outs
        print(f"  captured C{ci}: {len(turn_outs)} turns "
              f"(tool_turns={sum(1 for o in turn_outs if o['tool_calls'])})",
              flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

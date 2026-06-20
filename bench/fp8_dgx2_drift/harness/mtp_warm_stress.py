#!/usr/bin/env python3
"""MTP x warm-restore stress repro (2026-06-10).

The warm-hit token-stutter only manifests with --speculative under real
agentic traffic. Hypothesis: high draft-rejection rates (unpredictable
text) x many decode steps x Marconi warm restores desync rewind
bookkeeping. This script maximizes those three factors and scans every
reply for stutter signatures.

Per iteration: one long chained conversation (10k-token system prompt so
every turn is multi-chunk and warm), tools defined, thinking left at the
server default, temp 0.3, prompts that demand UNPREDICTABLE output
(random-looking identifiers, mixed prose/code) to tank MTP acceptance,
plus exact-path echo demands so corruption is detectable.

Usage: python3 mtp_warm_stress.py [--iters 3] [--turns 8]
Exit code 1 if any stutter suspect found (prints them).
"""

import argparse
import json
import re
import sys
import urllib.request

FILLER = (
    "Consider the module layout under /tmp/harness-proj/src with files "
    "main.rs, routes.rs, state.rs and tests in /tmp/harness-proj/tests. "
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file to disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Absolute path"},
                    "content": {"type": "string"},
                },
                "required": ["filePath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

# Unpredictable content requests tank n-gram-style draft acceptance;
# exact path echoes make stutter detectable.
PROMPTS = [
    "Write /tmp/stress-zq7/src/dispatch_v2.rs via the write tool: a Rust enum "
    "with 12 variants named after obscure chemical elements, each with a "
    "distinct u64 discriminant that is a large prime.",
    "Now write /tmp/stress-zq7/src/metrics_agg.rs via the write tool: a struct "
    "with 9 fields mixing exotic identifier names like qx_phi_rate, "
    "lorenz_tail_p99, and brk_hyst_window, with doc comments.",
    "Run via bash: ls -la /tmp/stress-zq7/src/ && echo checkpoint-alpha-7731",
    "Write /tmp/stress-zq7/tests/prop_fuzz_matrix.rs via the write tool with "
    "3 proptest-style test stubs using unusual generator names.",
    "Write /tmp/stress-zq7/src/wire_codec.rs via the write tool: varint "
    "encode/decode with bit-twiddling and named constants like MASK_0x7F_CONT.",
    "Run via bash: cargo check --manifest-path /tmp/stress-zq7/Cargo.toml 2>&1 | tail -3",
    "Write /tmp/stress-zq7/src/sched_ring.rs via the write tool: a ring buffer "
    "with wrap-around arithmetic and seqlock-style version counters.",
    "Final: run via bash: ls /tmp/stress-zq7/src/ /tmp/stress-zq7/tests/ && echo done-tag-zq7",
]

STUTTER = re.compile(
    r"(/tmp/[\w\-./]*?)(\w{2,6})\2(?=[\w\-./])"  # doubled fragment inside a path
)


def chat(messages, temp=0.3):
    body = {
        "model": "x",
        "messages": messages,
        "max_tokens": 700,
        "temperature": temp,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    req = urllib.request.Request(
        "http://localhost:8888/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    d = json.load(urllib.request.urlopen(req, timeout=240))
    return d["choices"][0]["message"]


def scan(s):
    out = []
    for m in STUTTER.finditer(s or ""):
        out.append(m.group(0))
    # explicit expected-path check
    for p in re.findall(r"/tmp/[\w\-./]+", s or ""):
        if "/tmp/stress-zq7" not in p and "/tmp/harness-proj" not in p:
            out.append("OFFPATH:" + p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--turns", type=int, default=8)
    args = ap.parse_args()

    suspects = []
    for it in range(args.iters):
        system = (
            f"You are a coding agent (iteration {it}). Use tools with EXACT "
            "absolute paths. " + FILLER * 320
        )
        msgs = [{"role": "system", "content": system}]
        for k in range(min(args.turns, len(PROMPTS))):
            msgs.append({"role": "user", "content": PROMPTS[k] + "\n" + FILLER * 10})
            try:
                m = chat(msgs)
            except Exception as e:
                print(f"iter {it} turn {k + 1}: request error {e}")
                break
            tcs = m.get("tool_calls") or []
            blob = json.dumps([t["function"]["arguments"] for t in tcs]) + (
                m.get("content") or ""
            )
            hits = scan(blob)
            paths = []
            for tc in tcs:
                try:
                    a = json.loads(tc["function"]["arguments"])
                    paths.append(a.get("filePath") or a.get("command", "")[:60])
                except Exception:
                    paths.append("UNPARSEABLE:" + tc["function"]["arguments"][:60])
                    hits.append("UNPARSEABLE_ARGS")
            print(f"iter {it} turn {k + 1}: tcs={len(tcs)} paths={paths}"
                  + (f"  SUSPECTS={hits}" if hits else ""))
            sys.stdout.flush()
            if hits:
                suspects.extend(hits)
            msgs.append(
                {"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs}
            )
            for tc in tcs:
                msgs.append(
                    {"role": "tool", "tool_call_id": tc.get("id", "x"), "content": "ok"}
                )
    print("TOTAL SUSPECTS:", len(suspects))
    for s in suspects[:20]:
        print("  ", s)
    sys.exit(1 if suspects else 0)


if __name__ == "__main__":
    main()

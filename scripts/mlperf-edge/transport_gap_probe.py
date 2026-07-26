#!/usr/bin/env python3
"""Measure the chat-vs-completions transport gap — the largest remaining
addressable slice after the GDN_REGRESIDENT fold.

After the fold the wall (3834 s) decomposes as: decode 62.4% (roofline-bound,
K=4 verified optimal), FIXED per-turn TTFT 24.1% (919 ms x 1007), marginal
prefill 13.4%. The fixed slice is now the biggest non-decode target, and the
dgx2 server-side law attributes ~207 ms of it to the `/v1/chat/completions`
path against only ~75 ms for `/v1/completions` — a ~132 ms/turn delta worth
~133 s (3.5%) over 1007 turns.

That gap is NOT GPU work: both endpoints run the identical prefill and decode.
Whatever differs is CPU-side request handling — chat-template rendering,
message-list tokenization, tool-schema processing, response assembly. This
probe isolates it by sending the SAME token content through both endpoints
against the SAME warm prefix, so the model work is held constant and only the
request path varies.

Three cells separate the candidate causes:
  plain     - short prompt, no tools   -> baseline template cost
  long      - ~38k-char prompt, no tools -> does the gap scale with prompt size
                                          (tokenization/render) or is it fixed?
  tools     - short prompt WITH a tool schema -> tool-schema handling cost
The MLPerf harness sends tool schemas on essentially every turn, so if the gap
lives in `tools`, it is fully in play for the benchmark.

Usage: transport_gap_probe.py <port> <model> <out.json> [--reps 9]
"""
import argparse
import json
import statistics
import time
import urllib.request

CHUNK = ("def resolve(self, name):\n    for k in type(self).__mro__:\n"
         "        if name in vars(k): return vars(k)[name]\n    raise KeyError(name)\n\n")

TOOLS = [{"type": "function", "function": {
    "name": "bash",
    "description": "Run a bash command in the workspace and return its output.",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string",
                                              "description": "The command to run"}},
                   "required": ["command"]}}}]


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            s = raw.decode("utf-8", "ignore").strip()
            if not s.startswith("data:"):
                continue
            p = s[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            ch = (d.get("choices") or [{}])[0]
            got = ch.get("text") or (ch.get("delta") or {}).get("content") \
                or (ch.get("delta") or {}).get("tool_calls")
            if got:
                return (time.time() - t0) * 1000.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=9)
    a = ap.parse_args()
    M = a.model

    SHORT = CHUNK * 3
    LONG = CHUNK * 340          # ~38k chars, the harness p50 prompt depth

    def comp(text, _tools=None):
        return "/v1/completions", {"model": M, "prompt": text, "max_tokens": 8,
                                   "temperature": 0, "seed": 42, "stream": True}

    def chat(text, tools=None):
        b = {"model": M, "messages": [{"role": "user", "content": text}],
             "max_tokens": 8, "temperature": 0, "seed": 42, "stream": True}
        if tools:
            b["tools"] = tools
        return "/v1/chat/completions", b

    cells = {
        "plain": (SHORT, None),
        "long":  (LONG, None),
        "tools": (SHORT, TOOLS),
    }
    out = {}
    for name, (text, tools) in cells.items():
        row = {}
        for api, mk in (("completions", comp), ("chat", chat)):
            path, body = mk(text, tools)
            post(a.port, path, body)                      # warm the prefix
            vals = [post(a.port, path, body) for _ in range(a.reps)]
            row[api] = {"p50": statistics.median(vals), "min": min(vals), "vals": vals}
        gap = row["chat"]["p50"] - row["completions"]["p50"]
        row["gap_ms"] = gap
        out[name] = row
        print(f"{name:8s} completions p50={row['completions']['p50']:7.1f}  "
              f"chat p50={row['chat']['p50']:7.1f}  gap={gap:+7.1f} ms", flush=True)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    g = {k: v["gap_ms"] for k, v in out.items()}
    print(f"\ngap plain={g['plain']:+.1f}  long={g['long']:+.1f}  tools={g['tools']:+.1f} ms")
    print("Reading:")
    print("  gap(long) >> gap(plain)  -> cost scales with prompt: template render / tokenization")
    print("  gap(tools) >> gap(plain) -> cost is tool-schema handling (in play EVERY harness turn)")
    print("  all gaps similar & small -> transport is not the 207 ms; the fixed slice is elsewhere")
    print(f"\nFor scale: 132 ms/turn x 1007 turns = {132*1007/1000:.0f} s = "
          f"{100*132*1007/1000/3834:.1f}% of the post-fold 3834 s wall.")


if __name__ == "__main__":
    main()

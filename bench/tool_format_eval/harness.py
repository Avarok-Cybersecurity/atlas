# SPDX-License-Identifier: AGPL-3.0-only
"""Tool-format eval harness — Layer 1 (behavioral) + Layer 2 (failure classifier).

Quantifies the multi-turn tool-call FORMAT failure on Qwen3.6-35B (and any
OpenAI-compatible Atlas endpoint) with confidence intervals, and classifies each
failure as a PARSING problem vs a GENERATION (decode/inference) problem.

Why two layers:
  - Layer 1 gives a validity RATE with a Wilson 95% CI, so config A/Bs are
    distinguishable above N=10 noise (the reason this harness exists).
  - Layer 2 reads the RAW model text on every non-ok turn and bins it:
      * parser_miss     -> model emitted well-formed <tool_call><function=…>,
                           but Atlas parsed nothing  => PARSING bug
      * malformed_opener-> model emitted <parameter…>/<arg…>/bare <function…>
                           (wrong dialect)            => GENERATION/decode
      * runaway         -> finish=length, no call
      * text_answer     -> prose, no tool tags
    If failures are overwhelmingly `malformed_opener`, parsing is exonerated and
    the defect is upstream (decode/inference) — confirm with logit_probe.py /
    dump_analyze.py.

Usage:
  python3 bench/tool_format_eval/harness.py --n 50
  python3 bench/tool_format_eval/harness.py --n 50 --url http://localhost:8888 \
      --model nvidia/Qwen3.6-35B-A3B-NVFP4
"""
import argparse
import json
import math
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VALID_TOOLS = {"list_files", "read_file", "write_file", "run_command"}

TOOLDEFS = [
    ("list_files", "List files in a directory", {"path": {"type": "string"}}, ["path"]),
    ("read_file", "Read a file's contents", {"path": {"type": "string"}}, ["path"]),
    ("write_file", "Write content to a file",
     {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    ("run_command", "Run a shell command", {"command": {"type": "string"}}, ["command"]),
]


def tools(n):
    return [{"type": "function", "function": {
        "name": a, "description": b,
        "parameters": {"type": "object", "properties": c, "required": d}}}
        for a, b, c, d in TOOLDEFS[:n]]


# A multi-turn agentic context: one completed tool call + its result, model must
# now emit the NEXT tool call. This is the turn that degenerates under thinking.
def context():
    return [
        {"role": "system", "content": "You are a coding agent. Use tools. One action at a time."},
        {"role": "user", "content": "Read src/main.rs and tell me what it does."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "list_files", "arguments": '{"path":"src"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "list_files",
         "content": '["src/main.rs","src/lib.rs"]'},
    ]


def post(url, body, timeout=180):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# Well-formed qwen3_coder call the parser SHOULD extract.
WELLFORMED = re.compile(r"<tool_call>\s*<function=\w+>")
BADOPENER = re.compile(r"<\s*(parameter|arg|param)\b|<function\s+name=|<function=\w+>(?!.*</tool_call>)", re.S)


def classify(resp):
    c = resp["choices"][0]
    m = c["message"]
    fr = c["finish_reason"]
    content = m.get("content") or ""
    tcs = m.get("tool_calls") or []
    # ok: at least one known tool, valid JSON args, and not truncated.
    if tcs and fr != "length":
        good = True
        for t in tcs:
            if t["function"]["name"] not in VALID_TOOLS:
                good = False
            else:
                try:
                    json.loads(t["function"]["arguments"])
                except Exception:
                    good = False
        if good:
            return "ok"
        return "wrong_tool"
    # No usable tool call — inspect the raw text the model produced.
    if WELLFORMED.search(content):
        return "parser_miss"          # PARSING bug
    if BADOPENER.search(content):
        return "malformed_opener"     # GENERATION/decode
    if fr == "length":
        return "runaway"
    if content.strip():
        return "text_answer"
    return "empty"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def trial(url, model, ntools, thinking):
    body = {"model": model, "messages": context(), "tools": tools(ntools),
            "max_tokens": 400, "temperature": 0.7}
    body["chat_template_kwargs"] = {"enable_thinking": thinking}
    try:
        return classify(post(url, body))
    except Exception as e:
        return f"error:{type(e).__name__}"


def run_cell(url, model, ntools, thinking, n, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(lambda _: trial(url, model, ntools, thinking), range(n)))
    hist = {}
    for o in outs:
        hist[o] = hist.get(o, 0) + 1
    ok = hist.get("ok", 0)
    lo, hi = wilson(ok, n)
    return ok, n, lo, hi, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tool-counts", default="2,4")
    a = ap.parse_args()
    counts = [int(x) for x in a.tool_counts.split(",")]

    print(f"# Tool-format eval — {a.model}  N={a.n}/cell  workers={a.workers}")
    print(f"{'cell':<28}{'valid':>10}{'95% CI':>16}   failure-mode breakdown")
    print("-" * 100)
    for ntools in counts:
        for thinking in (True, False):
            ok, n, lo, hi, hist = run_cell(a.url, a.model, ntools, thinking, a.n, a.workers)
            fails = {k: v for k, v in sorted(hist.items(), key=lambda x: -x[1]) if k != "ok"}
            cell = f"tools={ntools} thinking={'ON' if thinking else 'OFF'}"
            ci = f"[{lo*100:4.0f},{hi*100:4.0f}]%"
            print(f"{cell:<28}{ok}/{n:<8}{ci:>16}   {fails}")
    print("-" * 100)
    print("Read: high `malformed_opener` share => GENERATION/decode (not parsing).")
    print("      high `parser_miss` share      => PARSING bug.")


if __name__ == "__main__":
    main()

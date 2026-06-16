# SPDX-License-Identifier: AGPL-3.0-only
"""Tool-format eval harness — Layer 3a: within-Atlas logit decision-point probe.

Answers "parsing vs inference" WITHOUT an external reference, using the
OpenAI `top_logprobs` field. For each generated token it asks: at the moment the
model commits to narration / a malformed opener, how did Atlas's own next-token
distribution RANK the correct structural token (`<tool_call>` / `<` / `function`)?

  - If the correct structural token is TOP-1 / high-prob but a wrong token was
    sampled => the forward pass favored the right answer; failure is sampling/
    decode DYNAMICS, not the logits.
  - If the correct structural token is BURIED (low rank/logprob) => Atlas's
    forward pass is mis-ranking => INFERENCE/decode issue. Confirm the *cause*
    (model vs Atlas bias) with dump_analyze.py (ATLAS_LOGIT_DUMP raw_topk vs bias).

This is parser-free: it inspects the token stream the model actually produced,
upstream of any tool-call parsing.

Usage:
  python3 bench/tool_format_eval/logit_probe.py --runs 8 --top 10
"""
import argparse
import json
import urllib.request

# Tokens that (as a prefix) begin a CORRECT qwen3_coder tool call.
STRUCT_PREFIXES = ("<tool_call", "<tool", "<", "tool_call", "function", "<function")


def context(ntools):
    defs = [
        ("list_files", "List files", {"path": {"type": "string"}}, ["path"]),
        ("read_file", "Read a file", {"path": {"type": "string"}}, ["path"]),
        ("write_file", "Write a file",
         {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        ("run_command", "Run a shell command", {"command": {"type": "string"}}, ["command"]),
    ][:ntools]
    tools = [{"type": "function", "function": {
        "name": a, "description": b,
        "parameters": {"type": "object", "properties": c, "required": d}}}
        for a, b, c, d in defs]
    msgs = [
        {"role": "system", "content": "You are a coding agent. Use tools. One action at a time."},
        {"role": "user", "content": "Read src/main.rs and tell me what it does."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "list_files", "arguments": '{"path":"src"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "list_files",
         "content": '["src/main.rs","src/lib.rs"]'},
    ]
    return msgs, tools


def post(url, body, timeout=180):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def struct_rank(top_logprobs):
    """Best (rank, logprob) among candidates whose token begins a tool-call opener."""
    for rank, cand in enumerate(top_logprobs):
        tok = cand["token"].strip()
        if any(tok.startswith(p) or p.startswith(tok) and tok for p in ("<tool_call", "<", "<function")):
            if tok in ("<", "<tool_call", "<tool", "<function", "<tool_call>", "<function="):
                return rank, cand["logprob"], cand["token"]
    return None


def probe_run(url, model, ntools, top, thinking):
    msgs, tools = context(ntools)
    body = {"model": model, "messages": msgs, "tools": tools, "max_tokens": 24,
            "temperature": 1.0, "logprobs": True, "top_logprobs": top,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    d = post(url, body)
    c = d["choices"][0]
    lp = (c.get("logprobs") or {}).get("content") or []
    chosen = [t["token"] for t in lp]
    # Decision point = first content token after </think> (index 0 of the
    # returned trace, since thinking is stripped from `content`). Also scan for
    # the first '<'-tag the model opens.
    rows = []
    for i, t in enumerate(lp[:12]):
        sr = struct_rank(t.get("top_logprobs", []))
        rows.append((i, t["token"], round(t["logprob"], 2),
                     None if sr is None else (sr[0], round(sr[1], 2))))
    return chosen, rows, c["finish_reason"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--model", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--ntools", type=int, default=2)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--thinking", default="on")
    a = ap.parse_args()
    think = a.thinking.lower() in ("on", "true", "1")
    print(f"# Layer-3a logit probe — {a.model}  thinking={'ON' if think else 'OFF'}  "
          f"tools={a.ntools}  top_logprobs={a.top}")
    print("At pos 0 (first post-</think> token): rank/logprob of the <tool_call> opener vs chosen.\n")
    pos0_struct_top1 = 0
    for r in range(a.runs):
        chosen, rows, fr = probe_run(a.url, a.model, a.ntools, a.top, think)
        p0 = rows[0] if rows else None
        if p0 and p0[3] is not None and p0[3][0] == 0:
            pos0_struct_top1 += 1
        first = "".join(chosen[:10]).replace("\n", "\\n")
        sr = "n/a" if not p0 or p0[3] is None else f"rank{p0[3][0]} lp{p0[3][1]}"
        print(f"run {r+1}: chose0={p0[1]!r:>10} lp{p0[2]:<6} | <tool_call> opener: {sr:<16} "
              f"| start={first!r} | finish={fr}")
    print(f"\n<tool_call>-opener was TOP-1 at the decision point in "
          f"{pos0_struct_top1}/{a.runs} runs.")
    print("low share => forward pass disfavors the structural opener => inference/decode, "
          "not parsing.")


if __name__ == "__main__":
    main()

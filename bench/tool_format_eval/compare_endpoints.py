# SPDX-License-Identifier: AGPL-3.0-only
"""Tool-format eval harness — Layer 4: Atlas vs vLLM forward-pass divergence.

The decisive "inference numerics vs model behavior" test, run over two
OpenAI-compatible endpoints (Atlas + your vLLM) — no dump files, no token-id
alignment. Both decode the SAME prompt GREEDILY (temperature 0) with
`top_logprobs`; we diff their per-position top-1 token (string) and top-K overlap
until the first divergence (the metal_kv_kld_compare.py convention: greedy paths
share context only while their argmax agrees).

Interpretation at the post-`</think>` decision region:
  - top-1 tokens AGREE between Atlas and vLLM, both narrate/drift
        => the bad distribution is the MODEL's genuine behavior; Atlas inference
           is faithful. vLLM "works" for a different reason (parser/format/post-
           processing), NOT because its forward pass is better.
  - top-1 DIVERGES EARLY (Atlas ranks the wrong token where vLLM ranks
    `<tool_call>` / the right token)
        => Atlas's forward pass (NVFP4 quant kernels / KV) is numerically off
           => a real Atlas INFERENCE bug.

Caveat: if Atlas serves NVFP4 and vLLM serves FP8, some divergence is precision,
not a bug — prefer matched precision when possible; otherwise read the *pattern*
(does Atlas uniquely disfavor the structural opener?) not tiny logprob deltas.

Usage:
  python3 bench/tool_format_eval/compare_endpoints.py \
      --atlas-url http://localhost:8888 --atlas-model nvidia/Qwen3.6-35B-A3B-NVFP4 \
      --vllm-url  http://OTHER:8000   --vllm-model  Qwen/Qwen3.6-35B-A3B-FP8 \
      --top 8 --thinking on --max-tokens 40
"""
import argparse
import json
import urllib.request


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


def trace(url, model, ntools, top, thinking, max_tokens):
    """Greedy decode with top_logprobs; return list of (chosen, [(tok,logprob)...])."""
    msgs, tools = context(ntools)
    body = {"model": model, "messages": msgs, "tools": tools,
            "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": True, "top_logprobs": top,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    lp = (d["choices"][0].get("logprobs") or {}).get("content") or []
    return [(t["token"], [(x["token"], round(x["logprob"], 2))
                          for x in t.get("top_logprobs", [])]) for t in lp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-url", default="http://localhost:8888")
    ap.add_argument("--atlas-model", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--vllm-model", required=True)
    ap.add_argument("--ntools", type=int, default=4)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--thinking", default="on")
    ap.add_argument("--max-tokens", type=int, default=40)
    a = ap.parse_args()
    think = a.thinking.lower() in ("on", "true", "1")

    A = trace(a.atlas_url, a.atlas_model, a.ntools, a.top, think, a.max_tokens)
    V = trace(a.vllm_url, a.vllm_model, a.ntools, a.top, think, a.max_tokens)
    n = min(len(A), len(V))
    print(f"# Layer-4 Atlas↔vLLM top_logprobs diff (greedy)  thinking={'ON' if think else 'OFF'}")
    print(f"# atlas={a.atlas_model}  vllm={a.vllm_model}  comparing {n} positions\n")
    print(f"{'pos':>3} {'atlas_top1':>14} {'vllm_top1':>14} {'=':>2} {'topK_jaccard':>12}")
    agree = 0
    diverged = None
    for i in range(n):
        at, av = A[i][0], V[i][0]
        sa = {t for t, _ in A[i][1]}
        sv = {t for t, _ in V[i][1]}
        jac = len(sa & sv) / max(1, len(sa | sv))
        same = at == av
        agree += int(same)
        print(f"{i:>3} {at!r:>14} {av!r:>14} {'=' if same else 'X':>2} {jac:>12.2f}")
        if not same and diverged is None:
            diverged = i
            # show where the wrong/right structural token ranks on each side
            ar = next((r for r, (t, _) in enumerate(A[i][1]) if t.strip().startswith("<")), None)
            vr = next((r for r, (t, _) in enumerate(V[i][1]) if t.strip().startswith("<")), None)
            print(f"    -> FIRST top-1 divergence at pos {i}. "
                  f"'<'-tag rank: atlas={ar} vllm={vr}")
    print(f"\ntop-1 agreement: {agree}/{n}"
          + (f"  (first divergence at pos {diverged})" if diverged is not None else "  (full agreement)"))
    print("high agreement / both drift => model behavior, Atlas inference faithful "
          "(vLLM wins on parser/format).")
    print("early divergence, atlas uniquely disfavors '<' => Atlas forward-pass/quant bug.")


if __name__ == "__main__":
    main()

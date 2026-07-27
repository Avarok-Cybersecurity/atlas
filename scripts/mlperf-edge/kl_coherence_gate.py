#!/usr/bin/env python3
"""Fold gate: coherence + KL logit drift + output-divergence, baseline vs candidate serve.

Usage: python3 kl_coherence_gate.py <baseline_port> <candidate_port> [out.json]

Greedy (temp 0). For an output-neutral decode change both legs must emit the same tokens and
mean top-logprob KL must be ~0. A numeric change is QUANTIFIED here (KL, first-divergence pos)
so we fold on measured drift, not vibes.

PASS criteria (printed): coherent (no degeneration + valid tool call), token-match >= 0.99 of
positions, mean_KL < 1e-3. Any fail => DO NOT FOLD.
"""
import json
import sys
import urllib.request

BASE, CAND = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "/workspace/.wt-decode-fold/gate_result.json"

COHERE_PROMPTS = [
    "In exactly three sentences, explain what a KV cache stores during autoregressive decoding.",
    "Write a short Python function that returns the nth Fibonacci number iteratively.",
    "List three reasons a speculative draft token gets rejected during verification.",
]
TOOL_PROMPT = "What is the weather in Paris? Use the get_weather tool."
TOOLS = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get current weather for a location",
          "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                         "required": ["location"]}}}]


def call(port, messages, tools=None, logprobs=False, max_tokens=200):
    body = {"model": "qwen", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "seed": 42}
    if tools:
        body["tools"] = tools
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 10
    req = urllib.request.Request(f"http://0.0.0.0:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def token_logprobs(resp):
    """Return list of {tok, top:{tok:logprob}} per generated position, or None if unsupported."""
    try:
        content = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, TypeError):
        return None
    out = []
    for c in content:
        top = {t["token"]: t["logprob"] for t in c.get("top_logprobs", [])}
        out.append({"tok": c["token"], "top": top})
    return out


def kl(p_lp, q_lp):
    """KL(P||Q) over the union of top tokens, from logprob dicts. Missing => floor.

    BOTH sides are renormalized over the shared support. Normalizing P but not Q adds a
    constant -log(sum p) to every position -- the top-k logprobs only carry ~94% of the
    mass, so identical inputs scored ~0.061 instead of 0. That made this gate's
    `mean_kl < 1e-3` PASS threshold unreachable even for a byte-identical config, i.e.
    the gate could only ever return FAIL. Verified after the fix: KL(p, p) == 0.0
    exactly, on two independent controls (same-serve A/B and a config-identical
    re-serve). Found on dgx2, 2026-07-25, during the fp8-KV A/B.
    """
    import math as _m
    toks = set(p_lp) | set(q_lp)
    FLOOR = -30.0
    ps = {t: _m.exp(p_lp.get(t, FLOOR)) for t in toks}
    qs = {t: _m.exp(q_lp.get(t, FLOOR)) for t in toks}
    zp = sum(ps.values()) or 1.0
    zq = sum(qs.values()) or 1.0
    d = 0.0
    for t in toks:
        pv = ps[t] / zp
        qv = qs[t] / zq
        if pv <= 0:
            continue
        d += pv * (_m.log(pv) - _m.log(qv if qv > 0 else _m.exp(FLOOR)))
    return max(d, 0.0)


def degenerate(text):
    if not text or len(text) < 5:
        return True
    words = text.split()
    if len(words) > 8 and len(set(words)) / len(words) < 0.25:  # heavy repetition
        return True
    return False


result = {"coherent": True, "tool_ok": False, "positions": 0, "matched": 0,
          "first_divergence": None, "mean_kl": 0.0, "max_kl": 0.0, "notes": []}
kls = []
for p in COHERE_PROMPTS:
    msgs = [{"role": "user", "content": p}]
    rb = call(BASE, msgs, logprobs=True)
    rc = call(CAND, msgs, logprobs=True)
    tb = rb["choices"][0]["message"]["content"] or ""
    tc = rc["choices"][0]["message"]["content"] or ""
    if degenerate(tc):
        result["coherent"] = False
        result["notes"].append(f"CANDIDATE degenerate on: {p[:40]}")
    lb, lc = token_logprobs(rb), token_logprobs(rc)
    if lb and lc:
        n = min(len(lb), len(lc))
        for i in range(n):
            result["positions"] += 1
            if lb[i]["tok"] == lc[i]["tok"]:
                result["matched"] += 1
            elif result["first_divergence"] is None:
                result["first_divergence"] = i
            k = kl(lb[i]["top"], lc[i]["top"])
            kls.append(k)
    else:
        # logprobs unsupported -> fall back to byte-identical text check
        result["notes"].append("logprobs unsupported; byte-identical fallback")
        result["positions"] += 1
        result["matched"] += 1 if tb == tc else 0

# tool-call smoke on candidate
rt = call(CAND, [{"role": "user", "content": TOOL_PROMPT}], tools=TOOLS, max_tokens=200)
tc_msg = rt["choices"][0]["message"]
calls = tc_msg.get("tool_calls") or []
result["tool_ok"] = any(c.get("function", {}).get("name") == "get_weather" for c in calls)
if not result["tool_ok"]:
    result["notes"].append(f"no valid get_weather tool call: {str(tc_msg)[:120]}")

if kls:
    result["mean_kl"] = sum(kls) / len(kls)
    result["max_kl"] = max(kls)
match_frac = result["matched"] / result["positions"] if result["positions"] else 0.0
result["match_frac"] = match_frac

PASS = (result["coherent"] and result["tool_ok"] and match_frac >= 0.99
        and result["mean_kl"] < 1e-3)
result["VERDICT"] = "PASS" if PASS else "FAIL"
with open(OUT, "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps({k: result[k] for k in
      ("VERDICT", "coherent", "tool_ok", "match_frac", "mean_kl", "max_kl",
       "first_divergence", "positions", "notes")}, indent=2))

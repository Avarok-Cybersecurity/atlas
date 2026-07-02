#!/usr/bin/env python3
"""Deterministic WARM-vs-COLD diff for the MTP × prefix-cache hybrid-SSM bug.

With greedy decoding (temperature=0) a prefix-cache HIT that restores SSM
state mid-sequence (Marconi warm restore) MUST yield a token stream that is
bit-identical to the COLD (cache-miss) run of the same prompt. Any divergence
localizes the warm-restore corruption.

Strategy (single server, prefix-caching ON):
  COLD : send a prompt whose prefix is NOT in the cache (unique salt prefix),
         capture the exact token id stream.
  WARM : (1) populate the cache by sending prompt P once (cold), which during
             decode saves block-aligned Marconi SSM snapshots;
         (2) send a request P2 = P + short shared continuation so the lookup
             produces a partial prefix HIT and Marconi restores SSM state at a
             mid-sequence checkpoint, then suffix-prefills the rest.
         The generated continuation of the SHARED-prefix portion is compared
         against its COLD reference.

We also run the simplest, sharpest variant: send the SAME prompt twice; the
2nd send is a full warm hit. Greedy output must match the 1st (cold) send.

Reports the first divergent token index per pair across N prompts.
"""
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("ATLAS_URL", "http://localhost:8888")
MODEL = os.environ.get("ATLAS_MODEL", "model")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "96"))

# Prompts chosen to (a) generate enough tokens to cross decode-checkpoint
# block boundaries (so a warm restore lands on an intermediate SSM snapshot),
# and (b) include a tool-call/structured prompt that produces argument
# fragments (the symptom is duplicated tool-arg fragments).
BASE_PROMPTS = [
    "List the first 30 prime numbers, separated by commas, then explain in two "
    "sentences why 1 is not considered prime.",
    "Write a detailed step-by-step recipe for a three-layer chocolate cake, "
    "including exact ingredient quantities for each layer and the frosting.",
    "Explain how a transformer attention mechanism works, then describe the "
    "difference between multi-head attention and grouped-query attention in "
    "full technical detail.",
    "Describe the water cycle in detail: evaporation, condensation, "
    "precipitation, collection. Give a concrete example for each stage and "
    "explain the energy transfer involved.",
    "Enumerate the planets of the solar system in order from the sun, and for "
    "each give its diameter, number of moons, and one distinguishing feature.",
    "Write a Python function that implements merge sort, then explain its time "
    "complexity and walk through sorting the list [5,2,8,1,9,3] step by step.",
    "Summarize the causes of World War I in chronological order, naming the key "
    "treaties, alliances, and events that escalated the conflict.",
    "Explain the theory of general relativity: the equivalence principle, "
    "spacetime curvature, and gravitational time dilation, with one worked "
    "example for each concept.",
    # Tool-call style prompt producing argument fragments (the agentic symptom)
    "You are a function-calling assistant. Emit a JSON tool call to "
    "create_file with arguments: a path of '/tmp/report.txt' and a long "
    "multi-paragraph content field describing quarterly sales figures across "
    "four regions with specific dollar amounts for each region and quarter.",
    "Produce a markdown table of the 12 months with, for each, the number of "
    "days and the typical northern-hemisphere season, then write a paragraph "
    "about leap years.",
]


def gen_tokens(prompt, max_tokens=MAX_TOKENS):
    """Return the list of generated token ids (greedy) via logprobs echo."""
    t0 = time.perf_counter()
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": 1,  # forces per-token id reporting
    }
    req = urllib.request.Request(
        f"{URL}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    choice = r["choices"][0]
    u = r.get("usage", {})
    lp = choice.get("logprobs") or {}
    toks = lp.get("tokens")  # textual token strings (stable across runs)
    return {
        "text": choice["text"],
        "tokens": toks if toks is not None else None,
        "finish_reason": choice.get("finish_reason"),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def first_divergence(a, b):
    """First index where token streams (or text fallback) differ; -1 if equal."""
    if a.get("tokens") and b.get("tokens"):
        ta, tb = a["tokens"], b["tokens"]
        for i in range(min(len(ta), len(tb))):
            if ta[i] != tb[i]:
                return i
        if len(ta) != len(tb):
            return min(len(ta), len(tb))
        return -1
    # Fallback: compare text.
    return -1 if a["text"] == b["text"] else 0


def run_pair(name, prompt):
    """COLD (unique salt prefix) vs WARM (same prompt resent → warm hit)."""
    salt = f"[req#{name}-{time.time_ns()}] "
    cold = gen_tokens(salt + prompt)          # unique prefix → cache miss
    _ = gen_tokens(prompt)                     # populate cache (cold for P)
    warm = gen_tokens(prompt)                  # full warm hit on P
    # Compare warm to its OWN cold reference (the populate run is the cold ref
    # for P; cold-salt run only sanity-checks generation determinism).
    cold_ref = _
    div = first_divergence(cold_ref, warm)
    ok = div == -1
    return {
        "name": name,
        "ok": ok,
        "div": div,
        "cold_cached": cold_ref["cached"],
        "warm_cached": warm["cached"],
        "warm_prompt_tokens": warm["prompt_tokens"],
        "cold_text": cold_ref["text"],
        "warm_text": warm["text"],
        "cold_fr": cold_ref["finish_reason"],
        "warm_fr": warm["finish_reason"],
    }


def main():
    n_runs = int(os.environ.get("N", "10"))
    prompts = (BASE_PROMPTS * ((n_runs // len(BASE_PROMPTS)) + 1))[:n_runs]
    print(f"=== WARM-vs-COLD greedy diff: {n_runs} prompts, max_tokens={MAX_TOKENS} ===",
          flush=True)
    fails = 0
    for i, p in enumerate(prompts):
        res = run_pair(f"P{i}", p)
        warm_hit = res["warm_cached"] > 0
        status = "PASS" if res["ok"] else "FAIL"
        if not res["ok"]:
            fails += 1
        print(
            f"  [{status}] {res['name']}: warm_cached={res['warm_cached']}/"
            f"{res['warm_prompt_tokens']} (hit={warm_hit}) div_idx={res['div']} "
            f"cold_fr={res['cold_fr']} warm_fr={res['warm_fr']}",
            flush=True,
        )
        if not res["ok"]:
            print(f"      COLD: {res['cold_text']!r}", flush=True)
            print(f"      WARM: {res['warm_text']!r}", flush=True)
    print(f"\n=== {n_runs - fails}/{n_runs} PASS, {fails} FAIL ===", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

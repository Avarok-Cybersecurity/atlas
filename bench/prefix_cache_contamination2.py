#!/usr/bin/env python3
"""Sharper cross-conversation contamination check.

For 3 independent "victim" prompts (conv B, C, D), each with a different
first-token (guaranteeing distinct session_hash from ALPHA):
  1. Fire victim cold on head-PC and head-no-PC (via worker).
     Output must match — establishes baseline that head-PC with an EMPTY
     cache produces identical output to worker.
  2. Pollute head-PC's cache with ALPHA repeated K times.
  3. Fire the exact same victim again on head-PC.
     Output must STILL match worker's cold output. Any divergence = a
     leak from ALPHA's state into the victim's forward pass.

Also runs a "shared-persona" (GAMMA-style) victim whose prompt shares
the persona-prefix tokens with ALPHA — tests that partial prefix match +
snapshot session gate do not leak recurrent SSM state across the
diverging suffix.
"""
import json
import sys
import time
import urllib.request
import os

HEAD = "http://localhost:8888"
WORKER = os.environ.get("ATLAS_WORKER_URL", "http://127.0.0.1:8888")
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
MAX_TOKENS = 48

ALPHA = (
    "You are a pirate named Captain Blackbeard. Answer every question "
    "in exaggerated pirate speech with 'arrr' and 'matey'.\n\n"
    "Q: What is 7 plus 12?\nA:"
)

VICTIMS = {
    # Shares ZERO prefix with ALPHA (different first token)
    "B_math_tutor": (
        "As a formal mathematics tutor, state the result concisely.\n\n"
        "Q: What is 25 minus 9?\nA:"
    ),
    # Shares ZERO prefix with ALPHA
    "C_chef": (
        "Speaking as a French chef, answer in one sentence.\n\n"
        "Q: Name three fruits.\nA:"
    ),
    # Shares the ENTIRE persona prefix with ALPHA, diverges only at
    # the question. Tests the partial-prefix match case.
    "D_shared_persona": (
        "You are a pirate named Captain Blackbeard. Answer every question "
        "in exaggerated pirate speech with 'arrr' and 'matey'.\n\n"
        "Q: Name three fruits.\nA:"
    ),
}


def gen(url, prompt):
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    u = r["usage"]
    return {
        "text": r["choices"][0]["text"],
        "finish_reason": r["choices"][0]["finish_reason"],
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def main():
    results = []

    # Phase 1: cold baseline for every victim on BOTH servers
    #          (before any ALPHA warming on head).
    print("=== Phase 1: cold baseline for each victim, head vs worker ===", flush=True)
    baseline = {}
    for name, prompt in VICTIMS.items():
        h = gen(HEAD, prompt)
        w = gen(WORKER, prompt)
        match = h["text"] == w["text"]
        baseline[name] = {"head": h, "worker": w, "match": match}
        print(f"  {name}: head={h['completion_tokens']}tok cached={h['cached_tokens']}/{h['prompt_tokens']} "
              f"worker={w['completion_tokens']}tok → match={match}", flush=True)
        if not match:
            print(f"    HEAD:   {h['text']!r}", flush=True)
            print(f"    WORKER: {w['text']!r}", flush=True)
        results.append((f"Phase1 {name}: head==worker baseline", match))

    # Phase 2: warm head with ALPHA N times to pollute the cache.
    print("\n=== Phase 2: warm head with ALPHA 3× ===", flush=True)
    for i in range(3):
        r = gen(HEAD, ALPHA)
        print(f"  ALPHA run {i+1}: {r['completion_tokens']}tok cached={r['cached_tokens']}/{r['prompt_tokens']}", flush=True)

    # Phase 3: re-fire each victim on head AFTER ALPHA pollution.
    #          Head output must STILL match worker baseline (worker is clean —
    #          PC is disabled there).
    print("\n=== Phase 3: victim after ALPHA pollution — must match Phase-1 worker baseline ===", flush=True)
    for name, prompt in VICTIMS.items():
        h = gen(HEAD, prompt)
        baseline_w = baseline[name]["worker"]["text"]
        match = h["text"] == baseline_w
        baseline_h = baseline[name]["head"]["text"]
        match_h = h["text"] == baseline_h
        print(f"  {name} post-pollution: head={h['completion_tokens']}tok cached={h['cached_tokens']}/{h['prompt_tokens']}", flush=True)
        print(f"    == worker baseline: {match} | == head baseline: {match_h}", flush=True)
        if not match:
            print(f"    BASELINE: {baseline_w!r}", flush=True)
            print(f"    NOW:      {h['text']!r}", flush=True)
        results.append((f"Phase3 {name}: no cross-leak from ALPHA", match))

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {status}: {name}", flush=True)
    print(f"\n  {'ALL PASS — no cross-conversation contamination detected' if all_ok else 'CONTAMINATION DETECTED'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

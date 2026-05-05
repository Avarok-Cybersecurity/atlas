#!/usr/bin/env python3
"""Cross-conversation contamination check for --enable-prefix-caching.

Methodology:
  1. Warm the PC-enabled server with conversation ALPHA — a specific
     persona + question — sent N times. This populates the radix tree
     and (for SSM/hybrid models) the Marconi snapshot index with
     ALPHA's session_hash.
  2. Send conversation BETA — a totally different persona + question —
     to BOTH the PC-enabled head AND the PC-off worker. BETA's prompt
     shares NO prefix with ALPHA (different first tokens), so the
     session_hash differs and there must be no state reuse.
  3. Send conversation GAMMA — shares the SAME system prompt as ALPHA
     but has a DIFFERENT user question — to both servers. This tests
     that partial-prefix matching within a session doesn't leak state
     across the boundary where the prompts diverge.
  4. Compare head vs worker outputs character-for-character (temp=0).
     Any divergence = contamination bug.

Also prints head's prefix-cache metrics at the end so you can verify
(a) the warming actually produced hits, (b) BETA/GAMMA hits only where
expected.
"""
import argparse
import json
import sys
import time
import urllib.request
import os

HEAD_URL = "http://localhost:8888"
WORKER_URL = os.environ.get("ATLAS_WORKER_URL", "http://127.0.0.1:8888")
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"

ALPHA_PROMPT = (
    "You are a pirate named Captain Blackbeard. Answer every question "
    "in exaggerated pirate speech with 'arrr' and 'matey'.\n\n"
    "Q: What is 7 plus 12?\nA:"
)
# BETA shares ZERO prefix with ALPHA — different first token already.
BETA_PROMPT = (
    "As a formal mathematics tutor, concisely state the result.\n\n"
    "Q: What is 25 minus 9?\nA:"
)
# GAMMA shares ALPHA's persona but asks a different question.
# The system prompt prefix is identical, so walk() will match that prefix
# and the snapshot-index session_hash gate is the only thing preventing
# ALPHA's state from leaking into GAMMA.
GAMMA_PROMPT = (
    "You are a pirate named Captain Blackbeard. Answer every question "
    "in exaggerated pirate speech with 'arrr' and 'matey'.\n\n"
    "Q: Name three fruits.\nA:"
)

MAX_TOKENS = 64


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
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    text = r["choices"][0]["text"]
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {
        "text": text,
        "prompt_tokens": u["prompt_tokens"],
        "cached_tokens": cached,
        "wall_ms": round(ttft_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-n", type=int, default=3, help="Times to repeat ALPHA")
    args = ap.parse_args()

    # 1. Warm head with ALPHA — each should be byte-identical (temp=0).
    print(f"\n=== 1. Warm PC-enabled head with ALPHA ({args.warm_n}x) ===", flush=True)
    alpha_outs_head = []
    for i in range(args.warm_n):
        r = gen(HEAD_URL, ALPHA_PROMPT)
        alpha_outs_head.append(r)
        tag = "COLD" if i == 0 else f"warm{i}"
        print(f"  ALPHA head {tag}: wall={r['wall_ms']}ms cached={r['cached_tokens']}/{r['prompt_tokens']}", flush=True)
    # Also send ALPHA once on worker for reference.
    alpha_worker = gen(WORKER_URL, ALPHA_PROMPT)
    print(f"  ALPHA worker: wall={alpha_worker['wall_ms']}ms cached={alpha_worker['cached_tokens']}/{alpha_worker['prompt_tokens']}", flush=True)

    alpha_text_head = alpha_outs_head[0]["text"]
    alpha_texts_match = all(o["text"] == alpha_text_head for o in alpha_outs_head)
    alpha_head_vs_worker = alpha_text_head == alpha_worker["text"]
    print(f"  ALPHA head repeats identical: {alpha_texts_match}", flush=True)
    print(f"  ALPHA head == ALPHA worker: {alpha_head_vs_worker}", flush=True)

    # 2. BETA: zero shared prefix with ALPHA. Must match head vs worker.
    print("\n=== 2. BETA — zero-shared-prefix, new persona ===", flush=True)
    beta_head = gen(HEAD_URL, BETA_PROMPT)
    beta_worker = gen(WORKER_URL, BETA_PROMPT)
    print(f"  BETA head:   wall={beta_head['wall_ms']}ms cached={beta_head['cached_tokens']}/{beta_head['prompt_tokens']}", flush=True)
    print(f"  BETA worker: wall={beta_worker['wall_ms']}ms cached={beta_worker['cached_tokens']}/{beta_worker['prompt_tokens']}", flush=True)
    beta_match = beta_head["text"] == beta_worker["text"]
    print(f"  BETA head == BETA worker: {beta_match}", flush=True)
    if not beta_match:
        print(f"  HEAD text:   {beta_head['text']!r}", flush=True)
        print(f"  WORKER text: {beta_worker['text']!r}", flush=True)

    # 3. GAMMA: shared persona prefix with ALPHA, different question.
    #    Tests that walk() partial match + snapshot session gate do not leak.
    print("\n=== 3. GAMMA — shared persona prefix with ALPHA, different question ===", flush=True)
    gamma_head = gen(HEAD_URL, GAMMA_PROMPT)
    gamma_worker = gen(WORKER_URL, GAMMA_PROMPT)
    print(f"  GAMMA head:   wall={gamma_head['wall_ms']}ms cached={gamma_head['cached_tokens']}/{gamma_head['prompt_tokens']}", flush=True)
    print(f"  GAMMA worker: wall={gamma_worker['wall_ms']}ms cached={gamma_worker['cached_tokens']}/{gamma_worker['prompt_tokens']}", flush=True)
    gamma_match = gamma_head["text"] == gamma_worker["text"]
    print(f"  GAMMA head == GAMMA worker: {gamma_match}", flush=True)
    if not gamma_match:
        print(f"  HEAD text:   {gamma_head['text']!r}", flush=True)
        print(f"  WORKER text: {gamma_worker['text']!r}", flush=True)

    # 4. Re-fire ALPHA on head — must STILL match the first ALPHA output
    #    even after BETA and GAMMA have run through.
    print("\n=== 4. Re-fire ALPHA on head after BETA + GAMMA — must still match ===", flush=True)
    alpha_rerun = gen(HEAD_URL, ALPHA_PROMPT)
    print(f"  ALPHA head rerun: wall={alpha_rerun['wall_ms']}ms cached={alpha_rerun['cached_tokens']}/{alpha_rerun['prompt_tokens']}", flush=True)
    alpha_stable = alpha_rerun["text"] == alpha_text_head
    print(f"  ALPHA rerun == ALPHA original: {alpha_stable}", flush=True)
    if not alpha_stable:
        print(f"  ORIG: {alpha_text_head!r}", flush=True)
        print(f"  NEW:  {alpha_rerun['text']!r}", flush=True)

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    all_ok = alpha_texts_match and alpha_head_vs_worker and beta_match and gamma_match and alpha_stable
    results = [
        ("ALPHA head repeats identical (temp=0 sanity)", alpha_texts_match),
        ("ALPHA head == ALPHA worker",                   alpha_head_vs_worker),
        ("BETA head == BETA worker (no cross-leak)",     beta_match),
        ("GAMMA head == GAMMA worker (no prefix-share leak)", gamma_match),
        ("ALPHA rerun stable after BETA+GAMMA",          alpha_stable),
    ]
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}", flush=True)
    print(f"\n  {'ALL PASS — no cross-conversation contamination' if all_ok else 'FAILURES ABOVE'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

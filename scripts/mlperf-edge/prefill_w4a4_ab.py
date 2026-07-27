#!/usr/bin/env python3
"""Cold-prefill A/B probe for ATLAS_FP4_PREFILL (W4A4 native FP4 MMA dense-FFN prefill).

Prefill is the regime where FP4 activations can pay: M = seqlen >> 8, tiles are full
and the GEMM is compute-bound, unlike decode at M<=8 where activations are <1% of
traffic. This measures whether it actually pays, and what it costs in accuracy.

W4A4 prefill is LOSSY by construction (cos ~0.99 vs fp32), so unlike an
output-neutral change the greedy output hash is EXPECTED to differ. That makes the
token-match rate and the top-logprob KL the real gate, not the sha.

Each prompt is issued cold (a fresh unique preamble defeats the prefix cache and the
session hash), so TTFT is dominated by prefill rather than by a cache hit.

Usage: prefill_w4a4_ab.py <port> <tag> <out.json>
"""
import argparse
import json
import math
import statistics
import time
import urllib.request

MODEL = "centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf"

# Long, self-contained prompts. Prefill cost scales with these, so they are the
# signal; the generation is capped short to keep decode out of the measurement.
BODY = (
    "A paged KV cache stores per-layer key and value tensors in fixed-size blocks "
    "so that sequences of different lengths can share an allocator without "
    "fragmentation. Speculative decoding proposes several tokens from a cheaper "
    "drafter and verifies them in one target forward pass. NVFP4 packs two 4-bit "
    "E2M1 values per byte with an FP8 block scale every sixteen elements. A gated "
    "delta-rule layer carries a recurrent state that must be checkpointed before a "
    "speculative verify so it can be rolled back on rejection. "
)

QUESTIONS = [
    "Summarize the tradeoff between draft chain depth and acceptance rate in three sentences.",
    "Explain in three sentences why prefill is compute-bound while decode is bandwidth-bound.",
    "In three sentences, describe what a block scale contributes to a 4-bit quantized GEMM.",
    "Explain in three sentences what must be rolled back when a speculative draft is rejected.",
]
# Repeat counts chosen to land near ~1k / ~4k / ~8k prompt tokens.
REPEATS = [12, 48, 96]


def post(port, body, timeout=600):
    req = urllib.request.Request(
        f"http://0.0.0.0:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def kl(p_top, q_top):
    """KL(p||q) over the union of the two top-k supports, in nats."""
    keys = set(p_top) | set(q_top)
    floor = -30.0
    total = 0.0
    for k in keys:
        lp = p_top.get(k, floor)
        lq = q_top.get(k, floor)
        total += math.exp(lp) * (lp - lq)
    return max(total, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("tag")
    ap.add_argument("out")
    args = ap.parse_args()

    runs = []
    for rep in REPEATS:
        for qi, q in enumerate(QUESTIONS):
            # Unique preamble => cold every time (no prefix-cache hit, new session).
            uniq = f"[doc {args.tag}-{rep}-{qi}-{time.time_ns()}] "
            prompt = uniq + (BODY * rep) + "\n\n" + q
            t0 = time.time()
            resp = post(args.port, {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 48, "temperature": 0, "seed": 42,
                "logprobs": True, "top_logprobs": 20,
            })
            wall = (time.time() - t0) * 1000.0
            ch = resp["choices"][0]
            text = ch["message"]["content"] or ""
            usage = resp.get("usage", {})
            content = (ch.get("logprobs") or {}).get("content") or []
            tops = [{t["token"]: t["logprob"] for t in c.get("top_logprobs", [])}
                    for c in content]
            toks = [c["token"] for c in content]
            runs.append({
                "rep": rep, "qi": qi,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "wall_ms": wall, "text": text, "tokens": toks, "tops": tops,
            })
            print(f"[{args.tag}] rep={rep:<3} q={qi} ptok={usage.get('prompt_tokens')} "
                  f"wall={wall:8.1f}ms n={usage.get('completion_tokens')}")

    by_rep = {}
    for rep in REPEATS:
        w = [r["wall_ms"] for r in runs if r["rep"] == rep]
        by_rep[str(rep)] = {"median_wall_ms": statistics.median(w), "n": len(w)}
    out = {"tag": args.tag, "by_rep": by_rep, "runs": runs}
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    for rep, v in by_rep.items():
        print(f"[{args.tag}] rep={rep}: median wall {v['median_wall_ms']:.1f} ms")


if __name__ == "__main__":
    main()

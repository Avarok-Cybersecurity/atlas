#!/usr/bin/env python3
"""Warm multi-turn TTFT/TPOT probe for the snapshot-fold A/B.

One growing conversation. Turns >=2 are fully warm (prefix-cache hit + carried
drafter), which is the regime the MLPerf agentic workload lives in (987/1007 warm).
Greedy temp0/seed42, so `combined_sha` MUST match across the two legs -- the
snapshot fix is a lookup change and has to be output-neutral.

The >1024-token system prompt is load-bearing: the server derives the session from
a hash of the first min(1024, len) prompt tokens, so a shorter system prompt makes
every turn look like a NEW session (cold drafter) -- the artifact that poisoned an
earlier cold-prompt K=4 sweep.

Turn count is deliberately deeper than the 8-turn original: on strix the warm-TTFT
median moved but the p99 tail did NOT, because the probe never reached the context
depth the spikes live at. Reported `ctx_chars` tracks how deep each leg actually got.

Usage: warm_probe.py <port> <tag> <out.json> [--turns 16] [--maxtok 300]
"""
import argparse
import hashlib
import json
import statistics
import time
import urllib.request

SYSTEM = (
    "You are a precise, terse engineering assistant embedded in a code review "
    "tool. You answer with concrete technical detail and never speculate. "
    "When you are unsure you say so in one sentence. You keep answers under "
    "200 words unless asked otherwise. You always prefer measured numbers over "
    "adjectives, and you name the file and line whenever you refer to code. "
    "You never apologise, never restate the question, and never emit lists "
    "longer than five items. Your domain is GPU inference serving: paged "
    "attention, KV caches, speculative decoding, quantization formats, and "
    "CUDA kernel scheduling. You treat every claim about performance as "
    "requiring a measurement, and you distinguish an estimate from an "
    "observation explicitly. You write in plain prose. "
) * 12  # >1024 tokens on purpose -- see module docstring.

TURNS = [
    "Explain what a KV cache stores during autoregressive decoding and why paged allocation helps.",
    "Now explain how speculative decoding changes the number of target forward passes.",
    "Describe how a draft tree differs from a draft chain and when each wins.",
    "Walk through NVFP4 quantization: packing, block scales, and dequant in a GEMM epilogue.",
    "Explain why memory bandwidth bounds single-stream decode on unified-memory devices.",
    "Describe how prefix caching interacts with a recurrent SSM state across turns.",
    "Explain what rejection sampling preserves in speculative decoding and why.",
    "Summarize the trade-offs between deeper draft chains and wider draft trees.",
    "Explain how a gated delta-rule layer carries state and what a checkpoint must capture.",
    "Describe what shared-memory bank conflicts cost in a tiled GEMM and how padding fixes them.",
    "Explain why arithmetic intensity differs between prefill and decode for the same weights.",
    "Walk through what happens on a prefix-cache miss for turn N of a long conversation.",
    "Describe how CUDA graphs reduce launch overhead and when they cannot be used.",
    "Explain the difference between a roofline bound and a latency bound in a decode step.",
    "Describe how tensor-core tile shapes constrain the M dimension in a batched GEMV.",
    "Summarize what limits throughput once weights are already read once per token.",
]


def stream_turn(port, messages, max_tokens):
    body = json.dumps({
        "model": "qwen", "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "seed": 42, "stream": True,
    }).encode()
    req = urllib.request.Request(f"http://0.0.0.0:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    chunks = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                if ttft is None:
                    ttft = (time.time() - t0) * 1000.0
                ntok += 1
                chunks.append(delta["content"])
    total = (time.time() - t0) * 1000.0
    tpot = (total - ttft) / max(ntok - 1, 1) if ttft and ntok > 1 else 0.0
    return ttft or 0.0, tpot, ntok, "".join(chunks)


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("tag")
    ap.add_argument("out")
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--maxtok", type=int, default=300)
    args = ap.parse_args()

    messages = [{"role": "system", "content": SYSTEM}]
    runs = []
    combined = hashlib.sha256()
    for i in range(min(args.turns, len(TURNS))):
        messages.append({"role": "user", "content": TURNS[i]})
        ctx_chars = sum(len(m["content"]) for m in messages)
        ttft, tpot, ntok, text = stream_turn(args.port, messages, args.maxtok)
        messages.append({"role": "assistant", "content": text})
        combined.update(text.encode())
        warm = i >= 2
        runs.append({"turn": i, "warm": warm, "ttft": ttft, "tpot": tpot, "ntok": ntok,
                     "ctx_chars": ctx_chars, "sha": hashlib.sha256(text.encode()).hexdigest()[:16]})
        print(f"[{args.tag}] t{i:<2} {'warm' if warm else 'cold'}: "
              f"ttft={ttft:8.1f}ms tpot={tpot:6.2f}ms n={ntok:<4} ctx={ctx_chars}")

    warm_runs = [r for r in runs if r["warm"]]
    warm_ttfts = [r["ttft"] for r in warm_runs if r["ttft"] > 0]
    warm_tpots = [r["tpot"] for r in warm_runs if r["tpot"] > 0]
    summary = {
        "tag": args.tag,
        "turns": len(runs),
        "ttft_warm_median": statistics.median(warm_ttfts) if warm_ttfts else 0.0,
        "ttft_warm_mean": statistics.mean(warm_ttfts) if warm_ttfts else 0.0,
        "ttft_warm_p90": pct(warm_ttfts, 90),
        "ttft_warm_max": max(warm_ttfts) if warm_ttfts else 0.0,
        "tpot_warm_median": statistics.median(warm_tpots) if warm_tpots else 0.0,
        "tpot_warm_mean": statistics.mean(warm_tpots) if warm_tpots else 0.0,
        "max_ctx_chars": max(r["ctx_chars"] for r in runs) if runs else 0,
        "combined_sha": combined.hexdigest()[:16],
        "runs": runs,
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{args.tag}] WARM TTFT median={summary['ttft_warm_median']:.1f} "
          f"mean={summary['ttft_warm_mean']:.1f} p90={summary['ttft_warm_p90']:.1f} "
          f"max={summary['ttft_warm_max']:.1f} ms | TPOT median={summary['tpot_warm_median']:.2f}ms "
          f"| sha={summary['combined_sha']}")


if __name__ == "__main__":
    main()

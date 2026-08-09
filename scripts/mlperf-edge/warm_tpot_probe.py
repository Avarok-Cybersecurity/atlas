#!/usr/bin/env python3
"""Warm multi-turn TPOT probe. One growing conversation; turns >=3 are fully
warm (prefix-cache hit + carried drafter), which is the regime the MLPerf
agentic workload lives in (987/1007 warm). Greedy temp0/seed42.

The >1024-token system prompt is load-bearing: the server derives the session
from a hash of the first min(1024, len) prompt tokens, so a shorter system
prompt would make every turn look like a NEW session (cold drafter) — the
exact artifact that poisoned the earlier cold-prompt K=4 sweep.

Usage: warm_tpot_probe.py <port> <tag> <out.json> [--turns 8] [--maxtok 300]
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
) * 12  # >1024 tokens on purpose — see module docstring.

TURNS = [
    "Explain what a KV cache stores during autoregressive decoding and why paged allocation helps.",
    "Now explain how speculative decoding changes the number of target forward passes.",
    "Describe how a draft tree differs from a draft chain and when each wins.",
    "Walk through NVFP4 quantization: packing, block scales, and dequant in a GEMM epilogue.",
    "Explain why memory bandwidth bounds single-stream decode on unified-memory devices.",
    "Describe how prefix caching interacts with a recurrent SSM state across turns.",
    "Explain what rejection sampling preserves in speculative decoding and why.",
    "Summarize the trade-offs between deeper draft chains and wider draft trees.",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("tag")
    ap.add_argument("out")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--maxtok", type=int, default=300)
    args = ap.parse_args()

    messages = [{"role": "system", "content": SYSTEM}]
    runs = []
    combined = hashlib.sha256()
    for i in range(min(args.turns, len(TURNS))):
        messages.append({"role": "user", "content": TURNS[i]})
        ttft, tpot, ntok, text = stream_turn(args.port, messages, args.maxtok)
        messages.append({"role": "assistant", "content": text})
        combined.update(text.encode())
        warm = i >= 2
        runs.append({"turn": i, "warm": warm, "ttft": ttft, "tpot": tpot,
                     "ntok": ntok, "sha": hashlib.sha256(text.encode()).hexdigest()[:16]})
        print(f"[{args.tag}] t{i} {'warm' if warm else 'cold'}: "
              f"ttft={ttft:7.1f}ms tpot={tpot:6.2f}ms n={ntok}")

    warm_tpots = [r["tpot"] for r in runs if r["warm"] and r["tpot"] > 0]
    summary = {
        "tag": args.tag,
        "tpot_warm_median": statistics.median(warm_tpots) if warm_tpots else 0.0,
        "tpot_warm_mean": statistics.mean(warm_tpots) if warm_tpots else 0.0,
        "combined_sha": combined.hexdigest()[:16],
        "runs": runs,
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{args.tag}] WARM TPOT median={summary['tpot_warm_median']:.2f}ms "
          f"mean={summary['tpot_warm_mean']:.2f}ms sha={summary['combined_sha']}")


if __name__ == "__main__":
    main()

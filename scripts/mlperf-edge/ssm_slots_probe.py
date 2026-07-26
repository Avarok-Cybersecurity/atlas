#!/usr/bin/env python3
"""Multi-session SSM snapshot-eviction probe — the instrument for --ssm-cache-slots.

A single growing conversation CANNOT test slot pressure: it only ever occupies one
slot, so it never evicts anything. That is exactly why the earlier warm-TTFT probe
found the snapshot fix neutral yet could say nothing about the e2e TTFT p99 tail.

This probe interleaves S independent conversations round-robin, so by the time a
session's next turn arrives, S-1 other sessions have pushed snapshots into the pool.
With S > slots, anchors get evicted between turns and the turn pays a full SSM
replay -- the signature the e2e tail shows (one 26.6 s outlier, p99 5354 ms).

Reports warm-turn TTFT p50/p90/p99, which is what --ssm-cache-slots should move.

Usage: ssm_slots_probe.py <port> <tag> <out.json> [--sessions 24] [--turns 6]
"""
import argparse
import json
import statistics
import time
import urllib.request

MODEL = "centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf"

# >1024 tokens so each session hashes to a distinct, stable session id. The {sid}
# makes every session's prefix genuinely distinct rather than a shared prefix.
SYSTEM_TMPL = (
    "You are engineering assistant number {sid}, embedded in a code review tool. "
    "You answer with concrete technical detail and never speculate. You keep "
    "answers under 200 words. You prefer measured numbers over adjectives, and "
    "you name the file and line whenever you refer to code. You never apologise "
    "and never restate the question. Your domain is GPU inference serving: paged "
    "attention, KV caches, speculative decoding, quantization formats, and CUDA "
    "kernel scheduling. You treat every performance claim as requiring a "
    "measurement, and you distinguish an estimate from an observation. "
) * 12

TURNS = [
    "Explain what a KV cache stores during autoregressive decoding.",
    "Now explain how speculative decoding changes the target forward-pass count.",
    "Describe how prefix caching interacts with a recurrent SSM state across turns.",
    "Explain why memory bandwidth bounds single-stream decode.",
    "Describe what a snapshot must capture for a gated delta-rule layer.",
    "Summarize the trade-off between draft chain depth and acceptance.",
]


def turn(port, messages, max_tokens=64):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 42, "stream": True}).encode()
    req = urllib.request.Request(f"http://0.0.0.0:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = []
    with urllib.request.urlopen(req, timeout=900) as r:
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
                chunks.append(delta["content"])
    return ttft or 0.0, "".join(chunks)


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("tag")
    ap.add_argument("out")
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--turns", type=int, default=6)
    args = ap.parse_args()

    convos = [[{"role": "system", "content": SYSTEM_TMPL.format(sid=s)}]
              for s in range(args.sessions)]
    runs = []
    # Round-robin: every session advances one turn before any session advances twice.
    for ti in range(min(args.turns, len(TURNS))):
        for si in range(args.sessions):
            convos[si].append({"role": "user", "content": TURNS[ti]})
            ttft, text = turn(args.port, convos[si])
            convos[si].append({"role": "assistant", "content": text})
            runs.append({"session": si, "turn": ti, "ttft": ttft, "warm": ti >= 1})
        done = [r["ttft"] for r in runs if r["turn"] == ti]
        print(f"[{args.tag}] turn {ti}: n={len(done)} p50={statistics.median(done):.0f} "
              f"p90={pct(done,90):.0f} max={max(done):.0f} ms", flush=True)

    warm = [r["ttft"] for r in runs if r["warm"] and r["ttft"] > 0]
    summary = {
        "tag": args.tag, "sessions": args.sessions, "turns": args.turns,
        "warm_n": len(warm),
        "warm_p50": statistics.median(warm) if warm else 0.0,
        "warm_p90": pct(warm, 90), "warm_p95": pct(warm, 95), "warm_p99": pct(warm, 99),
        "warm_max": max(warm) if warm else 0.0,
        "warm_mean": statistics.mean(warm) if warm else 0.0,
        "runs": runs,
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{args.tag}] WARM TTFT n={len(warm)} p50={summary['warm_p50']:.0f} "
          f"p90={summary['warm_p90']:.0f} p95={summary['warm_p95']:.0f} "
          f"p99={summary['warm_p99']:.0f} max={summary['warm_max']:.0f} ms")


if __name__ == "__main__":
    main()

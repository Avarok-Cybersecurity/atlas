#!/usr/bin/env python3
"""Streaming decode-gap measurement against an Atlas /v1/chat/completions endpoint.

Reports TTFT, per-chunk inter-arrival gaps (median / mean after warm-up), and the
derived decode tok/s, plus the server-attested usage.completion_tokens. Fingerprint
fields are printed so the number can be quoted later (measurement-discipline rule 1).
"""
import json, sys, time, urllib.request, statistics, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8890)
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--effort", default="low")
ap.add_argument("--prompt", default="code")
args = ap.parse_args()

PROMPTS = {
    "code": "Write a complete Rust implementation of an LRU cache with generic key and value types, "
            "using a HashMap and a doubly linked list of indices into a Vec arena. Include get, put, "
            "capacity handling, and unit tests. Explain each design decision briefly in comments.",
    "prose": "Explain in detail how the Marconi tail-checkpoint scheme for recurrent state caches works, "
             "why intermediate snapshots matter for warm turns, and what failure modes a replay seam can have.",
}
prompt = PROMPTS[args.prompt]

def one_run():
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": args.effort},
    }
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times = []
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    times.append(time.perf_counter())
                if ch[0].get("finish_reason"):
                    finish = ch[0]["finish_reason"]
    if len(times) < 20:
        return None
    ttft = times[0] - t0
    gaps = [b - a for a, b in zip(times[:-1], times[1:])]
    warm = gaps[5:]
    med = statistics.median(warm)
    mean = statistics.fmean(warm)
    ctoks = usage.get("completion_tokens") if usage else None
    wall_decode = times[-1] - times[0]
    server_tps = (ctoks - 1) / wall_decode if ctoks else None
    return dict(chunks=len(times), ttft_ms=ttft * 1000, gap_med_ms=med * 1000, gap_mean_ms=mean * 1000,
                tps_from_gaps=1.0 / mean, completion_tokens=ctoks, tps_server_tokens=server_tps, finish=finish)

print(f"FINGERPRINT port={args.port} model={args.model} prompt={args.prompt} max_tokens={args.max_tokens} "
      f"temp=0 effort={args.effort} stream=1 date={time.strftime('%Y-%m-%dT%H:%M:%S')}")
results = []
for i in range(args.repeats):
    r = one_run()
    print(f"run {i}: {r}")
    if r:
        results.append(r)
if results:
    meds = [r["gap_med_ms"] for r in results]
    tps = [r["tps_server_tokens"] or r["tps_from_gaps"] for r in results]
    print(f"SUMMARY n={len(results)} gap_median_ms={statistics.median(meds):.2f} "
          f"decode_tok_s_median={statistics.median(tps):.2f} ttft_ms={[round(r['ttft_ms']) for r in results]}")

#!/usr/bin/env python3
"""Queueing behaviour under a mixed-length burst: FIFO vs SLAI vs SLAI+co-dispatch.

Every timing here is CLIENT-SIDE, measured from the moment the request is
submitted — the server's `time_to_first_token_ms` starts at the beginning of a
request's PROCESSING and therefore hides queue wait entirely, which is what made
an earlier harness report cold_ratio=1.00 at C=4 (see EXL3_DECODE_PERF.md).

The burst is deliberately HETEROGENEOUS: one long prompt submitted FIRST, then
several short ones. FIFO must serve the long one first and every short request
waits behind it; SLAI selects the shortest pending prompts, so the short ones
should clear first and median TTFT should fall sharply while the long one's
TTFT gets no worse than the total prefill time (prefill is saturated — this
redistributes latency, it cannot create throughput).

Reports per-request TTFT and end-to-end wall, then median/p99 across the burst.
"""
import argparse, json, random, statistics, threading, time, urllib.request, datetime

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8899)
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--label", default="")
ap.add_argument("--long-tokens", type=int, default=8000)
ap.add_argument("--short-tokens", type=int, default=600)
ap.add_argument("--shorts", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--repeats", type=int, default=2)
args = ap.parse_args()

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
RATIO = 1.037


def make(n_tokens, rng):
    n = max(8, int(n_tokens / RATIO) - 12)
    return f"salt-{rng.randrange(10**9)}. Answer in one word:\n" + " ".join(rng.choice(WORDS) for _ in range(n))


def stream_one(prompt, tag, out, idx, t_submit):
    """Streamed POST; first byte of content is the honest TTFT from submission."""
    body = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens, "stream": True,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    ttft = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            chunk = raw[6:].strip()
            if chunk == b"[DONE]":
                break
            try:
                d = json.loads(chunk)
            except Exception:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t_submit
                ntok += 1
    out[idx] = {"tag": tag, "ttft_s": ttft if ttft is not None else float("nan"),
                "wall_s": time.perf_counter() - t_submit, "tokens": ntok}


print(f"FINGERPRINT probe_queueing label={args.label} port={args.port} long={args.long_tokens} "
      f"short={args.short_tokens} x{args.shorts} max_tokens={args.max_tokens} "
      f"date={datetime.datetime.utcnow().isoformat()}Z", flush=True)

agg = []
for rep in range(args.repeats):
    rng = random.Random(time.time_ns())
    prompts = [("long", make(args.long_tokens, rng))]
    prompts += [(f"short{i}", make(args.short_tokens, rng)) for i in range(args.shorts)]
    out = [None] * len(prompts)
    t0 = time.perf_counter()
    ts = []
    for i, (tag, p) in enumerate(prompts):
        t = threading.Thread(target=stream_one, args=(p, tag, out, i, t0))
        ts.append(t)
        t.start()
        time.sleep(0.002)   # submit in order, long first — FIFO's worst case
    [t.join() for t in ts]
    for o in out:
        print(f"  rep{rep} {o['tag']:<7} ttft={o['ttft_s']*1000:8.0f} ms  wall={o['wall_s']:6.2f} s  tok={o['tokens']}", flush=True)
    shorts = [o["ttft_s"] for o in out if o["tag"].startswith("short")]
    longs = [o["ttft_s"] for o in out if o["tag"] == "long"]
    burst_wall = max(o["wall_s"] for o in out)
    agg.append({"short_med": statistics.median(shorts), "long": longs[0],
                "all_med": statistics.median([o["ttft_s"] for o in out]),
                "all_max": max(o["ttft_s"] for o in out), "burst_wall": burst_wall})
    print(f"  rep{rep} short_med_ttft={statistics.median(shorts)*1000:.0f} ms  "
          f"long_ttft={longs[0]*1000:.0f} ms  burst_wall={burst_wall:.2f} s", flush=True)

m = lambda f: statistics.median([a[f] for a in agg])
print(f"SUMMARY {args.label} short_med_ttft_ms={m('short_med')*1000:.0f} long_ttft_ms={m('long')*1000:.0f} "
      f"all_med_ttft_ms={m('all_med')*1000:.0f} all_max_ttft_ms={m('all_max')*1000:.0f} "
      f"burst_wall_s={m('burst_wall'):.2f}", flush=True)

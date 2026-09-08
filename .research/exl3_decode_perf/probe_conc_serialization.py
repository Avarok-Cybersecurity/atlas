#!/usr/bin/env python3
"""Cells for ab_concurrency_serialization.sh — cross-stream serialization.

cold1/cold4 : ~3000-token unique prompts, max_tokens=16. TTFT dominated by
              prefill, which faults many PLE n-gram rows; `resolve` holds the
              PLE table mutex, so a lock-bound engine improves MORE at C=4.
dec1/dec4   : ~600-token prompts, max_tokens=400. Decode tok/s, where the
              Marconi decode checkpoints fire (every ATLAS_DECODE_CKPT_BLOCKS
              blocks of generated tokens, per sequence).
warm4       : cold4's prompts + a short suffix. Catches the cost of coarser
              checkpoint anchors — a longer replay tail on the next turn.

Prints one CELL line per cell and a SUMMARY. Ratios (C=4 vs C=1) are the point:
an absolute number here is one arm of an A/B, never a headline.
"""
import argparse, json, random, statistics, threading, time, urllib.request, datetime

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8899)
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--label", default="")
ap.add_argument("--repeats", type=int, default=2)
args = ap.parse_args()

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
RATIO = 1.037


def body(prompt, max_tokens):
    return {"model": args.model, "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "user", "content": prompt}]}


def post(prompt, max_tokens):
    t0 = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body(prompt, max_tokens)).encode(),
        headers={"content-type": "application/json"}), timeout=1800)
    d = json.load(r)
    u = d.get("usage", {})
    wall = time.perf_counter() - t0
    ttft = u.get("time_to_first_token_ms") or 0.0
    comp = u.get("completion_tokens") or 0
    dec_s = max(wall - ttft / 1000.0, 1e-6)
    return {"ttft_ms": ttft, "prompt_tokens": u.get("prompt_tokens"), "completion": comp,
            "decode_tok_s": (comp - 1) / dec_s if comp > 1 else 0.0, "wall_s": wall,
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


def make(n_tokens, rng):
    n = max(8, int(n_tokens / RATIO) - 12)
    return f"salt-{rng.randrange(10**9)}. Summarise in one word:\n" + " ".join(rng.choice(WORDS) for _ in range(n))


def cell(name, prompts, max_tokens):
    out = [None] * len(prompts)
    def go(i):
        out[i] = post(prompts[i], max_tokens)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(len(prompts))]
    [t.start() for t in ts]; [t.join() for t in ts]
    ttfts = [o["ttft_ms"] for o in out]
    decs = [o["decode_tok_s"] for o in out]
    print(f"CELL {args.label} {name} C={len(prompts)} prompt_tok={out[0]['prompt_tokens']} "
          f"ttft_ms={[round(t) for t in ttfts]} ttft_max={max(ttfts):.0f} "
          f"decode_tok_s={[round(d, 2) for d in decs]} agg_decode={sum(decs):.2f} "
          f"cached={[o['cached'] for o in out]}", flush=True)
    return {"ttft_max": max(ttfts), "ttft_med": statistics.median(ttfts), "agg_decode": sum(decs)}


print(f"FINGERPRINT probe_conc_serialization label={args.label} port={args.port} "
      f"date={datetime.datetime.utcnow().isoformat()}Z", flush=True)
rng = random.Random(time.time_ns())
res = {}
for rep in range(args.repeats):
    cold_1 = [make(3000, rng)]
    res.setdefault("cold1", []).append(cell(f"cold1.r{rep}", cold_1, 16))
    cold_4 = [make(3000, rng) for _ in range(4)]
    res.setdefault("cold4", []).append(cell(f"cold4.r{rep}", cold_4, 16))
    res.setdefault("warm4", []).append(cell(f"warm4.r{rep}", [p + " Now answer again." for p in cold_4], 16))
    dec_1 = [make(600, rng)]
    res.setdefault("dec1", []).append(cell(f"dec1.r{rep}", dec_1, 400))
    dec_4 = [make(600, rng) for _ in range(4)]
    res.setdefault("dec4", []).append(cell(f"dec4.r{rep}", dec_4, 400))

med = lambda k, f: statistics.median([r[f] for r in res[k]])
print(f"SUMMARY {args.label} "
      f"cold1_ttft={med('cold1','ttft_max'):.0f} cold4_ttft={med('cold4','ttft_max'):.0f} "
      f"cold_ratio={med('cold4','ttft_max')/max(med('cold1','ttft_max'),1e-9):.2f} "
      f"warm4_ttft={med('warm4','ttft_max'):.0f} "
      f"dec1_agg={med('dec1','agg_decode'):.2f} dec4_agg={med('dec4','agg_decode'):.2f} "
      f"dec_scale={med('dec4','agg_decode')/max(med('dec1','agg_decode'),1e-9):.2f}x", flush=True)

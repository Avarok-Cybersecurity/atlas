#!/usr/bin/env python3
"""Measure the per-forward prefill floor: the fixed cost of one prefill dispatch.

This is the input to the "is slicing worth it" decision. If prefill cost is
`floor + tokens/throughput`, then slicing a chunk into N pieces pays the floor N
times. A small floor means slicing is nearly free and can bound decode stalls; a
large one (Holo's fused step reportedly sat on a ~250ms floor) means small slices
buy overhead instead of latency.

Method: single stream, minimal generation so decode is negligible, and a UNIQUE
random prefix per request so the prefix cache cannot skip the work. Sweep prompt
length below --max-prefill-tokens so each request is exactly one chunk, then fit
a line through (tokens, TTFT).

Reports the intercept (floor), slope (tok/s), and — the number that actually
decides it — the floor's share of a full chunk.
"""
import os, json, random, statistics as st, string, sys, time, urllib.request

URL = os.environ.get("ATLAS_URL", "http://localhost:8888/v1/chat/completions")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "laguna-s-2.1"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
# ~1.4 tokens per word for this filler; sizes chosen to stay in one chunk.
SIZES = [256, 512, 1024, 2048, 4096, 6144, 8000]
WORD = "cache block token prefill decode kernel arena stream radix scheduler".split()


def prompt_of(target_tokens, tag):
    # Unique head defeats the prefix cache; body sized to hit the token target.
    head = f"[{tag}-{''.join(random.choices(string.ascii_lowercase, k=12))}] "
    body = " ".join(random.choice(WORD) for _ in range(int(target_tokens * 0.78)))
    return head + body + "\nReply with exactly: K"


def one(target, tag):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt_of(target, tag)}],
            "max_tokens": 250, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    wall = time.time() - t0
    u = d.get("usage", {})
    return u.get("prompt_tokens"), u.get("completion_tokens"), wall


if __name__ == "__main__":
    print(f"{'target':>8}{'prompt_tok':>12}{'gen':>6}{'wall s':>9}  (median of %d)" % REPEATS)
    pts = []
    for size in SIZES:
        walls, ptoks = [], []
        for r in range(REPEATS):
            ptok, gen, wall = one(size, f"floor{size}r{r}")
            walls.append(wall); ptoks.append(ptok)
            time.sleep(0.4)
        w, p = st.median(walls), st.median(ptoks)
        pts.append((p, w))
        print(f"{size:>8}{p:>12}{gen:>6}{w:>9.3f}")

    # Least-squares fit wall = floor + tokens/throughput.
    n = len(pts)
    sx = sum(p for p, _ in pts); sy = sum(w for _, w in pts)
    sxx = sum(p * p for p, _ in pts); sxy = sum(p * w for p, w in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    floor = (sy - slope * sx) / n
    print(f"\n  per-forward floor : {floor * 1000:7.1f} ms")
    print(f"  throughput        : {1.0 / slope:7.0f} tok/s" if slope > 0 else "  throughput: n/a")
    for chunk in (1024, 2048, 4096, 8192):
        cost = floor + slope * chunk
        print(f"  chunk {chunk:>5}: {cost*1000:7.1f} ms total, floor is {floor/cost*100:4.1f}% of it")
    print("\n  Slicing a chunk into N pieces pays the floor N times; if the floor is a")
    print("  large share at the slice size you want, raise the slice floor instead.")

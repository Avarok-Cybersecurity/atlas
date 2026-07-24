import json, random, time, urllib.request

URL = "http://localhost:8081/v1/chat/completions"
MODEL = "nvidia/Qwen3.6-27B-NVFP4"

# Deterministic filler. Each prompt gets a UNIQUE leading marker so the radix
# prefix cache cannot share a single block -> every measurement is a true cold
# prefill. Lengths are in target tokens; actual counts come from usage.
POOL = ("system module cache latency kernel tensor buffer schedule pipeline "
        "gradient matrix vector throughput register occupancy warp lane stride "
        "prefetch dispatch retire commit branch predictor allocator arena page "
        "descriptor fragment shader compute queue fence semaphore barrier atomic "
        "coherent snapshot replay checkpoint rollback session transcript token").split()

def make_prompt(target_tokens, marker):
    rng = random.Random(hash(marker) & 0xffffffff)
    # ~0.75 tok/word for this vocabulary; overshoot slightly, trimmed by feel.
    nwords = int(target_tokens * 0.78)
    body = " ".join(rng.choice(POOL) for _ in range(nwords))
    return (f"[trace-id {marker}] Below is a log excerpt. Read it and then answer.\n\n"
            f"{body}\n\nIn one short sentence, what is the trace id above?")

def ttft(prompt, max_tok=8):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tok, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); tfirst = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                c = json.loads(d).get("choices", [{}])[0].get("delta", {}).get("content")
            except Exception:
                continue
            if c and tfirst is None:
                tfirst = time.time() - t0
                break
    return (tfirst if tfirst is not None else time.time() - t0) * 1000.0

def prompt_tokens(prompt):
    # Warm follow-up (same prompt, so it hits cache) purely to read the real count.
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read()).get("usage", {}).get("prompt_tokens", -1)

TARGETS = [256, 1024, 2048, 4096, 8192]
DRAWS = 3
rows = []
for t in TARGETS:
    samples = []
    ntok = -1
    for d in range(DRAWS):
        p = make_prompt(t, f"{t}-{d}-a7f2")
        ms = ttft(p)
        if ntok < 0:
            ntok = prompt_tokens(p)
        samples.append(ms)
        print("  target=%5d draw=%d ttft=%8.1fms" % (t, d, ms), flush=True)
    samples.sort()
    med = samples[len(samples) // 2]
    rows.append((t, ntok, med, samples[0], samples[-1]))
    print("target=%5d  ntok=%5d  median=%8.1fms  min=%8.1f max=%8.1f" %
          (t, ntok, med, samples[0], samples[-1]), flush=True)

print("\n== COLD TTFT vs PROMPT LENGTH ==")
print("%8s %8s %11s %11s %10s" % ("target", "ntok", "median_ms", "tok/s", "ms/1k_tok"))
for t, n, med, lo, hi in rows:
    tps = (n / (med / 1000.0)) if n > 0 and med > 0 else 0
    print("%8d %8d %11.1f %11.1f %10.1f" % (t, n, med, tps, med / max(n, 1) * 1000))

# Slope fit: fixed overhead (intercept) vs per-token prefill cost (slope).
pts = [(n, med) for (t, n, med, lo, hi) in rows if n > 0]
if len(pts) >= 2:
    mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    slope = num / den; inter = my - slope * mx
    print("\nfit: ttft_ms = %.4f * ntok + %.1f" % (slope, inter))
    print("  -> fixed overhead  = %.0f ms" % inter)
    print("  -> prefill rate    = %.1f tok/s (1/slope)" % (1000.0 / slope))

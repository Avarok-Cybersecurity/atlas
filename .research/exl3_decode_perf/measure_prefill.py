#!/usr/bin/env python3
"""Cold prefill throughput at a target prompt length: unique (salted) prompt per request so the
prefix cache cannot serve it, max_tokens=1, prefill tok/s = server-attested prompt_tokens / wall
to first (and only) token. Prints a fingerprint line."""
import json, sys, time, urllib.request, random, argparse, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8888)
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--tokens", type=int, nargs="+", default=[8000, 11000])
ap.add_argument("--repeats", type=int, default=2)
args = ap.parse_args()

WORDS = ("kernel stream tensor cache block schedule verify draft router expert layer norm "
         "gate highway carry snapshot commit rewind window hash gather slot arena page token "
         "prefix budget latency bandwidth roofline launch cooperative barrier reduction").split()

def make_prompt(n_tokens, salt):
    rng = random.Random(salt)
    # ~1.3 tokens/word for this vocabulary; the server's usage.prompt_tokens is what we report
    words = [rng.choice(WORDS) for _ in range(int(n_tokens / 1.3))]
    body = " ".join(words)
    return f"Session salt {salt}. Summarize the following log in one sentence.\n\n{body}\n\nOne sentence:"

def one(n_tokens, salt):
    prompt = make_prompt(n_tokens, salt)
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps({"model": args.model, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 1, "temperature": 0.0,
                         "chat_template_kwargs": {"reasoning_effort": "low"}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    pt = d["usage"]["prompt_tokens"]
    return pt, wall, pt / wall

print(f"FINGERPRINT port={args.port} model={args.model} max_tokens=1 temp=0 salted-unique-prompts "
      f"date={time.strftime('%Y-%m-%dT%H:%M:%S')}")
for n in args.tokens:
    res = []
    for i in range(args.repeats):
        pt, wall, tps = one(n, salt=int(time.time() * 1000) % 100000 + i)
        print(f"target~{n}: prompt_tokens={pt} wall={wall:.2f}s prefill={tps:.0f} tok/s")
        res.append((pt, tps))
    print(f"SUMMARY target~{n}: prompt_tokens~{res[0][0]} prefill_tok_s_median={statistics.median(t for _, t in res):.0f}")

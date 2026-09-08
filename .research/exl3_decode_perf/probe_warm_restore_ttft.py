#!/usr/bin/env python3
"""TTFT probe for the prefix-cache warm-restore path (the operator's TUI sweep
showed the 128-token, 97%-cached prompt with the WORST TTFT at every C: 729 ms
at C=1 vs 209 ms for a 512-token, 100%-cached prompt).

For each prompt length L in --tokens, against a running server:
  cold    : unique salted prompt of ~L tokens                 (no cache hit)
  full    : the same prompt again                              (100% cached)
  tail    : the same prompt with the last ~3% of words changed (~97% cached)
  cold2   : a second unique prompt of ~L tokens                (cold again; drift control)
Each request: max_tokens=1, temperature 0, non-streaming; records the server's
time_to_first_token_ms, cached_tokens, prompt_tokens and the client wall.
Repeats the whole cycle --repeats times with fresh salts. Prints a table + a
FINGERPRINT line. Read TTFT for the hypothesis "restore + tiny prefill is
slower than a short cold prefill"; anything else here is a number, not a claim.
"""
import argparse, json, random, time, urllib.request, sys, datetime

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8899)
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048])
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--label", default="")
args = ap.parse_args()

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
RATIO = 1.037  # tokens per word measured on this tokenizer for this word class


def prompt(n_tokens, salt, rng):
    n_words = max(4, int(n_tokens / RATIO) - 12)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return f"salt-{salt}. Repeat the last word of this list:\n{body}"


def tail_variant(p, rng):
    head, body = p.split("\n", 1)
    words = body.split(" ")
    k = max(1, len(words) * 3 // 100)
    for i in range(len(words) - k, len(words)):
        words[i] = rng.choice(WORDS) + "x"
    return head + "\n" + " ".join(words)


def req(p):
    d = {"model": args.model, "temperature": 0, "max_tokens": 1, "stream": False,
         "messages": [{"role": "user", "content": p}]}
    t0 = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions", data=json.dumps(d).encode(),
        headers={"content-type": "application/json"}), timeout=600)
    j = json.load(r)
    wall = (time.perf_counter() - t0) * 1000
    u = j.get("usage", {})
    return {"wall_ms": wall, "ttft_ms": u.get("time_to_first_token_ms"),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


print(f"FINGERPRINT probe_warm_restore_ttft label={args.label} port={args.port} model={args.model} tokens={args.tokens} repeats={args.repeats} date={datetime.datetime.utcnow().isoformat()}Z", flush=True)
rows = []
for rep in range(args.repeats):
    for L in args.tokens:
        rng = random.Random(time.time_ns())
        salt = rng.randrange(10**9)
        p = prompt(L, salt, rng)
        seq = [("cold", p), ("full", p), ("tail", tail_variant(p, rng)), ("cold2", prompt(L, rng.randrange(10**9), rng))]
        for name, pp in seq:
            r = req(pp)
            rows.append((rep, L, name, r))
            print(f"rep={rep} L={L:5d} {name:5s} prompt_tokens={r['prompt_tokens']} cached={r['cached']} ttft_ms={r['ttft_ms']:.0f} wall_ms={r['wall_ms']:.0f}", flush=True)
            time.sleep(0.5)

print("SUMMARY (median ttft_ms over repeats)")
import statistics
for L in args.tokens:
    line = f"  L={L:5d}"
    for name in ("cold", "full", "tail", "cold2"):
        vals = [r["ttft_ms"] for rep, l, n, r in rows if l == L and n == name and r["ttft_ms"] is not None]
        line += f"  {name}={statistics.median(vals):.0f}" if vals else f"  {name}=n/a"
    print(line)

#!/usr/bin/env python3
"""Where does prefix reuse stop paying when the divergence moves back from the
prompt tail?

On a hybrid-SSM model the KV radix can match any block prefix, but the pass can
only be SKIPPED from an SSM snapshot at or below the divergence. Prefill
snapshots fire only at CHUNK ends plus the last two block boundaries below the
prompt end (`prefill_b/save_checkpoint.rs`: the checkpoint interval is a FILTER
over chunk boundaries, never a generator), so with an 8K chunk a prompt shorter
than the chunk has anchors ONLY in its final ~2 blocks.

Prediction under that reading: editing the last ~1-2% of a prompt reuses; editing
earlier costs a full cold prefill even though `cached_tokens` still reports a
large match. This probe measures TTFT against the divergence position.

Per prompt length L: prime with a cold request, then for each fraction f in
--divergence (share of the prompt kept identical) send a variant whose tail after
f*L is regenerated, and record server TTFT + cached_tokens. A final exact repeat
bounds the best case. n=--repeats fresh salts.
"""
import argparse, json, random, time, urllib.request, statistics, datetime

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8899)
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--tokens", type=int, nargs="+", default=[2048])
ap.add_argument("--divergence", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--label", default="")
args = ap.parse_args()

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
RATIO = 1.037


def make_words(n, rng):
    return [rng.choice(WORDS) for _ in range(n)]


def render(head, words):
    return f"{head}\n" + " ".join(words)


def req(p):
    d = {"model": args.model, "temperature": 0, "max_tokens": 1, "stream": False,
         "messages": [{"role": "user", "content": p}]}
    t0 = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions", data=json.dumps(d).encode(),
        headers={"content-type": "application/json"}), timeout=900)
    j = json.load(r)
    u = j.get("usage", {})
    return {"wall_ms": (time.perf_counter() - t0) * 1000, "ttft_ms": u.get("time_to_first_token_ms"),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


print(f"FINGERPRINT probe_partial_hit_cliff label={args.label} port={args.port} tokens={args.tokens} "
      f"divergence={args.divergence} repeats={args.repeats} date={datetime.datetime.utcnow().isoformat()}Z", flush=True)
rows = []
for rep in range(args.repeats):
    for L in args.tokens:
        rng = random.Random(time.time_ns())
        n_words = max(16, int(L / RATIO) - 12)
        head = f"salt-{rng.randrange(10**9)}. Repeat the last word of this list:"
        base = make_words(n_words, rng)
        cold = req(render(head, base))
        rows.append((rep, L, "cold", 0.0, cold))
        print(f"rep={rep} L={L} cold           prompt={cold['prompt_tokens']} cached={cold['cached']} ttft_ms={cold['ttft_ms']:.0f}", flush=True)
        for f in args.divergence:
            keep = int(n_words * f)
            variant = base[:keep] + make_words(n_words - keep, rng)
            r = req(render(head, variant))
            rows.append((rep, L, "div", f, r))
            print(f"rep={rep} L={L} keep={f:<5} prompt={r['prompt_tokens']} cached={r['cached']} ttft_ms={r['ttft_ms']:.0f}", flush=True)
            time.sleep(0.3)
        again = req(render(head, base))
        rows.append((rep, L, "exact", 1.0, again))
        print(f"rep={rep} L={L} exact repeat   prompt={again['prompt_tokens']} cached={again['cached']} ttft_ms={again['ttft_ms']:.0f}", flush=True)

print("\nSUMMARY median ttft_ms by kept-prefix fraction (cold and exact bound the range)")
for L in args.tokens:
    def med(kind, f=None):
        v = [r["ttft_ms"] for rep, l, k, ff, r in rows if l == L and k == kind and (f is None or abs(ff - f) < 1e-9) and r["ttft_ms"] is not None]
        return statistics.median(v) if v else None
    print(f"  L={L}: cold={med('cold'):.0f}")
    for f in args.divergence:
        m = med("div", f)
        print(f"    keep {f:>5}: ttft={m:.0f}" if m is not None else f"    keep {f:>5}: n/a")
    print(f"    exact    : ttft={med('exact'):.0f}")

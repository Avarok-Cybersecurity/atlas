#!/usr/bin/env python3
"""Prefill throughput across a concurrency x input-length grid.

`prefill_floor.py` fits a floor+throughput model on a SINGLE stream. That says
nothing about whether prefill *batches* — the question concurrency raises. This
sweeps C concurrent requests at a fixed input length and reports aggregate
prefill tok/s, so a flat row means prefill is serialising and a rising one means
it is genuinely co-dispatching.

Prefill time is taken from Atlas's own `usage.time_to_first_token_ms`, so decode
never contaminates the measurement. Aggregate throughput uses the SLOWEST TTFT
in the batch (all C prompts are in flight over that window, so C*ISL tokens
land in max-TTFT seconds).

Every request gets a unique random prefix: with prefix caching on, a repeated
prompt would be skipped entirely and the run would measure cache hits.

Usage:
  python3 prefill_matrix.py <model> [--conc 1,2,4,8] [--isl 1024,2048,4096,8192,16384]
"""
import argparse, json, os, random, statistics as st, string, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("ATLAS_URL", "http://localhost:8888/v1/chat/completions")


def make_prompt(target_tokens, rng):
    """~target_tokens of unique filler. ~0.75 words/token is close enough for
    these shapes; the exact count is read back from usage.prompt_tokens."""
    uniq = "".join(rng.choice(string.ascii_lowercase) for _ in range(24))
    words = int(target_tokens * 0.75)
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return f"Reference {uniq}. {body}\n\nReply with the single word: ok"


WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "server kernel tensor buffer stream socket packet vector matrix cache "
         "compile deploy measure profile allocate schedule dispatch resolve").split()


def one(model, prompt, timeout):
    req = urllib.request.Request(
        URL,
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # >=250 so the request is a realistic generation, not a degenerate
            # 1-token probe; TTFT is what we read, so decode length is inert.
            "max_tokens": 250,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    u = d.get("usage", {})
    return {
        "ttft_s": (u.get("time_to_first_token_ms") or 0) / 1000.0,
        "prompt_tokens": u.get("prompt_tokens", 0),
        "wall_s": time.time() - t0,
    }


def cell(model, conc, isl, timeout, rng):
    prompts = [make_prompt(isl, rng) for _ in range(conc)]
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(one, model, p, timeout) for p in prompts]
        rows = []
        for f in futs:
            try:
                rows.append(f.result())
            except Exception as e:
                rows.append({"err": str(e)[:60]})
    ok = [r for r in rows if "err" not in r and r["ttft_s"] > 0]
    if not ok:
        return None
    toks = sum(r["prompt_tokens"] for r in ok)
    span = max(r["ttft_s"] for r in ok)
    return {
        "conc": conc, "isl": isl, "n_ok": len(ok), "prompt_tokens_total": toks,
        "ttft_median_s": st.median([r["ttft_s"] for r in ok]),
        "ttft_max_s": span,
        "prefill_tok_s": toks / span if span > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--isl", default="1024,2048,4096,8192,16384")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    concs = [int(x) for x in a.conc.split(",")]
    isls = [int(x) for x in a.isl.split(",")]
    rng = random.Random(1234)

    out = []
    print(f"  prefill tok/s (aggregate; {a.model})")
    print("  ISL \\ C  " + "".join(f"{c:>12}" for c in concs))
    for isl in isls:
        row = []
        for c in concs:
            r = cell(a.model, c, isl, a.timeout, rng)
            out.append(r)
            row.append(f"{r['prefill_tok_s']:>12,.0f}" if r else f"{'ERR':>12}")
        print(f"  {isl:>6}   " + "".join(row))

    print("\n  median TTFT seconds")
    print("  ISL \\ C  " + "".join(f"{c:>12}" for c in concs))
    for isl in isls:
        row = []
        for c in concs:
            r = next((x for x in out if x and x["conc"] == c and x["isl"] == isl), None)
            row.append(f"{r['ttft_median_s']:>12.2f}" if r else f"{'ERR':>12}")
        print(f"  {isl:>6}   " + "".join(row))

    print("\n  A FLAT row = prefill is serialising across streams.")
    print("  A rising row = prefill genuinely co-dispatches.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump([x for x in out if x], f, indent=1)
        print(f"\n  wrote {a.json}")


if __name__ == "__main__":
    main()

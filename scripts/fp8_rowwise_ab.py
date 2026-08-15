#!/usr/bin/env python3
"""A/B harness for the FP8 weight-precision levers: KL drift + throughput.

Two modes, deliberately SEQUENTIAL rather than two live serves:

    collect <port> <tag> <out.json>   drive one serve, record everything
    compare <control.json> <cand.json>

`kl_coherence_gate.py` takes two ports at once, which is right on a discrete
GPU and wrong here: a 27B pair on one unified 121 GB pool is the co-tenancy
that has taken this box's SSH down before, and co-tenancy corrupts the
throughput half of the measurement long before it OOMs. So each leg is
measured alone and the comparison happens offline, the same split
`prefill_w4a4_ab.py` / `prefill_w4a4_compare.py` use.

WHAT IS BEING MEASURED, and why these three things:

  * KL drift, mean and p99, over the top-20 logprob support at each matched
    position. This is NOT a pass/fail here. A weight-precision change is not
    output-neutral by construction, so "KL is nonzero" is the expected result,
    not a failure -- the number says HOW FAR the distribution moved, which is
    what makes a fold decision evidence instead of a vibe.
  * Token match and first-divergence position, for the same reason.
  * Throughput, prefill and decode separately. The whole trade of keeping a
    weight at higher precision is memory and speed against quality, and a
    quality win nobody priced is not a result.

The KL function is the corrected one from `kl_coherence_gate.py` -- BOTH sides
renormalised over the shared support. Normalising only P adds a constant
-log(sum p) to every position, which made an identical config score ~0.061
instead of 0 and put that gate's own threshold out of reach. Verified there:
KL(p, p) == 0.0 exactly.

Usage:
    python3 scripts/fp8_rowwise_ab.py collect 8897 baseline /tmp/base.json
    python3 scripts/fp8_rowwise_ab.py compare /tmp/base.json /tmp/cand.json
"""
import json
import math
import statistics
import sys
import time
import urllib.request

# Greedy, seeded, and short: every assertion here is about the DISTRIBUTION at
# each position, so sampling variance would be noise added to the thing being
# measured.
GEN_TOKENS = 96
TOP_K = 20

# Mixed on purpose. Weight-precision damage does not show up uniformly: prose
# tolerates a blurred argmax, structured output and arithmetic do not, and the
# long-context item is where an accumulated drift becomes visible.
PROMPTS = [
    "In exactly three sentences, explain what a KV cache stores during autoregressive decoding.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "List three reasons a speculative draft token gets rejected during verification.",
    "Compute 17 * 23 and then subtract 91. Show each step on its own line.",
    "Return a JSON object with keys name, age, city for a person named Ada, 36, London. JSON only.",
    "Name the first six prime numbers, separated by commas, and nothing else.",
]

# One long prompt drives the prefill number. Cold each run (the tag is
# interpolated) so a prefix-cache hit cannot be mistaken for prefill speed.
LONG_BODY = (
    "A paged KV cache stores per-layer key and value tensors in fixed-size blocks so that "
    "sequences of different lengths share one allocator without fragmentation. Each block "
    "holds a fixed number of token slots, and a per-sequence block table maps logical "
    "positions onto physical blocks. Prefill fills these blocks in chunks; decode appends "
    "one token per step and occasionally allocates a fresh block. "
)


def post(port, body, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(port, prompt, max_tokens=GEN_TOKENS, logprobs=True):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = TOP_K
    t0 = time.time()
    resp = post(port, body)
    wall = time.time() - t0
    return resp, wall


def positions(resp):
    """[{tok, top:{token: logprob}}] per generated position, [] if unsupported."""
    try:
        content = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, TypeError):
        return []
    return [
        {"tok": c["token"], "top": {t["token"]: t["logprob"] for t in c.get("top_logprobs", [])}}
        for c in content
    ]


def collect(port, tag, out_path):
    runs = []
    for qi, p in enumerate(PROMPTS):
        resp, wall = ask(port, p)
        u = resp.get("usage", {})
        runs.append({
            "qi": qi,
            "prompt": p,
            "wall_s": wall,
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "text": resp["choices"][0]["message"].get("content") or "",
            "positions": positions(resp),
        })
        print(f"  [{tag}] q{qi} {u.get('completion_tokens')} tok in {wall:.2f}s", flush=True)

    # Prefill: one long cold prompt, 8 generated tokens so decode is negligible.
    long_prompt = f"[{tag}-cold-{int(time.time())}] " + LONG_BODY * 24 + "\nSummarise in one sentence."
    resp, wall = ask(port, long_prompt, max_tokens=8, logprobs=False)
    ptok = resp.get("usage", {}).get("prompt_tokens") or 0
    prefill = {"prompt_tokens": ptok, "wall_s": wall, "tok_per_s": ptok / wall if wall else 0.0}
    print(f"  [{tag}] prefill {ptok} tok in {wall:.2f}s = {prefill['tok_per_s']:.0f} tok/s", flush=True)

    # Decode: short prompt, long generation, so the wall is dominated by decode.
    resp, wall = ask(port, "Count from 1 to 200, separated by commas.", max_tokens=400, logprobs=False)
    ctok = resp.get("usage", {}).get("completion_tokens") or 0
    decode = {"completion_tokens": ctok, "wall_s": wall, "tok_per_s": ctok / wall if wall else 0.0}
    print(f"  [{tag}] decode {ctok} tok in {wall:.2f}s = {decode['tok_per_s']:.1f} tok/s", flush=True)

    with open(out_path, "w") as fh:
        json.dump({"tag": tag, "runs": runs, "prefill": prefill, "decode": decode}, fh)
    print(f"wrote {out_path}")


def kl(p_lp, q_lp):
    """KL(P||Q) over the union of top tokens; both sides renormalised.

    Lifted from kl_coherence_gate.py, including its 2026-07-25 fix: the top-k
    logprobs carry only ~94% of the mass, so normalising one side and not the
    other scores identical inputs at ~0.061 instead of 0.
    """
    toks = set(p_lp) | set(q_lp)
    floor = -30.0
    ps = {t: math.exp(p_lp.get(t, floor)) for t in toks}
    qs = {t: math.exp(q_lp.get(t, floor)) for t in toks}
    zp = sum(ps.values()) or 1.0
    zq = sum(qs.values()) or 1.0
    total = 0.0
    for t in toks:
        pv, qv = ps[t] / zp, qs[t] / zq
        if pv > 0:
            total += pv * (math.log(pv) - math.log(qv if qv > 0 else math.exp(floor)))
    return max(total, 0.0)


def compare(ctl_path, cnd_path):
    with open(ctl_path) as fh:
        ctl = json.load(fh)
    with open(cnd_path) as fh:
        cnd = json.load(fh)

    print(f"=== {ctl['tag']}  vs  {cnd['tag']} ===\n")
    print("throughput")
    for phase in ("prefill", "decode"):
        a, b = ctl[phase]["tok_per_s"], cnd[phase]["tok_per_s"]
        delta = (b / a - 1.0) * 100 if a else 0.0
        print(f"  {phase:8} {a:9.1f} -> {b:9.1f} tok/s   {delta:+6.1f}%")

    ci = {r["qi"]: r for r in ctl["runs"]}
    fi = {r["qi"]: r for r in cnd["runs"]}
    all_kl, matches, firsts = [], [], []
    print("\nper-prompt drift  (KL over the SHARED PREFIX only — see note below)")
    print(f"  {'qi':>3} {'match':>7} {'meanKL':>9} {'maxKL':>9} {'1st diff':>9} {'nKL':>5}")
    for qi in sorted(set(ci) & set(fi)):
        a, b = ci[qi]["positions"], fi[qi]["positions"]
        n = min(len(a), len(b))
        if n == 0:
            continue
        m = sum(1 for i in range(n) if a[i]["tok"] == b[i]["tok"]) / n
        first = next((i for i in range(n) if a[i]["tok"] != b[i]["tok"]), -1)
        # ★ KL ONLY UP TO THE FIRST DIVERGENCE.
        #
        # These are FREE-RUNNING generations: each side conditions on its own
        # previous token. The moment one token differs the two models are
        # continuing DIFFERENT sentences, and a distribution comparison across
        # different contexts measures nothing about precision — it measures
        # that the sentences differ, which the token-match column already says.
        # Scoring past that point produced mean KL ~5 nats and p99 pinned at
        # the -30 floor, numbers that look alarming and mean nothing.
        shared = n if first < 0 else first
        ks = [kl(a[i]["top"], b[i]["top"]) for i in range(shared)]
        all_kl.extend(ks)
        matches.append(m)
        firsts.append(first if first >= 0 else n)
        mk = statistics.mean(ks) if ks else float("nan")
        xk = max(ks) if ks else float("nan")
        print(f"  {qi:>3} {m:>6.1%} {mk:>9.5f} {xk:>9.5f} "
              f"{'none' if first < 0 else first:>9} {len(ks):>5}")

    if not all_kl:
        print("\nNO LOGPROBS RETURNED — the endpoint did not honour top_logprobs; "
              "drift is UNMEASURED, which is not the same as zero.")
        return
    s = sorted(all_kl)
    p99 = s[min(len(s) - 1, int(0.99 * (len(s) - 1)))]
    print(f"\nOVERALL  token match {statistics.mean(matches):.1%}   "
          f"mean KL {statistics.mean(all_kl):.5f}   p99 KL {p99:.5f}   max KL {s[-1]:.5f}")
    print(f"median first divergence: {statistics.median(firsts):.1f} of {GEN_TOKENS} generated")
    print("\nA weight-precision change is NOT output-neutral: nonzero drift is the")
    print("expected result and is not a failure. These numbers size the change so")
    print("the fold decision is made on measured drift; pair them with a quality")
    print("run (the vision/video gates, or BFCL) before concluding which side is better.")
    print("\nLIMIT OF THIS MEASURE: KL is scored only on the shared prefix, because")
    print("past the first divergence the two sides are continuing different")
    print("sentences. For a change expected to diverge, the sharper instrument is")
    print("TEACHER-FORCED logprobs over one fixed token sequence (prompt_logprobs),")
    print("where both sides see identical context at every position and KL stays")
    print("meaningful for the whole window. That is the upgrade this harness wants.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

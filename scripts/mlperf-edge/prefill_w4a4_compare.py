#!/usr/bin/env python3
"""Compare the two ATLAS_FP4_PREFILL legs: prefill speed vs accuracy drift.

W4A4 prefill is lossy by construction, so a differing output hash is expected and
is NOT the gate. The gates are:

  * token match  -- fraction of positions where both legs emit the same token,
                    measured only over the common prefix length per prompt.
  * KL drift     -- KL(control || candidate) over the top-20 logprob support at
                    each matched position, in nats. Reported as mean and p99.
  * coherence    -- Gate C2 text/tool-call, checked separately by the runner.

Speed is reported per prompt-length bucket, since the whole thesis is that FP4
activations only pay once M = seqlen is large enough to fill the tiles.

Usage: prefill_w4a4_compare.py <ctl.json> <fp4.json>
"""
import json
import math
import statistics
import sys


def kl(p_top, q_top):
    keys = set(p_top) | set(q_top)
    floor = -30.0
    total = 0.0
    for k in keys:
        lp = p_top.get(k, floor)
        lq = q_top.get(k, floor)
        total += math.exp(lp) * (lp - lq)
    return max(total, 0.0)


with open(sys.argv[1]) as fh:
    ctl = json.load(fh)
with open(sys.argv[2]) as fh:
    cnd = json.load(fh)

print("=== prefill speed (cold, median wall per prompt-size bucket) ===")
print(f"{'repeats':>8} {'ctl ms':>10} {'fp4 ms':>10} {'delta':>10} {'speedup':>9}")
for rep in ctl["by_rep"]:
    a = ctl["by_rep"][rep]["median_wall_ms"]
    b = cnd["by_rep"].get(rep, {}).get("median_wall_ms")
    if b is None:
        continue
    print(f"{rep:>8} {a:>10.1f} {b:>10.1f} {b - a:>+10.1f} {a / b:>8.3f}x")

# Pair runs by (rep, question index) so we compare like with like.
ci = {(r["rep"], r["qi"]): r for r in ctl["runs"]}
fi = {(r["rep"], r["qi"]): r for r in cnd["runs"]}

print("\n=== accuracy drift (control || fp4) ===")
print(f"{'repeats':>8} {'ptok':>7} {'tok match':>11} {'mean KL':>10} {'p99 KL':>10} {'1st diff':>9}")
all_kl, all_match, all_first = [], [], []
for key in sorted(set(ci) & set(fi)):
    a, b = ci[key], fi[key]
    n = min(len(a["tokens"]), len(b["tokens"]))
    if n == 0:
        continue
    match = sum(1 for i in range(n) if a["tokens"][i] == b["tokens"][i]) / n
    kls = [kl(a["tops"][i], b["tops"][i]) for i in range(n)
           if i < len(a["tops"]) and i < len(b["tops"])]
    first = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), -1)
    all_kl.extend(kls)
    all_match.append(match)
    all_first.append(first if first >= 0 else n)
    mk = statistics.mean(kls) if kls else 0.0
    pk = max(kls) if kls else 0.0
    print(f"{key[0]:>8} {a['prompt_tokens'] or 0:>7} {match:>10.1%} {mk:>10.5f} {pk:>10.5f} "
          f"{'none' if first < 0 else first:>9}")

if all_kl:
    all_kl.sort()
    p99 = all_kl[min(len(all_kl) - 1, int(0.99 * (len(all_kl) - 1)))]
    print(f"\nOVERALL  token match {statistics.mean(all_match):.1%}  "
          f"mean KL {statistics.mean(all_kl):.5f}  p99 KL {p99:.5f}  max KL {all_kl[-1]:.5f}")
    print(f"median first-divergence position: {statistics.median(all_first):.1f} of ~48 generated")
    # Same thresholds the fold gate uses; W4A4 is lossy so these are expected to
    # be the deciding numbers rather than a formality.
    print(f"\nGATE  mean KL < 1e-3 : {'PASS' if statistics.mean(all_kl) < 1e-3 else 'FAIL'}"
          f"   token match >= 0.99 : {'PASS' if statistics.mean(all_match) >= 0.99 else 'FAIL'}")

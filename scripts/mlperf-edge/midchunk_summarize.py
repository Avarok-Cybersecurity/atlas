#!/usr/bin/env python3
"""Summarize the midchunk gate: warm TTFT across reps, plus the trajectory caveat.

Reports the MONOTONIC probe only. The first-round replay probe is excluded on
purpose: it alternates back to the base prompt between reps, and there is one
tail slot per session, so it thrashes exactly the structure under test.

Two things are reported separately and must not be conflated:
  * warm TTFT  -- what midchunk is supposed to move (tail-checkpoint retention).
  * emitted token counts -- if the legs diverge, TPOT is trajectory-confounded
    and any TPOT delta is NOT quotable as a speed result.

Usage: midchunk_summarize.py <outdir>
"""
import glob
import json
import os
import statistics
import sys

OUT = sys.argv[1]


def load(leg):
    reps = []
    for p in sorted(glob.glob(os.path.join(OUT, f"{leg}.tpot.r*.json"))):
        try:
            with open(p) as fh:
                reps.append(json.load(fh))
        except Exception:
            pass
    return reps


print("=== midchunk gate: monotonic warm-conversation probe ===\n")
agg = {}
for leg in ("mc_off", "mc_on"):
    reps = load(leg)
    if not reps:
        print(f"{leg}: no reps found")
        continue
    warm_ttft, warm_tpot, ntok = [], [], []
    for d in reps:
        for r in d.get("runs", []):
            if r.get("warm"):
                warm_ttft.append(r["ttft"])
                warm_tpot.append(r["tpot"])
                ntok.append(r.get("ntok", 0))
    agg[leg] = {
        "reps": len(reps), "n_warm": len(warm_ttft),
        "ttft_mean": statistics.mean(warm_ttft), "ttft_med": statistics.median(warm_ttft),
        "ttft_min": min(warm_ttft), "ttft_max": max(warm_ttft),
        "ttft_sd": statistics.pstdev(warm_ttft),
        "tpot_med": statistics.median(warm_tpot), "tok": sum(ntok),
    }
    a = agg[leg]
    print(f"{leg}: reps={a['reps']} warm_turns={a['n_warm']}")
    print(f"   warm TTFT  mean={a['ttft_mean']:7.1f}  med={a['ttft_med']:7.1f}  "
          f"sd={a['ttft_sd']:6.1f}  range=[{a['ttft_min']:.0f}, {a['ttft_max']:.0f}] ms")
    print(f"   warm TPOT  med={a['tpot_med']:6.2f} ms   tokens emitted={a['tok']}")

if len(agg) == 2:
    o, n = agg["mc_off"], agg["mc_on"]
    d_mean = n["ttft_mean"] - o["ttft_mean"]
    print(f"\nwarm TTFT mean: {o['ttft_mean']:.1f} -> {n['ttft_mean']:.1f} ms "
          f"({d_mean:+.1f} ms, {100*d_mean/o['ttft_mean']:+.1f}%)")
    print(f"warm TTFT sd:   {o['ttft_sd']:.1f} -> {n['ttft_sd']:.1f} ms "
          f"(stability; midchunk should REDUCE this if it is retaining the tail)")
    tok_div = abs(n["tok"] - o["tok"]) / max(1, o["tok"])
    print(f"\ntokens emitted: {o['tok']} vs {n['tok']}  (divergence {100*tok_div:.1f}%)")
    if tok_div > 0.02:
        print("  -> trajectories DIVERGED: TPOT is confounded and must NOT be quoted as a")
        print("     speed result. Only the TTFT comparison is usable here.")
    else:
        print("  -> trajectories match closely: TPOT comparison is usable.")
    print("\nVERDICT REQUIRES the BFCL subset above to hold. midchunk's known failure")
    print("mode is cross-request output corruption, which no latency number can see.")

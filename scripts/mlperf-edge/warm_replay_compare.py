#!/usr/bin/env python3
"""Verdict for the ATLAS_GDN_REGRESIDENT A/B.

Two gates, both must hold before this is a fold candidate:

1. TOKEN EQUALITY. The kernel claims cosine 1.0 / max|dH|~1e-8 vs WY4, i.e. it is
   the same acceptance class, not a lossy tradeoff. So the completions must match
   exactly. A mismatch does NOT automatically mean the kernel is wrong — greedy
   spec-decode output is trajectory-dependent — but it removes the "free win"
   framing and forces a full KL/accuracy gate before anything ships.

2. TTFT WIN THAT SCALES WITH DELTA. Replay cost is proportional to the replayed
   suffix, so a real recurrence speedup must grow with delta. A flat offset
   across all deltas is measuring something else (serve noise, admission cost).
   This is the falsifier that separates "the kernel is faster" from "this run was
   faster", and it is why the probe sweeps delta instead of testing one point.

Usage: warm_replay_compare.py <control.json> <regresident.json>
"""
import json
import statistics
import sys

with open(sys.argv[1]) as fh:
    ctl = json.load(fh)
with open(sys.argv[2]) as fh:
    cnd = json.load(fh)

print("=== ATLAS_GDN_REGRESIDENT A/B (warm Marconi replay path) ===")
print(f"base prompt: {ctl['base_chars']} chars\n")
print(f"{'cell':11s} {'delta':>8s} {'control p50':>12s} {'regres p50':>12s} "
      f"{'speedup':>8s} {'saved':>9s} {'tokens':>8s}")

rows = []
for name in ctl["cells"]:
    if name not in cnd["cells"]:
        continue
    c, r = ctl["cells"][name], cnd["cells"][name]
    same = c["texts"] == r["texts"]
    sp = c["p50"] / r["p50"] if r["p50"] else 0.0
    rows.append((c["delta_chars"], c["p50"], r["p50"], sp))
    print(f"{name:11s} {c['delta_chars']:8d} {c['p50']:12.1f} {r['p50']:12.1f} "
          f"{sp:8.3f} {c['p50']-r['p50']:+8.1f}ms {'MATCH' if same else 'DIFFER':>8s}")

allsame = all(ctl["cells"][n]["texts"] == cnd["cells"][n]["texts"]
              for n in ctl["cells"] if n in cnd["cells"])
print(f"\nGATE 1 token equality across all cells: {'PASS' if allsame else 'FAIL — needs a full KL/accuracy gate'}")

# Does the saving grow with delta? Slope of (control-regres) vs delta_chars.
if len(rows) >= 3:
    xs = [r[0] for r in rows]
    ys = [r[1] - r[2] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    print(f"GATE 2 saving vs delta slope: {slope*1000:+.3f} ms per 1000 delta chars")
    if slope > 0:
        print("       -> saving GROWS with replay length: consistent with a real recurrence speedup")
    else:
        print("       -> saving does NOT grow with replay length: NOT the recurrence; "
              "treat any headline delta as noise/offset")

best = max(rows, key=lambda r: r[1] - r[2]) if rows else None
if best:
    print(f"\nlargest absolute saving: {best[1]-best[2]:+.1f} ms at delta={best[0]} chars "
          f"({best[3]:.2f}x)")
print("\nNOTE: this probe measures the REPLAY path only. Translating to wall requires the "
      "e2e delta distribution (golden run: p50 210 / mean 331 / p90 698 new tokens, "
      "marginal-prefill slice = 771 s of a 4104 s wall).")

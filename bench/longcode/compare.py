#!/usr/bin/env python3
"""Paired before/after diff of two harness summary JSONs.

Because seeds are fixed, seed N in run A and run B are the SAME stochastic
trajectory — so per-seed deltas are a paired comparison (far tighter than
unpaired at N=10). Reports per-seed completeness flips and the aggregate
metric deltas. No gate logic here; harness.py owns the gate.
"""
import json
import sys


def load(p):
    d = json.loads(open(p).read())
    return d, {s["seed"]: s for s in d["per_seed"]}


def main():
    if len(sys.argv) != 3:
        print("usage: compare.py <before.summary.json> <after.summary.json>")
        return 2
    a, am = load(sys.argv[1])
    b, bm = load(sys.argv[2])
    print(f"BEFORE {a['label']}  vs  AFTER {b['label']}")
    print(f"  completeness: {a['completeness_pass_count']}/{a['n']}"
          f"  ->  {b['completeness_pass_count']}/{b['n']}")
    for key in (
        "valid_js_line_count",
        "duplicate_declaration_count",
        "tokens_to_first_degeneration",
    ):
        av = a[key].get("mean")
        bv = b[key].get("mean")
        if av is None or bv is None:
            print(f"  {key}: {av} -> {bv}")
        else:
            print(f"  {key}: {av} -> {bv}  (Δ {round(bv - av, 2):+})")
    print("  per-seed completeness flips:")
    flips = 0
    for seed in sorted(set(am) | set(bm)):
        pa = am.get(seed, {}).get("completeness_pass")
        pb = bm.get(seed, {}).get("completeness_pass")
        if pa != pb:
            flips += 1
            arrow = "FIXED" if pb else "REGRESSED"
            print(f"    seed {seed}: {pa} -> {pb}  [{arrow}]")
    if not flips:
        print("    (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

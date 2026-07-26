#!/usr/bin/env python3
"""Join drafter SHADOW_TOPK lines with verify SHADOW_TGT lines (Phase 0 of the
tree-spec plan) -> per-depth conditional top-k coverage.

Usage: parse_shadow_topk.py <serve_log> [<serve_log>...] [-o out.json]

Log grammar (concurrency-1, chronological, so a streaming last-writer join on
position is sound):
  SHADOW_TOPK pos=<P> ids=[i1, i2, ...] probs=[p1, p2, ...]
  SHADOW_TGT base=<B> v=[v0,v1,...] drafts=[d0,d1,...]
Draft i (drafter position B+i) is judged against v_i. Depth = i+1. Depth-d
stats are conditioned on the spine prefix having been accepted
(drafts[j] == v[j] for all j < i), which is what a tree's hedges see.
"""
import argparse
import json
import re
import sys

TOPK_RE = re.compile(r"SHADOW_TOPK pos=(\d+) ids=\[([^\]]*)\] probs=\[([^\]]*)\]")
TGT_RE = re.compile(r"SHADOW_TGT base=(\d+) v=\[([^\]]*)\] drafts=\[([^\]]*)\]")


def ints(s):
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def floats(s):
    return [float(x) for x in s.replace(" ", "").split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("-o", "--out", default="shadow_topk_stats.json")
    ap.add_argument("--kmax", type=int, default=4)
    args = ap.parse_args()

    # depth -> {"n": conditioned samples, "rank_hits": [top1..topk cumulative], "miss": n}
    depths = {}
    uncond = {}
    last_topk = {}  # pos -> (ids, probs), last writer wins (chronological)
    n_tgt = n_joined = n_spine_mismatch = 0

    def bump(table, d, rank, kmax):
        e = table.setdefault(d, {"n": 0, "hits": [0] * kmax, "miss": 0})
        e["n"] += 1
        if rank is None:
            e["miss"] += 1
        else:
            for r in range(rank, kmax):
                e["hits"][r] += 1  # cumulative: hit at rank contributes to top-r>=rank

    for path in args.logs:
        with open(path, errors="ignore") as f:
            for line in f:
                m = TOPK_RE.search(line)
                if m:
                    last_topk[int(m.group(1))] = (ints(m.group(2)), floats(m.group(3)))
                    continue
                m = TGT_RE.search(line)
                if not m:
                    continue
                n_tgt += 1
                base = int(m.group(1))
                v = ints(m.group(2))
                drafts = ints(m.group(3))
                prefix_ok = True
                for i, d_tok in enumerate(drafts):
                    entry = last_topk.get(base + i)
                    if entry is None:
                        break
                    ids, _probs = entry
                    if ids and ids[0] != d_tok:
                        n_spine_mismatch += 1  # sanity: spine should be top-1
                    tgt = v[i]
                    rank = ids.index(tgt) if tgt in ids else None
                    depth = i + 1
                    bump(uncond, depth, rank, args.kmax)
                    if prefix_ok:
                        bump(depths, depth, rank, args.kmax)
                        n_joined += 1
                    prefix_ok = prefix_ok and (d_tok == tgt)

    def fmt(table):
        out = {}
        for d in sorted(table):
            e = table[d]
            n = max(e["n"], 1)
            out[d] = {
                "n": e["n"],
                **{f"top{r + 1}": round(e["hits"][r] / n, 4) for r in range(args.kmax)},
                "miss": round(e["miss"] / n, 4),
            }
        return out

    result = {
        "conditional": fmt(depths),
        "unconditional": fmt(uncond),
        "n_tgt_lines": n_tgt,
        "n_joined_conditional": n_joined,
        "n_spine_mismatch": n_spine_mismatch,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    if n_spine_mismatch > 0.01 * max(n_joined, 1):
        print(
            f"WARNING: {n_spine_mismatch} spine!=top1 mismatches — join may be "
            "misaligned (check concurrency==1 and log ordering)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Enumerate spine+hedge tree shapes and predict E / TPOT from measured
per-depth conditional top-k coverage (Phase 0 of the tree-spec plan).

Usage: tree_shape_search.py shadow_topk_stats.json [--s 112] [--per-row-pct 1.2]

Model: spine = top-1 chain of length L; at each depth d, w_d in {0..3} hedge
leaves (ranks 2..1+w_d, no children). Nodes N = L + sum(w_d) <= 7 (verify
width M = N+1 <= 8). Recursive expectation, hedge terms haircut x0.95:

  E_len(d) = s_d*(1 + E_len(d+1)) + 0.95*(h_d(1+w_d) - s_d)   [hedge = leaf]
  E = 1 + E_len(1)     (bonus token)

Depths beyond the measured max reuse the last measured conditionals
(plateau assumption). Also predicts plain chains K=3..8. Applies a verify-
width cost model S'(M) = S * (1 + per_row_pct/100 * max(0, M-3)) and ranks
by predicted TPOT = S'/E. Emits the plan's GO/NO-GO verdict.
"""
import argparse
import itertools
import json

KMAX = 4


def load(path):
    with open(path) as fh:
        stats = json.load(fh)["conditional"]
    depths = sorted(int(d) for d in stats)
    s, h = {}, {}
    for d in depths:
        e = stats[str(d)]
        s[d] = e["top1"]
        h[d] = [e[f"top{r}"] for r in range(1, KMAX + 1)]  # cumulative top-r
    dmax = max(depths)
    for d in range(dmax + 1, 9):  # plateau extrapolation
        s[d], h[d] = s[dmax], h[dmax]
    return s, h, dmax


def e_tree(spine_len, widths, s, h, haircut=0.95):
    e = 0.0
    for d in range(spine_len, 0, -1):
        w = widths[d - 1]
        hedge = haircut * (h[d][min(w, KMAX - 1)] - s[d]) if w > 0 else 0.0
        e = s[d] * (1 + e) + hedge
    return 1.0 + e


def e_chain(k, s):  # k = verify width incl bonus; k-1 drafts
    e = 0.0
    for d in range(k - 1, 0, -1):
        e = s[d] * (1 + e)
    return 1.0 + e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--s", type=float, default=112.0, help="per-step wall ms at M=3")
    ap.add_argument("--per-row-pct", type=float, default=1.2,
                    help="S growth %% per verify row beyond M=3")
    ap.add_argument("--max-nodes", type=int, default=7)
    args = ap.parse_args()

    s, h, dmax = load(args.stats)
    print(f"measured depths 1..{dmax}; s={ {d: round(s[d], 3) for d in range(1, dmax + 1)} }")

    def sprime(m):
        return args.s * (1 + args.per_row_pct / 100.0 * max(0, m - 3))

    rows = []
    for spine_len in range(1, 6):
        for widths in itertools.product(range(4), repeat=spine_len):
            nodes = spine_len + sum(widths)
            if nodes > args.max_nodes:
                continue
            m = nodes + 1
            e = e_tree(spine_len, list(widths), s, h)
            rows.append((sprime(m) / e, e, m, f"spine{spine_len}+w{list(widths)}"))
    for k in range(3, 9):
        e = e_chain(k, s)
        rows.append((sprime(k) / e, e, k, f"chain-K{k}"))

    rows.sort()
    print(f"\n{'TPOT_pred':>9} {'E_pred':>6} {'M':>2}  shape")
    for tpot, e, m, name in rows[:15]:
        print(f"{tpot:9.2f} {e:6.3f} {m:2d}  {name}")

    best_tree = min((r for r in rows if r[3].startswith("spine")), default=None)
    best_chain = min((r for r in rows if r[3].startswith("chain")), default=None)
    bt, bc = best_tree, best_chain
    print(f"\nbest tree : TPOT {bt[0]:.2f}ms E={bt[1]:.3f} {bt[3]} (M={bt[2]})")
    print(f"best chain: TPOT {bc[0]:.2f}ms E={bc[1]:.3f} {bc[3]} (M={bc[2]})")
    uplift = (bc[0] - bt[0]) / bc[0] * 100
    print(f"tree uplift over best chain: {uplift:+.1f}% TPOT")

    e_pred = bt[1]
    if e_pred >= 4.1:
        verdict = "GO — full plan (predicted TPOT <= ~29.5ms)"
    elif e_pred >= 3.5:
        verdict = "CONDITIONAL GO — proceed; Phase 3 dynamic tree required for <30ms"
    else:
        verdict = "NO-GO on tree — pivot to chain-widening (user-confirmed fallback)"
    if uplift < 8 and e_pred < 4.1:
        verdict += " | NOTE: tree uplift over chain <8% — prefer chain-widening first"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
